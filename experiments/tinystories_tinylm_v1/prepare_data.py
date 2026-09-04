"""Download and audit pinned TinyStories data plus the GPT-2 tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER_ID = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATASET_PATTERNS = ["README.md", ".gitattributes", "data/*.parquet"]
TOKENIZER_PATTERNS = [
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if ".cache" in path.parts:
            continue
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face-compatible endpoint; set before library imports.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "raw" / "tinystories"
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=ROOT / "data" / "tokenizer" / "gpt2"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "tinystories_manifest.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    data_dir = args.data_dir.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=data_dir,
        allow_patterns=DATASET_PATTERNS,
        max_workers=4,
    )
    snapshot_download(
        repo_id=TOKENIZER_ID,
        repo_type="model",
        revision=TOKENIZER_REVISION,
        local_dir=tokenizer_dir,
        allow_patterns=TOKENIZER_PATTERNS,
        max_workers=4,
    )

    data_files = {
        "train": str(data_dir / "data" / "train-*.parquet"),
        "validation": str(data_dir / "data" / "validation-*.parquet"),
    }
    dataset = load_dataset("parquet", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if set(dataset) != {"train", "validation"}:
        raise RuntimeError(f"unexpected TinyStories splits: {sorted(dataset)}")
    if "text" not in dataset["train"].column_names:
        raise RuntimeError("TinyStories train split has no 'text' column")
    if len(tokenizer) != 50257 or tokenizer.eos_token_id != 50256:
        raise RuntimeError("downloaded tokenizer is not the expected GPT-2 tokenizer")

    api = HfApi(endpoint=args.endpoint)
    dataset_sha = api.dataset_info(DATASET_ID, revision=DATASET_REVISION).sha
    tokenizer_sha = api.model_info(TOKENIZER_ID, revision=TOKENIZER_REVISION).sha
    if dataset_sha != DATASET_REVISION or tokenizer_sha != TOKENIZER_REVISION:
        raise RuntimeError("resolved upstream revision does not match the pinned commit")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_endpoint": args.endpoint,
        "dataset": {
            "id": DATASET_ID,
            "revision": dataset_sha,
            "license": "cdla-sharing-1.0",
            "splits": {name: {"rows": len(split)} for name, split in dataset.items()},
            "columns": dataset["train"].column_names,
            "files": inventory(data_dir),
        },
        "tokenizer": {
            "id": TOKENIZER_ID,
            "revision": tokenizer_sha,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "files": inventory(tokenizer_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "datasets": __import__("datasets").__version__,
            "transformers": __import__("transformers").__version__,
            "huggingface_hub": __import__("huggingface_hub").__version__,
        },
        "notes": [
            "TinyStories has no official test split.",
            "No GPT-2 model weights were downloaded.",
            "Packing and evaluation split policy are intentionally not frozen yet.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

