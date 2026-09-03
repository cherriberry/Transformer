"""Benchmark efficient attention implementations and full attention.

This runner uses the repository's ``universal_benchmark.py`` interface.  The
default configuration is intentionally small enough for an 8 GB GPU, while
the sequence lengths and run count can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from models import LongformerSelfAttention

ROOT = Path(__file__).resolve().parent


def _add_source(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


COMPAT_ROOT = ROOT / "compat_deps"
for _dependency in ("einops", "local_attention", "axial_positional_embedding", "product_key_memory"):
    if importlib.util.find_spec(_dependency) is None:
        _add_source(COMPAT_ROOT)
        break

PERFORMER_ROOT = ROOT / "performer-pytorch"
REFORMER_ROOT = ROOT / "reformer-pytorch"
MEMFORMER_ROOT = ROOT.parent / "memformers"
_add_source(PERFORMER_ROOT)
_add_source(REFORMER_ROOT)


class PerformerAdapter(nn.Module):
    """Shape adapter around the Performer PyTorch implementation."""

    def __init__(self, dim: int, seq_len: int, heads: int = 8, depth: int = 1):
        super().__init__()
        from performer_pytorch import Performer

        self.model = Performer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim // heads,
            causal=False,
            local_attn_heads=0,
            nb_features=min(256, dim * 2),
            reversible=False,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x)


class ReformerAdapter(nn.Module):
    """Shape adapter around the Reformer LSH implementation."""

    def __init__(
        self,
        dim: int,
        seq_len: int,
        heads: int = 8,
        depth: int = 1,
        bucket_size: int = 64,
        n_hashes: int = 2,
    ):
        super().__init__()
        from reformer_pytorch import Reformer

        if seq_len % (bucket_size * 2) != 0:
            raise ValueError(
                "Reformer benchmark lengths must be divisible by "
                f"2 * bucket_size={bucket_size * 2}; "
                f"got seq_len={seq_len}"
            )
        self.model = Reformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim // heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=False,
            ff_chunks=1,
            attn_chunks=1,
            use_full_attn=False,
            full_attn_thres=0,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x)


def _load_memformer_attention():
    """Load the standalone attention module without importing Transformers."""
    path = MEMFORMER_ROOT / "memformers" / "models" / "membart" / "membart_attention.py"
    spec = importlib.util.spec_from_file_location("memformer_attention_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Memformer source from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MemBartEncoderAttention


class MemformerAdapter(nn.Module):
    """Memformer encoder attention with persistent external memory slots."""

    def __init__(
        self,
        dim: int,
        seq_len: int,
        heads: int = 8,
        memory_slots: int = 64,
    ):
        super().__init__()
        attention = _load_memformer_attention()
        self.attention = attention(embed_dim=dim, num_heads=heads)
        self.memory = nn.Parameter(torch.randn(1, memory_slots, dim) * 0.02)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        memory = self.memory.expand(x.shape[0], -1, -1)
        output, _, _ = self.attention(x, memory_states=memory)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Sequence lengths; Reformer lengths must be divisible by 128.",
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument(
        "--longformer-window",
        type=int,
        default=65,
        help="Odd total local window width (default: 65).",
    )
    parser.add_argument("--longformer-global-tokens", type=int, default=1)
    parser.add_argument("--output", default="benchmark_results.png")
    parser.add_argument("--json-output", default="benchmark_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from universal_benchmark import (
        StandardTransformerSelfAttention,
        UniversalBenchmark,
        get_environment_metadata,
    )

    common = {"dim": args.dim, "heads": args.heads}
    models = {
        "Standard": {"class": StandardTransformerSelfAttention, "params": common},
        "Longformer-Local": {
            "class": LongformerSelfAttention,
            "params": {**common, "window_size": args.longformer_window, "global_tokens": 0},
        },
        "Longformer-Global": {
            "class": LongformerSelfAttention,
            "params": {
                **common,
                "window_size": args.longformer_window,
                "global_tokens": args.longformer_global_tokens,
            },
        },
        "Memformer": {"class": MemformerAdapter, "params": common},
        "Performer": {"class": PerformerAdapter, "params": common},
        "Reformer": {"class": ReformerAdapter, "params": common},
    }

    benchmark = UniversalBenchmark("Efficient attention comparison")
    results = {}
    for name, config in models.items():
        result = benchmark.benchmark_model(
            config["class"],
            config["params"],
            args.seq_lengths,
            num_runs=args.runs,
            warmup_runs=args.warmup_runs,
        )
        results[name] = {
            "times": result.times,
            "memories": result.memories,
            "records": result.records,
        }

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    benchmark.plot_comparison(results, args.seq_lengths, args.output)
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "seq_lengths": args.seq_lengths,
                    "dim": args.dim,
                    "heads": args.heads,
                    "batch_size": 1,
                    "runs": args.runs,
                    "warmup_runs": args.warmup_runs,
                    "longformer_window": args.longformer_window,
                    "longformer_global_tokens": args.longformer_global_tokens,
                },
                "environment": get_environment_metadata(),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nResults:")
    for name, result in results.items():
        print(f"{name}: times={result['times']}, memories_mb={result['memories']}")


if __name__ == "__main__":
    main()
