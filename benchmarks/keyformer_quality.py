"""WikiText-2 perplexity evaluation with physically compressed KV caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from models.gpt2_keyformer import EngineConfig, GPT2KeyformerEngine


ROOT = Path(__file__).resolve().parents[1]


def load_wikitext_tokens(tokenizer, max_tokens: int) -> torch.Tensor:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([row["text"] for row in dataset if row["text"] and row["text"].strip()])
    # Avoid HF tokenizer length warnings by capping characters before encoding.
    approx_chars = max_tokens * 6
    tokens = tokenizer(text[:approx_chars], return_tensors="pt", truncation=False).input_ids[0]
    return tokens[:max_tokens]


@torch.no_grad()
def evaluate_policy(model, tokenizer, config: EngineConfig, tokens: torch.Tensor, context: int) -> dict:
    """Sliding-window teacher-forced PPL; each window is at most ``context`` tokens."""
    engine = GPT2KeyformerEngine.from_pretrained_shared(model, tokenizer, config, total_tokens=context)
    device = engine.device
    tokens = tokens.to(device)

    nll_sum = 0.0
    token_count = 0
    top1_correct = 0

    for start in range(0, max(0, tokens.numel() - 1), context):
        window = tokens[start : start + context]
        if window.numel() < 2:
            break
        logits = engine.prefill(window[:1].view(1, 1))
        for index in range(1, window.numel()):
            target = int(window[index].item())
            log_probs = F.log_softmax(logits[0, -1, :].float(), dim=-1)
            nll_sum += -float(log_probs[target].item())
            top1_correct += int(int(torch.argmax(log_probs).item()) == target)
            token_count += 1
            logits = engine.decode_step(window[index : index + 1].view(1, 1))

    mean_nll = nll_sum / max(1, token_count)
    return {
        "status": "ok",
        "policy": config.policy,
        "cache_ratio": config.cache_ratio,
        "recent_ratio": config.recent_ratio,
        "seed": config.seed,
        "context": context,
        "tokens_evaluated": token_count,
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "top1_accuracy": top1_correct / max(1, token_count),
        "budget": engine.budget,
        "recent_window": engine.recent_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "keyformer_experiment.yaml")
    parser.add_argument("--mode", choices=["quality_quick", "quality_final"], default="quality_quick")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mode = cfg[args.mode]
    model_name = cfg["model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" and cfg["dtype"].lower() in {"float16", "fp16"} else torch.float32

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.config._attn_implementation = "eager"

    tokens = load_wikitext_tokens(tokenizer, max_tokens=int(mode["max_tokens"]))
    contexts = mode.get("contexts") or [mode["context"]]
    rows: list[dict] = []

    for context in contexts:
        for cache_ratio in mode["cache_ratios"]:
            for policy in cfg["policies"]:
                effective_ratio = 1.0 if policy == "full_kv" else cache_ratio
                if policy == "full_kv" and cache_ratio != mode["cache_ratios"][0]:
                    continue
                for seed in mode["seeds"]:
                    tag = f"{policy} ctx={context} r={effective_ratio} seed={seed}"
                    print(f"RUN {tag}", flush=True)
                    config = EngineConfig(
                        model_name=model_name,
                        policy=policy,
                        cache_ratio=effective_ratio,
                        recent_ratio=cfg["recent_ratio_default"],
                        seed=seed,
                        dtype=cfg["dtype"],
                        device=device,
                        tau_init=cfg["gumbel"]["tau_init"],
                        tau_delta=cfg["gumbel"]["tau_delta"],
                    )
                    try:
                        row = evaluate_policy(model, tokenizer, config, tokens.clone(), context)
                        row["cache_ratio_requested"] = cache_ratio
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "status": "error",
                            "policy": policy,
                            "cache_ratio": effective_ratio,
                            "seed": seed,
                            "context": context,
                            "error": repr(exc),
                        }
                        print(f"FAIL {tag}: {exc}", flush=True)
                    rows.append(row)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    full = {
        (r["context"], r["seed"]): r
        for r in rows
        if r.get("status") == "ok" and r["policy"] == "full_kv"
    }
    for row in rows:
        if row.get("status") != "ok":
            continue
        baseline = full.get((row["context"], row["seed"]))
        if baseline:
            row["ppl_increase_vs_fullkv"] = row["perplexity"] / baseline["perplexity"] - 1.0
            row["nll_increase_vs_fullkv"] = row["nll"] - baseline["nll"]

    out = args.output or (ROOT / "results" / "wikitext2" / f"{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "mode": args.mode,
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "max_tokens_requested": mode["max_tokens"],
        "actual_tokens": int(tokens.numel()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
