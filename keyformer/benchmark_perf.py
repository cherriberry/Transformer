"""Synthetic GPT-2 Medium performance benchmark for Keyformer policies."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from models.gpt2_keyformer import EngineConfig, GPT2KeyformerEngine


ROOT = Path(__file__).resolve().parents[1]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_one(model, tokenizer, config: EngineConfig, prompt_len: int, gen_len: int, warmup: int) -> dict:
    total = prompt_len + gen_len
    engine = GPT2KeyformerEngine.from_pretrained_shared(model, tokenizer, config, total_tokens=total)
    device = engine.device
    torch.manual_seed(config.seed)
    input_ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)

    for _ in range(warmup):
        engine.generate_greedy(input_ids, max_new_tokens=min(4, gen_len))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    engine.reset()
    _cuda_sync()
    t0 = time.perf_counter()
    logits = engine.prefill(input_ids)
    _cuda_sync()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    decode_ms: list[float] = []
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    for _ in range(gen_len):
        _cuda_sync()
        t1 = time.perf_counter()
        step_logits = engine.decode_step(next_token)
        _cuda_sync()
        decode_ms.append((time.perf_counter() - t1) * 1000.0)
        next_token = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)

    summary = engine.cache_summary()
    peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    mean_decode = statistics.mean(decode_ms)
    return {
        "status": "ok",
        "policy": config.policy,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "cache_ratio": config.cache_ratio,
        "recent_ratio": config.recent_ratio,
        "seed": config.seed,
        "device": str(device),
        "dtype": str(engine.dtype),
        "prefill_ms": prefill_ms,
        "prefill_tokens_per_s": prompt_len / (prefill_ms / 1000.0),
        "decode_ms_mean": mean_decode,
        "decode_ms_p50": _percentile(decode_ms, 50),
        "decode_ms_p95": _percentile(decode_ms, 95),
        "decode_tokens_per_s": 1000.0 / mean_decode,
        "cuda_peak_bytes": int(peak_mem),
        "kv_bytes_total": int(summary["kv_bytes_total"]),
        "cached_tokens_max": int(summary["cached_tokens_max"]),
        "seen_tokens": int(summary["seen_tokens"]),
        "budget": int(summary["budget"]),
        "recent_window": int(summary["recent_window"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "keyformer_experiment.yaml")
    parser.add_argument("--mode", choices=["quick", "final"], default="quick")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mode = cfg[args.mode]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" and cfg["dtype"].lower() in {"float16", "fp16"} else torch.float32
    model_name = cfg["model_name"]
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.config._attn_implementation = "eager"

    rows: list[dict] = []
    for prompt_len in mode["prompt_lengths"]:
        for gen_len in mode["generation_lengths"]:
            for cache_ratio in mode["cache_ratios"]:
                for recent_ratio in mode.get("recent_ratios", [cfg["recent_ratio_default"]]):
                    for policy in cfg["policies"]:
                        effective_ratio = 1.0 if policy == "full_kv" else cache_ratio
                        # Avoid duplicate FullKV rows across ratios.
                        if policy == "full_kv" and cache_ratio != mode["cache_ratios"][0]:
                            continue
                        for seed in mode["seeds"]:
                            tag = f"{policy} p={prompt_len} g={gen_len} r={effective_ratio} seed={seed}"
                            print(f"RUN {tag}", flush=True)
                            config = EngineConfig(
                                model_name=model_name,
                                policy=policy,
                                cache_ratio=effective_ratio,
                                recent_ratio=recent_ratio,
                                seed=seed,
                                dtype=cfg["dtype"],
                                device=device,
                                tau_init=cfg["gumbel"]["tau_init"],
                                tau_delta=cfg["gumbel"]["tau_delta"],
                            )
                            try:
                                row = run_one(model, tokenizer, config, prompt_len, gen_len, mode["warmup"])
                                row["cache_ratio_requested"] = cache_ratio
                            except Exception as exc:  # noqa: BLE001
                                row = {
                                    "status": "error",
                                    "policy": policy,
                                    "prompt_len": prompt_len,
                                    "gen_len": gen_len,
                                    "cache_ratio": effective_ratio,
                                    "recent_ratio": recent_ratio,
                                    "seed": seed,
                                    "error": repr(exc),
                                }
                                print(f"FAIL {tag}: {exc}", flush=True)
                            rows.append(row)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

    full_index = {
        (r["prompt_len"], r["gen_len"], r["seed"]): r
        for r in rows
        if r.get("status") == "ok" and r["policy"] == "full_kv"
    }
    for row in rows:
        if row.get("status") != "ok":
            continue
        baseline = full_index.get((row["prompt_len"], row["gen_len"], row["seed"]))
        if baseline and baseline["decode_ms_mean"] > 0:
            row["speedup_vs_fullkv"] = baseline["decode_ms_mean"] / row["decode_ms_mean"]
            row["kv_bytes_saving_vs_fullkv"] = 1.0 - (row["kv_bytes_total"] / max(1, baseline["kv_bytes_total"]))
            if baseline["cuda_peak_bytes"] > 0:
                row["peak_mem_saving_vs_fullkv"] = 1.0 - (row["cuda_peak_bytes"] / baseline["cuda_peak_bytes"])

    out = args.output or (ROOT / "results" / "synthetic" / f"perf_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "mode": args.mode,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
