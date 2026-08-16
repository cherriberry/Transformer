"""CUDA benchmark of KV-cache policies on GPT-2 Medium and WikiText-2.

The runner deliberately uses ``attn_implementation=eager``: this makes GPT-2
return per-head attention probabilities, which are required to reproduce
Keyformer's accumulated Gumbel score.  It measures real model prefill/decode,
not the small algorithm-only attention module used by unit tests.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.keyformer import (
    FullKVPolicy, KeyformerPolicy, KVCacheState, RandomKVPolicy,
    RecentWindowPolicy, cache_metrics,
)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError("YAML configs require PyYAML; install requirements-keyformer.txt") from error
            cfg = yaml.safe_load(handle)
        else:
            cfg = json.load(handle)
    required = {"model", "dataset", "dtype", "decode_tokens", "prompt_lengths", "cache_ratios", "seeds", "policies"}
    missing = required.difference(cfg)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_policy(name: str, budget: int, recent: int, cfg: dict[str, Any], seed: int):
    if name == "full_kv":
        return FullKVPolicy()
    if name == "keyformer":
        options = cfg["keyformer"]
        return KeyformerPolicy(budget, recent, seed=seed, **options)
    if name == "recent_window":
        return RecentWindowPolicy(budget)
    if name == "random_plus_recent":
        return RandomKVPolicy(budget, recent, seed=seed)
    raise ValueError(f"unknown policy: {name}")


def _as_legacy_cache(past: Any) -> tuple:
    """Accept transformers tuple caches, including DynamicCache-compatible ones."""
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    return tuple((layer[0], layer[1]) for layer in past)


def _model_cache(legacy: tuple, config):
    """Wrap selected tensors in the current Transformers cache API."""
    try:
        from transformers.cache_utils import DynamicCache
        return DynamicCache(legacy, config=config)
    except ImportError:  # Transformers 4.x accepts the legacy tuple directly.
        return legacy


def _compress(past: Any, attentions: tuple, policies: list, states: list[KVCacheState | None], *, prompt: bool) -> tuple[tuple, list[KVCacheState]]:
    legacy = _as_legacy_cache(past)
    new_states: list[KVCacheState] = []
    compressed = []
    for layer, ((keys, values), attention, policy, old) in enumerate(zip(legacy, attentions, policies, states)):
        # attentions are normalized probabilities from the actual GPT-2 layer.
        # Recover normalized logits from attention probabilities. Exact zeros
        # are causal-mask entries and must remain excluded before Gumbel noise.
        probs = attention.float()
        logits = torch.where(
            probs > 0,
            probs.clamp_min(torch.finfo(torch.float32).tiny).log(),
            torch.full_like(probs, torch.finfo(torch.float32).min),
        )
        if prompt:
            score = policy.score_probabilities(logits, 0).sum(dim=-2)
            n = keys.shape[-2]
            positions = torch.arange(n, device=keys.device).view(1, 1, n).expand(keys.shape[0], keys.shape[1], n)
            state = KVCacheState(keys, values, positions, score, seen_tokens=n)
        else:
            assert old is not None
            score = policy.score_probabilities(logits, old.decode_step + 1).squeeze(-2)
            state = KVCacheState(
                keys, values,
                torch.cat((old.positions, torch.full_like(old.positions[..., :1], old.seen_tokens)), dim=-1),
                torch.cat((old.scores, torch.zeros_like(score[..., :1])), dim=-1) + score,
                decode_step=old.decode_step + 1, seen_tokens=old.seen_tokens + 1,
            )
        state = policy.select(state)
        new_states.append(state)
        compressed.append((state.keys, state.values))
    return tuple(compressed), new_states


def _timed_forward(model, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = model(**kwargs)
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def get_prompt(tokenizer, dataset_name: str, dataset_config: str | None, length: int, offset: int) -> torch.Tensor:
    from datasets import load_dataset
    data = load_dataset(dataset_name, dataset_config, split="test")
    # WikiText paragraphs vary in length. Concatenating a deterministic suffix
    # avoids silently skipping requested prompt lengths such as 256/1024.
    text = "\n".join(row["text"] for row in data.select(range(offset, len(data))) if row["text"].strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] < length:
        raise RuntimeError(f"WikiText sample has only {ids.shape[1]} tokens; need {length}")
    return ids[:, :length]


def environment() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "device_capability": torch.cuda.get_device_capability(0)}


def run_case(model, tokenizer, cfg, policy_name: str, prompt_length: int, ratio: float, seed: int, offset: int) -> dict[str, Any]:
    set_seed(seed)
    ids = get_prompt(tokenizer, cfg["dataset"], cfg.get("dataset_config"), prompt_length, offset).cuda()
    budget = max(1, round(prompt_length * ratio))
    recent = min(budget, round(budget * cfg.get("recent_window_ratio", 0.5)))
    policies = [build_policy(policy_name, budget, recent, cfg, seed + layer) for layer in range(model.config.n_layer)]
    with torch.inference_mode():
        prompt_positions = torch.arange(prompt_length, device=ids.device).unsqueeze(0)
        output, prefill_s = _timed_forward(
            model, input_ids=ids, position_ids=prompt_positions, use_cache=True, output_attentions=True
        )
        past, states = _compress(output.past_key_values, output.attentions, policies, [None] * len(policies), prompt=True)
        token = output.logits[:, -1:].argmax(dim=-1)
        decode_s = 0.0
        generated = []
        for step in range(cfg["decode_tokens"]):
            generated.append(int(token.item()))
            position = torch.full((1, 1), prompt_length + step, device=ids.device, dtype=torch.long)
            output, elapsed = _timed_forward(
                model, input_ids=token, position_ids=position, past_key_values=_model_cache(past, model.config),
                use_cache=True, output_attentions=True,
            )
            decode_s += elapsed
            past, states = _compress(output.past_key_values, output.attentions, policies, states, prompt=False)
            token = output.logits[:, -1:].argmax(dim=-1)
    full_tokens = prompt_length + cfg["decode_tokens"]
    per_layer = [cache_metrics(s, full_tokens) for s in states]
    return {"policy": policy_name, "seed": seed, "prompt_length": prompt_length, "cache_ratio": ratio,
            "cache_budget": budget, "recent_window": recent, "prefill_seconds": prefill_s,
            "decode_seconds": decode_s, "decode_ms_per_token": 1000 * decode_s / cfg["decode_tokens"],
            "generated_token_ids": generated, "cache": {"mean_compression_ratio": float(np.mean([x["compression_ratio"] for x in per_layer])),
                      "mean_kv_bytes": float(np.mean([x["kv_bytes"] for x in per_layer])), "per_layer": per_layer}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install benchmark dependencies: pip install -r requirements-keyformer.txt") from error
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg["dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype, attn_implementation="eager").cuda().eval()
    results = {"config": cfg, "environment": environment(), "results": []}
    output = Path(args.output or f"benchmark_results/keyformer_{cfg.get('name', 'run')}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    for seed in cfg["seeds"]:
        for prompt_length in cfg["prompt_lengths"]:
            for ratio in cfg["cache_ratios"]:
                for name in cfg["policies"]:
                    print(f"{name}: prompt={prompt_length}, cache={ratio:.0%}, seed={seed}", flush=True)
                    results["results"].append(run_case(model, tokenizer, cfg, name, prompt_length, ratio, seed, offset=seed))
                    # Persist progress: long eager-attention runs remain
                    # inspectable if interrupted after a completed case.
                    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
