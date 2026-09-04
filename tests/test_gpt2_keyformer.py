from __future__ import annotations

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from models.gpt2_keyformer import (
    EngineConfig,
    GPT2KeyformerEngine,
    compute_budget,
    max_abs_logit_error,
    stock_greedy_tokens,
)


@pytest.fixture(scope="module")
def gpt2_medium_cpu():
    name = "openai-community/gpt2-medium"
    tokenizer = GPT2TokenizerFast.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


def _engine(policy: str, total_tokens: int, cache_ratio: float = 1.0, seed: int = 0, device: str = "cpu"):
    return GPT2KeyformerEngine.from_config(
        EngineConfig(
            model_name="openai-community/gpt2-medium",
            policy=policy,
            cache_ratio=cache_ratio,
            recent_ratio=0.5,
            seed=seed,
            dtype="float32",
            device=device,
        ),
        total_tokens=total_tokens,
    )


def test_compute_budget_ratios():
    assert compute_budget(100, 1.0, 0.5) == (100, 50)
    assert compute_budget(100, 0.5, 0.5) == (50, 25)
    assert compute_budget(10, 0.25, 0.5)[0] == 3


@torch.no_grad()
def test_fullkv_prefill_matches_stock_logits(gpt2_medium_cpu):
    stock, tokenizer = gpt2_medium_cpu
    text = "Keyformer compresses the KV cache during autoregressive decoding."
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    stock_logits = stock(input_ids).logits
    engine = _engine("full_kv", total_tokens=input_ids.shape[1] + 8)
    ours = engine.prefill(input_ids)
    err = max_abs_logit_error(stock_logits, ours)
    assert err < 5e-4, f"max abs logit error too large: {err}"


@torch.no_grad()
def test_fullkv_prefill_plus_decode_matches_full_forward(gpt2_medium_cpu):
    stock, tokenizer = gpt2_medium_cpu
    prompt_ids = tokenizer(
        "The KV cache grows with every generated token during decoding.",
        return_tensors="pt",
    ).input_ids
    stock_tokens = stock_greedy_tokens(stock, prompt_ids, max_new_tokens=16)
    engine = _engine("full_kv", total_tokens=prompt_ids.shape[1] + 16)
    ours_tokens = engine.generate_greedy(prompt_ids, max_new_tokens=16)
    assert torch.equal(stock_tokens, ours_tokens)


@torch.no_grad()
def test_keyformer_100_percent_matches_fullkv_tokens():
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2-medium")
    prompt_ids = tokenizer("Compressing attention caches should be exact at full budget.", return_tensors="pt").input_ids
    total = prompt_ids.shape[1] + 32
    full = _engine("full_kv", total_tokens=total, cache_ratio=1.0, seed=0)
    kf = _engine("keyformer", total_tokens=total, cache_ratio=1.0, seed=0)
    full_tokens = full.generate_greedy(prompt_ids, max_new_tokens=32)
    kf_tokens = kf.generate_greedy(prompt_ids, max_new_tokens=32)
    assert torch.equal(full_tokens, kf_tokens)


@torch.no_grad()
def test_compressed_policies_respect_budget_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2-medium")
    prompt_ids = tokenizer("A somewhat longer prompt for cache pressure " * 8, return_tensors="pt").input_ids
    prompt_ids = prompt_ids[:, :128]
    gen = 64
    total = prompt_ids.shape[1] + gen
    for policy in ("keyformer", "recent_window", "random_plus_recent"):
        engine = _engine(policy, total_tokens=total, cache_ratio=0.5, seed=1, device="cuda")
        engine.generate_greedy(prompt_ids.to("cuda"), max_new_tokens=gen)
        summary = engine.cache_summary()
        assert summary["cached_tokens_max"] <= engine.budget
        assert summary["seen_tokens"] == total
        # Original positions remain chronological and within seen range.
        for cache in engine.caches:
            pos = cache.positions[0, 0].tolist()
            assert pos == sorted(pos)
            assert max(pos) < summary["seen_tokens"]


@torch.no_grad()
def test_random_policy_is_seed_reproducible():
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2-medium")
    prompt_ids = tokenizer("Reproducible random cache eviction.", return_tensors="pt").input_ids
    total = prompt_ids.shape[1] + 24
    a = _engine("random_plus_recent", total_tokens=total, cache_ratio=0.5, seed=123)
    b = _engine("random_plus_recent", total_tokens=total, cache_ratio=0.5, seed=123)
    c = _engine("random_plus_recent", total_tokens=total, cache_ratio=0.5, seed=999)
    tokens_a = a.generate_greedy(prompt_ids, max_new_tokens=24)
    tokens_b = b.generate_greedy(prompt_ids, max_new_tokens=24)
    tokens_c = c.generate_greedy(prompt_ids, max_new_tokens=24)
    assert torch.equal(tokens_a, tokens_b)
    assert not torch.equal(tokens_a, tokens_c)
