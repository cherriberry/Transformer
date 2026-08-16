from __future__ import annotations

import pytest
import torch

from models.keyformer import (
    CausalSelfAttentionWithKV,
    FullKVPolicy,
    KeyformerPolicy,
    KVCacheState,
    RandomKVPolicy,
    RecentWindowPolicy,
    cache_metrics,
    output_error_metrics,
)


def _state(scores: torch.Tensor) -> KVCacheState:
    """Create a cache whose key/value data encodes original positions."""
    batch, heads, tokens = scores.shape
    positions = torch.arange(tokens).view(1, 1, tokens).expand(batch, heads, tokens)
    vectors = positions.to(torch.float32).unsqueeze(-1).expand(batch, heads, tokens, 2).clone()
    return KVCacheState(vectors, vectors + 100, positions, scores.clone())


def _identity_attention(policy, dim: int = 8, heads: int = 2) -> CausalSelfAttentionWithKV:
    model = CausalSelfAttentionWithKV(dim, heads=heads, policy=policy)
    with torch.no_grad():
        for projection in (model.to_q, model.to_k, model.to_v):
            projection.weight.copy_(torch.eye(dim))
        model.to_out.weight.copy_(torch.eye(dim))
        model.to_out.bias.zero_()
    return model


def test_full_kv_prefill_and_decode_match_exact_full_causal_attention() -> None:
    torch.manual_seed(3)
    model = _identity_attention(FullKVPolicy())
    tokens = torch.randn(2, 7, 8)
    expected = model.forward_full(tokens)

    prompt_output, state = model.prefill(tokens[:, :4])
    decoded = []
    for index in range(4, 7):
        output, state = model.decode_step(tokens[:, index : index + 1], state)
        decoded.append(output)

    actual = torch.cat((prompt_output, *decoded), dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert state.num_tokens == 7
    assert state.positions[0, 0].tolist() == list(range(7))


def test_full_kv_cache_grows_one_token_per_decode_step() -> None:
    model = CausalSelfAttentionWithKV(8, heads=2, policy=FullKVPolicy())
    _, state = model.prefill(torch.randn(1, 3, 8))
    sizes = [state.num_tokens]
    for _ in range(4):
        _, state = model.decode_step(torch.randn(1, 1, 8), state)
        sizes.append(state.num_tokens)
    assert sizes == [3, 4, 5, 6, 7]


def test_recent_window_keeps_only_newest_original_positions() -> None:
    policy = RecentWindowPolicy(budget=4)
    selected = policy.select(_state(torch.zeros(1, 2, 8)))
    assert selected.num_tokens == 4
    assert selected.positions[0, 0].tolist() == [4, 5, 6, 7]
    assert selected.positions[0, 1].tolist() == [4, 5, 6, 7]


def test_keyformer_keeps_high_score_old_tokens_and_recent_window() -> None:
    scores = torch.tensor([[[0.1, 9.0, 0.2, 8.0, 0.3, 0.4, 0.5, 0.6]]])
    selected = KeyformerPolicy(
        budget=4, recent_window=2, gumbel_noise=False
    ).select(_state(scores))
    assert selected.positions[0, 0].tolist() == [1, 3, 6, 7]
    assert selected.keys[0, 0, :, 0].tolist() == [1.0, 3.0, 6.0, 7.0]


def test_keyformer_selection_is_independent_per_head() -> None:
    scores = torch.tensor(
        [[
            [9.0, 8.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 8.0, 0.0, 0.0],
        ]]
    )
    selected = KeyformerPolicy(
        budget=4, recent_window=2, gumbel_noise=False
    ).select(_state(scores))
    assert selected.positions[0, 0].tolist() == [0, 1, 4, 5]
    assert selected.positions[0, 1].tolist() == [2, 3, 4, 5]


def test_keyformer_cache_stays_at_budget_and_protects_recent_tokens() -> None:
    policy = KeyformerPolicy(budget=5, recent_window=2, gumbel_noise=False)
    model = CausalSelfAttentionWithKV(12, heads=3, policy=policy)
    _, state = model.prefill(torch.randn(2, 9, 12))
    assert state.num_tokens == 5
    assert state.positions[0, 0, -2:].tolist() == [7, 8]

    for expected_position in range(9, 14):
        _, state = model.decode_step(torch.randn(2, 1, 12), state)
        assert state.num_tokens == 5
        assert state.positions[0, 0, -2:].tolist() == [expected_position - 1, expected_position]


def test_absolute_positions_advance_even_without_a_recent_window() -> None:
    policy = KeyformerPolicy(
        budget=2, recent_window=0, gumbel_noise=False
    )
    model = CausalSelfAttentionWithKV(4, heads=1, policy=policy)
    _, state = model.prefill(torch.randn(1, 5, 4))
    assert state.seen_tokens == 5
    for expected_position in range(5, 9):
        _, state = model.decode_step(torch.randn(1, 1, 4), state)
        assert state.seen_tokens == expected_position + 1
        assert state.positions.max().item() <= expected_position
        assert torch.unique(state.positions).numel() == state.num_tokens


def test_score_accumulation_and_temperature_schedule() -> None:
    policy = KeyformerPolicy(
        budget=8, recent_window=2, tau_init=1.0, tau_delta=0.25, gumbel_noise=False
    )
    model = CausalSelfAttentionWithKV(4, heads=1, policy=policy)
    with torch.no_grad():
        model.to_q.weight.zero_()
        model.to_k.weight.zero_()
    _, state = model.prefill(torch.zeros(1, 3, 4))
    old_scores = state.scores.clone()
    _, state = model.decode_step(torch.zeros(1, 1, 4), state)
    assert policy.temperature(0) == pytest.approx(1.0)
    assert policy.temperature(4) == pytest.approx(2.0)
    assert torch.all(state.scores[..., :3] > old_scores)
    assert state.decode_step == 1


def test_current_token_participates_before_cache_eviction() -> None:
    full = _identity_attention(FullKVPolicy(), dim=4, heads=1)
    compressed = _identity_attention(
        RecentWindowPolicy(budget=1), dim=4, heads=1
    )
    compressed.load_state_dict(full.state_dict())
    prompt = torch.zeros(1, 2, 4)
    _, full_state = full.prefill(prompt)
    _, compressed_state = compressed.prefill(prompt)
    new_token = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]])
    full_output, _ = full.decode_step(new_token, full_state)
    compressed_output, compressed_state = compressed.decode_step(new_token, compressed_state)
    # With a one-token recent cache, the prompt was already compressed, but the
    # current token is still appended and attended before the next eviction.
    assert compressed_state.positions[0, 0].item() == 2
    assert compressed_output[0, 0, 0].item() > 0
    assert full_output.shape == compressed_output.shape


def test_random_baseline_is_reproducible_and_keeps_recent_tokens() -> None:
    state = _state(torch.zeros(1, 2, 10))
    first = RandomKVPolicy(5, recent_window=2, seed=42).select(state)
    second = RandomKVPolicy(5, recent_window=2, seed=42).select(state)
    assert torch.equal(first.positions, second.positions)
    assert first.positions[0, 0, -2:].tolist() == [8, 9]
    assert first.num_tokens == 5


def test_gumbel_scores_are_reproducible_for_equal_seed() -> None:
    logits = torch.randn(2, 3, 1, 7)
    first = KeyformerPolicy(4, 2, seed=11).score_probabilities(logits, 2)
    second = KeyformerPolicy(4, 2, seed=11).score_probabilities(logits, 2)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.sum(dim=-1), torch.ones_like(first.sum(dim=-1)))


def test_cache_and_output_metrics() -> None:
    state = _state(torch.zeros(1, 2, 4))
    metrics = cache_metrics(state, full_token_count=8)
    assert metrics["cached_tokens"] == 4
    assert metrics["compression_ratio"] == pytest.approx(0.5)
    assert metrics["kv_bytes"] == state.storage_bytes

    reference = torch.tensor([1.0, 2.0, 3.0])
    identical = output_error_metrics(reference, reference.clone())
    assert identical["mse"] == pytest.approx(0.0)
    assert identical["relative_l2"] == pytest.approx(0.0)
    assert identical["cosine_similarity"] == pytest.approx(1.0)


def test_invalid_configuration_and_state_shapes() -> None:
    with pytest.raises(ValueError, match="recent_window"):
        KeyformerPolicy(budget=4, recent_window=5)
    with pytest.raises(ValueError, match="tau_init"):
        KeyformerPolicy(budget=4, recent_window=2, tau_init=0)
    with pytest.raises(ValueError, match="positive"):
        RecentWindowPolicy(0)
    bad = KVCacheState(
        torch.zeros(1, 2, 3, 4),
        torch.zeros(1, 2, 3, 4),
        torch.zeros(1, 3),
        torch.zeros(1, 2, 3),
    )
    with pytest.raises(ValueError, match="positions"):
        bad.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_prefill_and_decode_smoke() -> None:
    model = CausalSelfAttentionWithKV(
        64,
        heads=4,
        policy=KeyformerPolicy(16, recent_window=4, seed=5),
    ).cuda()
    with torch.inference_mode():
        output, state = model.prefill(torch.randn(2, 32, 64, device="cuda"))
        decoded, state = model.decode_step(torch.randn(2, 1, 64, device="cuda"), state)
    assert output.is_cuda and decoded.is_cuda and state.keys.is_cuda
    assert state.num_tokens == 16
    assert torch.isfinite(decoded).all()
