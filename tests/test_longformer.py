from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from models.longformer_attention import LongformerSelfAttention
from universal_benchmark import StandardTransformerSelfAttention


def _copy_attention_weights(source, target) -> None:
    for name in ("to_q", "to_k", "to_v", "to_out"):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())


def test_output_shape_and_finite_values() -> None:
    model = LongformerSelfAttention(32, seq_len=37, heads=4, window_size=9, global_tokens=1)
    x = torch.randn(2, 37, 32)
    output = model(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_non_multiple_sequence_and_padding_mask() -> None:
    model = LongformerSelfAttention(24, seq_len=19, heads=3, window_size=7, global_tokens=2)
    x = torch.randn(2, 19, 24)
    mask = torch.ones(2, 19, dtype=torch.bool)
    mask[0, 13:] = False
    output = model(x, attention_mask=mask)
    assert output.shape == x.shape
    assert torch.count_nonzero(output[0, 13:]) == 0


def test_full_window_matches_standard_attention() -> None:
    torch.manual_seed(7)
    sequence = 9
    standard = StandardTransformerSelfAttention(32, sequence, heads=4, dropout=0.0)
    longformer = LongformerSelfAttention(
        32, sequence, heads=4, window_size=2 * sequence - 1, global_tokens=0, dropout=0.0
    )
    _copy_attention_weights(standard, longformer)
    x = torch.randn(2, sequence, 32)
    torch.testing.assert_close(longformer(x), standard(x), atol=1e-6, rtol=1e-5)


def test_global_query_can_read_distant_token() -> None:
    model = LongformerSelfAttention(4, seq_len=9, heads=1, window_size=3, global_tokens=1)
    with torch.no_grad():
        model.to_q.weight.zero_()
        model.to_k.weight.zero_()
        model.to_v.weight.copy_(torch.eye(4))
        model.to_out.weight.copy_(torch.eye(4))
        model.to_out.bias.zero_()

    x = torch.zeros(1, 9, 4)
    x[0, -1, 0] = 9.0
    output = model(x)
    assert output[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert output[0, 4, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_local_query_attends_to_global_key_once() -> None:
    model = LongformerSelfAttention(4, seq_len=9, heads=1, window_size=3, global_tokens=1)
    with torch.no_grad():
        model.to_q.weight.zero_()
        model.to_k.weight.zero_()
        model.to_v.weight.copy_(torch.eye(4))
        model.to_out.weight.copy_(torch.eye(4))
        model.to_out.bias.zero_()

    x = torch.zeros(1, 9, 4)
    x[0, 0, 0] = 3.0
    output = model(x)
    # Middle token has three local keys plus exactly one global key.
    assert output[0, 4, 0].item() == pytest.approx(0.75, abs=1e-6)


def test_invalid_configuration_and_mask_shape() -> None:
    with pytest.raises(ValueError, match="odd"):
        LongformerSelfAttention(16, seq_len=8, heads=4, window_size=8)
    model = LongformerSelfAttention(16, seq_len=8, heads=4, window_size=5)
    with pytest.raises(ValueError, match="attention_mask"):
        model(torch.randn(1, 8, 16), attention_mask=torch.ones(8))


def test_backward_is_finite() -> None:
    model = LongformerSelfAttention(16, seq_len=11, heads=4, window_size=5, global_tokens=1)
    x = torch.randn(2, 11, 16, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_implementation_does_not_form_sequence_by_sequence_einsum() -> None:
    source = Path("models/longformer_attention.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    einsum_equations = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "einsum"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert all("bhnm" not in equation for equation in einsum_equations)


def test_local_window_storage_scales_with_window_not_sequence_squared() -> None:
    model = LongformerSelfAttention(
        32, seq_len=1024, heads=4, window_size=65, global_tokens=0
    )
    keys = torch.randn(1, 4, 1024, 8)
    windows = model._local_windows(keys)
    assert windows.shape == (1, 4, 1024, 65, 8)
    # unfold returns a view over a linearly padded tensor rather than allocating
    # one distinct key vector for every query/key pair.
    assert windows.untyped_storage().nbytes() < 1 * 4 * 1024 * 1024 * 8 * 4
