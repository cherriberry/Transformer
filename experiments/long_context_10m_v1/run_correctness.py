"""Correctness gates for the B-role Longformer and Memformer implementations."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from b_role_experiments import MemformerSegmentAttention, StandardAttention  # noqa: E402
from models.longformer_attention import LongformerSelfAttention  # noqa: E402


def max_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def shared_longformer(dim: int, heads: int, n: int) -> tuple[StandardAttention, LongformerSelfAttention]:
    full = StandardAttention(dim, heads, causal=True)
    local = LongformerSelfAttention(dim, n, heads=heads, window_size=2 * n - 1,
                                    global_tokens=0, causal=True)
    for name in ("to_q", "to_k", "to_v", "to_out"):
        getattr(local, name).load_state_dict(getattr(full, name).state_dict())
    return full, local


def recovery_delta(factory, forward) -> float:
    model = factory()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(1, 16, 32)
    target = torch.randn(1, 16, 32)

    def step() -> float:
        loss = (forward(model, x) - target).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return float(loss)

    step()
    model_state, opt_state = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    uninterrupted = step()
    restored = factory(); restored_opt = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored.load_state_dict(model_state); restored_opt.load_state_dict(opt_state)
    loss = (forward(restored, x) - target).square().mean()
    restored_opt.zero_grad(set_to_none=True); loss.backward(); restored_opt.step()
    return abs(uninterrupted - float(loss))


def main() -> None:
    torch.manual_seed(17)
    dim, heads, n, t = 32, 4, 17, 9
    x = torch.randn(1, n, dim)
    changed = x.clone(); changed[:, t + 1:] = torch.randn_like(changed[:, t + 1:])

    full, local = shared_longformer(dim, heads, n)
    local.eval(); full.eval()
    lf_future = max_delta(local(x)[:, :t + 1], local(changed)[:, :t + 1])
    lf_equiv = max_delta(full(x), local(x))
    padded = torch.cat([x, torch.randn(1, 5, dim)], dim=1)
    pad_mask = torch.cat([torch.ones(1, n), torch.zeros(1, 5)], dim=1).bool()
    lf_pad_model = LongformerSelfAttention(dim, n + 5, heads=heads, window_size=2 * (n + 5) - 1,
                                           global_tokens=0, causal=True)
    lf_pad_model.load_state_dict(local.state_dict())
    lf_padding = max_delta(local(x), lf_pad_model(padded, attention_mask=pad_mask)[:, :n])
    grad_x = x.clone().requires_grad_(True); local(grad_x).square().mean().backward()
    lf_grad = bool(torch.isfinite(grad_x.grad).all()) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in local.parameters())

    mem = MemformerSegmentAttention(dim, heads, memory_slots=8, causal=True, detach_memory=False)
    mem.eval()
    mx = torch.randn(1, 16, dim)
    mchanged = mx.clone(); mchanged[:, 12:] = torch.randn_like(mchanged[:, 12:])
    mout, state = mem(mx, segment_length=8)
    mout_changed, _ = mem(mchanged, segment_length=8)
    mem_future = max_delta(mout[:, :12], mout_changed[:, :12])
    mpadded = torch.cat([mx, torch.randn(1, 8, dim)], dim=1)
    mem_padding = max_delta(mout, mem(mpadded, segment_length=8)[0][:, :16])

    seg1, seg2 = mx[:, :8], mx[:, 8:]
    zero = torch.zeros(1, 8, dim)
    _, state1 = mem.forward_segment(seg1, zero)
    hist_out, _ = mem.forward_segment(seg2, state1)
    reset_out, reset_state = mem.forward_segment(seg2, zero)
    reset_out_2, reset_state_2 = mem.forward_segment(seg2, zero)
    history_effect = max_delta(hist_out, reset_out)
    reset_delta = max(max_delta(reset_out, reset_out_2), max_delta(reset_state, reset_state_2))

    bx = torch.randn(2, 8, dim)
    _, bstate = mem.forward_segment(bx, torch.zeros(2, 8, dim))
    next_x = torch.randn(2, 8, dim)
    base_out, base_state = mem.forward_segment(next_x, bstate)
    perm = torch.tensor([1, 0])
    perm_out, perm_state = mem.forward_segment(next_x[perm], bstate[perm])
    reorder_delta = max(max_delta(base_out[perm], perm_out), max_delta(base_state[perm], perm_state))
    mem_grad_x = mx.clone().requires_grad_(True); mem(mem_grad_x, segment_length=8)[0].square().mean().backward()
    missing_grads = [name for name, p in mem.named_parameters()
                     if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]

    lf_recovery = recovery_delta(
        lambda: LongformerSelfAttention(32, 16, heads=4, window_size=17,
                                        global_tokens=0, causal=True),
        lambda model, value: model(value),
    )
    mem_recovery = recovery_delta(
        lambda: MemformerSegmentAttention(32, 4, memory_slots=8, causal=True,
                                          detach_memory=False),
        lambda model, value: model(value, segment_length=8)[0],
    )

    tests = {
        "longformer_future_invariance": {"value": lf_future, "limit": 1e-5},
        "longformer_full_window_equivalence": {"value": lf_equiv, "limit": 1e-5},
        "longformer_padding_invariance": {"value": lf_padding, "limit": 1e-5},
        "longformer_checkpoint_recovery": {"value": lf_recovery, "limit": 1e-7},
        "memformer_future_invariance": {"value": mem_future, "limit": 1e-5},
        "memformer_padding_by_causal_append": {"value": mem_padding, "limit": 1e-5},
        "memformer_history_effect": {"value": history_effect, "minimum": 1e-8},
        "memformer_reset": {"value": reset_delta, "limit": 1e-5},
        "memformer_batch_reorder": {"value": reorder_delta, "limit": 1e-5},
        "memformer_checkpoint_recovery": {"value": mem_recovery, "limit": 1e-7},
    }
    passed = lf_grad and not missing_grads
    for item in tests.values():
        if "limit" in item:
            passed &= math.isfinite(item["value"]) and item["value"] <= item["limit"]
        else:
            passed &= math.isfinite(item["value"]) and item["value"] >= item["minimum"]
    payload = {"protocol_version": "long_context_10m_v1", "dtype": "float32_reference",
               "status": "passed" if passed else "failed_correctness", "tests": tests,
               "longformer_all_gradients_finite": lf_grad,
               "memformer_missing_or_nonfinite_gradients": missing_grads,
               "memformer_state_shape": list(state.shape), "memformer_state_elements": state.numel()}
    out = Path(__file__).resolve().parent / "aggregate" / "correctness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
