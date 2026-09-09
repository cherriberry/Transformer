#!/usr/bin/env python3
"""Run a tiny CPU/CUDA forward, commit, read, backward, and state audit."""

from __future__ import annotations

import json

import torch

from adaptive_fact_memory import AdaptiveFactMemoryLM, FactMemoryConfig, SourceRole


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FactMemoryConfig(
        vocab_size=512,
        layers=2,
        hidden_size=64,
        heads=4,
        ffn_size=128,
        dropout=0.0,
        attention_budget=16,
        sink_tokens=4,
        memory_slots=8,
        memory_read_top_k=4,
        memory_fusion_layers=(1,),
        max_write_candidates=2,
        payload_tokens=6,
        assistant_slots=4,
    )
    model = AdaptiveFactMemoryLM(config).to(device).train()
    with torch.no_grad():
        model.fact_extractor.write_head.bias.fill_(8.0)
    tokens = torch.randint(1, config.vocab_size, (2, 24), device=device)
    roles = torch.full_like(tokens, int(SourceRole.USER))
    roles[:, 16:] = int(SourceRole.ASSISTANT)
    first = model.forward_round(
        tokens, source_roles=roles, commit=True, hard_memory=False
    )
    loss = first.logits.float().square().mean() + 0.01 * first.auxiliary_losses["memory_budget"]
    loss.backward()
    second = model.forward_round(
        tokens[:, :8],
        state=first.state.detach(),
        source_roles=roles[:, :8],
        commit=False,
    )
    record = {
        "device": str(device),
        "logits_shape": list(first.logits.shape),
        "next_position": first.state.next_position.tolist(),
        "active_slots": first.diagnostics["active_slots"].tolist(),
        "next_round_read_count": second.diagnostics["read_valid"].sum(dim=1).tolist(),
        "state_bytes": model.state_bytes(first.state),
        "cache_entries_per_layer": [
            diagnostic["cache_entries"].tolist()
            for diagnostic in first.diagnostics["layer_attention"]
        ],
        "loss": float(loss.detach()),
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
