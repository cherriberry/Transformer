from __future__ import annotations

import unittest

import torch

from adaptive_fact_memory import AdaptiveFactMemoryLM, FactMemoryConfig, SourceRole
from adaptive_fact_memory.memory import FactCandidates, FactMemoryController
from adaptive_fact_memory.state import FactMemoryState


def tiny_config(**overrides) -> FactMemoryConfig:
    values = dict(
        vocab_size=97,
        layers=2,
        hidden_size=32,
        heads=4,
        ffn_size=64,
        dropout=0.0,
        rope_max_position=256,
        attention_budget=8,
        sink_tokens=2,
        memory_slots=6,
        memory_read_top_k=3,
        memory_fusion_layers=(1,),
        max_write_candidates=2,
        payload_tokens=4,
        assistant_slots=2,
    )
    values.update(overrides)
    return FactMemoryConfig(**values)


def force_writes(model: AdaptiveFactMemoryLM) -> None:
    with torch.no_grad():
        model.fact_extractor.write_head.weight.zero_()
        model.fact_extractor.write_head.bias.fill_(10.0)
        model.fact_extractor.confidence_head.weight.zero_()
        model.fact_extractor.confidence_head.bias.fill_(4.0)
        model.memory_controller.assistant_write_head.weight.zero_()
        model.memory_controller.assistant_write_head.bias.fill_(10.0)


class AdaptiveFactMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_chunked_attention_matches_one_shot(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        token_ids = torch.randint(1, model.config.vocab_size, (1, 18))
        roles = torch.full_like(token_ids, int(SourceRole.USER))
        with torch.no_grad():
            whole = model.forward_round(token_ids, source_roles=roles)
            state = None
            chunks = []
            for start, end in ((0, 3), (3, 9), (9, 13), (13, 18)):
                output = model.forward_round(
                    token_ids[:, start:end],
                    state=state,
                    source_roles=roles[:, start:end],
                )
                chunks.append(output.logits)
                state = output.state
        torch.testing.assert_close(torch.cat(chunks, dim=1), whole.logits, atol=2e-5, rtol=2e-5)

    def test_prefix_is_causal_before_memory_commit(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        token_ids = torch.randint(1, model.config.vocab_size, (1, 14))
        roles = torch.full_like(token_ids, int(SourceRole.USER))
        with torch.no_grad():
            short = model.forward_round(token_ids[:, :8], source_roles=roles[:, :8])
            long = model.forward_round(token_ids, source_roles=roles)
        torch.testing.assert_close(short.logits[:, :7], long.logits[:, :7], atol=2e-5, rtol=2e-5)

    def test_local_cache_is_physically_bounded(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        state = None
        with torch.no_grad():
            for _ in range(7):
                token_ids = torch.randint(1, model.config.vocab_size, (2, 5))
                state = model.forward_round(token_ids, state=state).state
        self.assertTrue(torch.equal(state.next_position, torch.tensor([35, 35])))
        for cache in state.layer_caches:
            entries = cache.sink_valid.sum(dim=1) + cache.recent_valid.sum(dim=1)
            self.assertTrue(bool((entries <= model.config.attention_budget).all()))
            self.assertEqual(cache.recent_k.shape[2], model.config.recent_window)
            self.assertEqual(cache.sink_k.shape[2], model.config.sink_tokens)

    def test_commit_cannot_change_current_round_logits(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        force_writes(model)
        token_ids = torch.randint(1, model.config.vocab_size, (2, 10))
        roles = torch.full_like(token_ids, int(SourceRole.USER))
        roles[:, 6:] = int(SourceRole.ASSISTANT)
        with torch.no_grad():
            no_commit = model.forward_round(token_ids, source_roles=roles, commit=False)
            committed = model.forward_round(token_ids, source_roles=roles, commit=True)
        torch.testing.assert_close(no_commit.logits, committed.logits, atol=0, rtol=0)
        self.assertEqual(int(no_commit.state.memory.active.sum()), 0)
        self.assertTrue(bool((committed.diagnostics["active_slots"] > 0).all()))
        self.assertTrue(bool(committed.diagnostics["assistant_write_accepted"].all()))

        next_ids = torch.randint(1, model.config.vocab_size, (2, 4))
        next_roles = torch.full_like(next_ids, int(SourceRole.USER))
        with torch.no_grad():
            next_output = model.forward_round(
                next_ids, state=committed.state, source_roles=next_roles
            )
        self.assertTrue(bool(next_output.diagnostics["read_valid"].any()))

    def test_reset_and_batch_reorder_cover_all_state(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        force_writes(model)
        token_ids = torch.tensor(
            [[11, 12, 13, 14, 21, 22], [31, 32, 33, 34, 41, 42]], dtype=torch.long
        )
        roles = torch.full_like(token_ids, int(SourceRole.USER))
        roles[:, 4:] = int(SourceRole.ASSISTANT)
        with torch.no_grad():
            state = model.forward_round(token_ids, source_roles=roles, commit=True).state
        reordered = state.index_select(torch.tensor([1, 0]))
        torch.testing.assert_close(reordered.memory.keys[0], state.memory.keys[1])
        torch.testing.assert_close(
            reordered.memory.lexical_values[0], state.memory.lexical_values[1]
        )
        torch.testing.assert_close(
            reordered.memory.value_payload_ids[0], state.memory.value_payload_ids[1]
        )
        torch.testing.assert_close(reordered.layer_caches[0].recent_k[1], state.layer_caches[0].recent_k[0])

        reset = state.reset(torch.tensor([True, False]))
        self.assertEqual(float(reset.memory.active[0].sum()), 0.0)
        self.assertGreater(float(reset.memory.active[1].sum()), 0.0)
        self.assertEqual(float(reset.memory.lexical_values[0].abs().sum()), 0.0)
        self.assertFalse(bool(reset.memory.value_payload_mask[0].any()))
        self.assertEqual(int(reset.next_position[0]), 0)
        self.assertEqual(int(reset.next_position[1]), 6)
        self.assertFalse(bool(reset.layer_caches[0].sink_valid[0].any()))
        self.assertTrue(bool(reset.layer_caches[0].sink_valid[1].any()))

    def test_soft_write_path_propagates_budget_gradient(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).train()
        force_writes(model)
        token_ids = torch.randint(1, model.config.vocab_size, (2, 8))
        roles = torch.full_like(token_ids, int(SourceRole.USER))
        output = model.forward_round(
            token_ids, source_roles=roles, commit=True, hard_memory=False
        )
        loss = output.logits.float().square().mean() + output.auxiliary_losses["memory_budget"]
        loss.backward()
        gradient = model.fact_extractor.write_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_padding_does_not_advance_or_enter_cache(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        token_ids = torch.randint(1, model.config.vocab_size, (2, 7))
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool
        )
        with torch.no_grad():
            output = model.forward_round(token_ids, attention_mask=attention_mask)
        self.assertTrue(torch.equal(output.state.next_position, torch.tensor([7, 3])))
        self.assertEqual(float(output.logits[1, 3:].abs().sum()), 0.0)
        for cache in output.state.layer_caches:
            self.assertEqual(int(cache.sink_valid[1].sum() + cache.recent_valid[1].sum()), 3)

    def test_memory_is_hidden_until_end_of_prompt(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        state_with_memory = model.initial_state(1)
        state_without_value = model.initial_state(1)
        memory_key = torch.randn(model.config.hidden_size)
        memory_value = torch.randn(model.config.hidden_size) * 5
        for state in (state_with_memory, state_without_value):
            state.memory.keys[0, 0] = memory_key
            state.memory.active[0, 0] = 1
        state_with_memory.memory.values[0, 0] = memory_value
        token_ids = torch.randint(1, model.config.vocab_size, (1, 8))
        roles = torch.full_like(token_ids, int(SourceRole.USER))
        roles[:, 4:] = int(SourceRole.ASSISTANT)
        with torch.no_grad():
            with_memory = model.forward_round(
                token_ids, state=state_with_memory, source_roles=roles
            )
            without_value = model.forward_round(
                token_ids, state=state_without_value, source_roles=roles
            )
        torch.testing.assert_close(
            with_memory.logits[:, :3], without_value.logits[:, :3], atol=0, rtol=0
        )
        self.assertGreater(
            float((with_memory.logits[:, 3:] - without_value.logits[:, 3:]).abs().max()),
            0.0,
        )

    def test_retrieval_is_reused_during_assistant_continuation(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        state = model.initial_state(1)
        state.memory.keys[0, 0] = torch.randn(model.config.hidden_size)
        state.memory.values[0, 0] = torch.randn(model.config.hidden_size)
        state.memory.active[0, 0] = 1
        prompt = torch.randint(1, model.config.vocab_size, (1, 5))
        with torch.no_grad():
            opened = model.forward_round(
                prompt,
                state=state,
                source_roles=torch.full_like(prompt, int(SourceRole.USER)),
            )
            access_after_prompt = opened.state.memory.access_count.clone()
            continuation_ids = torch.randint(1, model.config.vocab_size, (1, 3))
            continued = model.forward_round(
                continuation_ids,
                state=opened.state,
                source_roles=torch.full_like(continuation_ids, int(SourceRole.ASSISTANT)),
            )
        self.assertTrue(bool(opened.state.round_open.all()))
        self.assertTrue(bool(continued.state.round_open.all()))
        torch.testing.assert_close(continued.state.memory.access_count, access_after_prompt)
        torch.testing.assert_close(
            continued.diagnostics["read_indices"], opened.diagnostics["read_indices"]
        )

    def test_streamed_round_matches_one_shot_commit(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        force_writes(model)
        user_ids = torch.randint(1, model.config.vocab_size, (1, 7))
        assistant_ids = torch.randint(1, model.config.vocab_size, (1, 5))
        whole_ids = torch.cat((user_ids, assistant_ids), dim=1)
        whole_roles = torch.cat(
            (
                torch.full_like(user_ids, int(SourceRole.USER)),
                torch.full_like(assistant_ids, int(SourceRole.ASSISTANT)),
            ),
            dim=1,
        )
        with torch.no_grad():
            whole = model.forward_round(
                whole_ids, source_roles=whole_roles, commit=True
            )
            user = model.forward_round(
                user_ids,
                source_roles=torch.full_like(user_ids, int(SourceRole.USER)),
                commit=False,
            )
            assistant_a = model.forward_round(
                assistant_ids[:, :2],
                state=user.state,
                source_roles=torch.full_like(
                    assistant_ids[:, :2], int(SourceRole.ASSISTANT)
                ),
                commit=False,
            )
            assistant_b = model.forward_round(
                assistant_ids[:, 2:],
                state=assistant_a.state,
                source_roles=torch.full_like(
                    assistant_ids[:, 2:], int(SourceRole.ASSISTANT)
                ),
                commit=True,
            )
        streamed_logits = torch.cat(
            (user.logits, assistant_a.logits, assistant_b.logits), dim=1
        )
        torch.testing.assert_close(streamed_logits, whole.logits, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            assistant_b.state.memory.keys, whole.state.memory.keys, atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(
            assistant_b.state.memory.values,
            whole.state.memory.values,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            assistant_b.state.memory.assistant_values,
            whole.state.memory.assistant_values,
            atol=2e-5,
            rtol=2e-5,
        )
        self.assertFalse(bool(assistant_b.state.pending.valid.any()))
        self.assertEqual(int(assistant_b.state.pending.assistant_count.sum()), 0)
        self.assertFalse(bool(assistant_b.state.round_open.any()))

    def test_retention_gate_can_physically_deactivate_a_slot(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).eval()
        state = model.initial_state(1)
        state.memory.keys[0, 0] = torch.randn(model.config.hidden_size)
        state.memory.values[0, 0] = torch.randn(model.config.hidden_size)
        state.memory.active[0, 0] = 1
        with torch.no_grad():
            model.memory_controller.retention_head.weight.zero_()
            model.memory_controller.retention_head.bias.fill_(-10.0)
            token_ids = torch.randint(1, model.config.vocab_size, (1, 4))
            output = model.forward_round(
                token_ids,
                state=state,
                source_roles=torch.full_like(token_ids, int(SourceRole.SYSTEM)),
                commit=True,
            )
        self.assertEqual(int(output.diagnostics["active_slots"]), 0)
        self.assertTrue(bool(output.diagnostics["retention_evicted"][0, 0]))

    def test_fixed_lru_eviction_does_not_use_frozen_retention_head(self) -> None:
        config = tiny_config(memory_slots=2, memory_read_top_k=2)
        controller = FactMemoryController(config).eval()
        state = FactMemoryState.empty(
            config, 1, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            state.active[:] = 1.0
            state.last_access[:] = torch.tensor([[3, 9]])
            state.keys[0, 0, 0] = 1.0
            state.keys[0, 1, 1] = 1.0
            candidate_key = torch.zeros(1, 1, config.hidden_size)
            candidate_key[0, 0, 2] = 1.0
            candidate = FactCandidates(
                keys=candidate_key,
                values=torch.ones(1, 1, config.hidden_size),
                lexical_values=torch.full((1, 1, config.hidden_size), 2.0),
                payload_ids=torch.tensor([[[42, 0, 0, 0]]]),
                payload_mask=torch.tensor([[[True, False, False, False]]]),
                value_payload_ids=torch.tensor([[[43, 0, 0, 0]]]),
                value_payload_mask=torch.tensor([[[True, False, False, False]]]),
                starts=torch.zeros(1, 1, dtype=torch.long),
                lengths=torch.ones(1, 1, dtype=torch.long),
                value_starts=torch.zeros(1, 1, dtype=torch.long),
                value_lengths=torch.ones(1, 1, dtype=torch.long),
                write_probability=torch.ones(1, 1),
                confidence=torch.ones(1, 1),
                valid=torch.ones(1, 1, dtype=torch.bool),
                start_logits=torch.zeros(1, 1),
                length_logits=torch.zeros(1, 1, config.payload_tokens),
                token_write_logits=torch.zeros(1, 1),
                value_start_logits=torch.zeros(1, 1),
                value_length_logits=torch.zeros(1, 1, config.payload_tokens),
            )
            updated, diagnostics = controller.commit_facts(
                state,
                candidate,
                torch.ones(1, dtype=torch.bool),
                hard=True,
                fixed_policy=True,
                fixed_eviction_policy="lru",
            )
        self.assertEqual(int(diagnostics["write_targets"][0, 0]), 0)
        self.assertEqual(int(updated.payload_ids[0, 0, 0]), 42)
        self.assertEqual(int(updated.value_payload_ids[0, 0, 0]), 43)
        self.assertEqual(float(updated.lexical_values[0, 0, 0]), 2.0)

    def test_value_span_excludes_padding_and_preserves_token_ids(self) -> None:
        config = tiny_config(max_write_candidates=1)
        model = AdaptiveFactMemoryLM(config, value_token_alignment=True).eval()
        extractor = model.fact_extractor
        hidden = torch.zeros(1, 5, config.hidden_size)
        hidden[0, 0, 0] = 5.0
        hidden[0, 2, 1] = 7.0
        input_ids = torch.tensor([[11, 12, 13, 14, 95]])
        token_mask = torch.tensor([[True, True, True, True, False]])
        roles = torch.full_like(input_ids, int(SourceRole.USER))
        with torch.no_grad():
            extractor.start_head.weight.zero_()
            extractor.start_head.weight[0, 0] = 1.0
            extractor.start_head.bias.zero_()
            extractor.length_head.weight.zero_()
            extractor.length_head.bias.zero_()
            extractor.length_head.bias[3] = 10.0
            extractor.value_start_head.weight.zero_()
            extractor.value_start_head.weight[0, 1] = 1.0
            extractor.value_start_head.bias.zero_()
            extractor.value_length_head.weight.zero_()
            extractor.value_length_head.bias.zero_()
            extractor.value_length_head.bias[1] = 10.0
            candidates = extractor(hidden, input_ids, token_mask, roles)
        self.assertEqual(int(candidates.starts[0, 0]), 0)
        self.assertEqual(int(candidates.lengths[0, 0]), 4)
        self.assertEqual(int(candidates.value_starts[0, 0]), 2)
        self.assertEqual(int(candidates.value_lengths[0, 0]), 2)
        self.assertEqual(candidates.value_payload_ids[0, 0, :2].tolist(), [13, 14])
        self.assertEqual(candidates.value_payload_mask[0, 0].tolist(), [True, True, False, False])
        self.assertNotIn(95, candidates.value_payload_ids[0, 0].tolist())

    def test_memory_token_loss_reaches_value_encoder(self) -> None:
        config = tiny_config(max_write_candidates=1)
        model = AdaptiveFactMemoryLM(config, value_token_alignment=True).train()
        token_ids = torch.tensor([[11, 12, 13, 14]])
        roles = torch.full_like(token_ids, int(SourceRole.USER))
        output = model.forward_round(token_ids, source_roles=roles)
        logits = model.memory_to_token_logits(
            output.supervision["candidate_lexical_values"][:, 0], 2
        )
        targets = torch.tensor([[21, 22]])
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), targets.reshape(-1)
        ).backward()
        self.assertIsNotNone(model.fact_extractor.lexical_projection.weight.grad)
        self.assertGreater(
            float(model.fact_extractor.lexical_projection.weight.grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(model.memory_token_projection.weight.grad)

    def test_default_model_keeps_legacy_value_path(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config(max_write_candidates=1)).eval()
        self.assertFalse(model.value_token_alignment)
        self.assertIsNone(model.fact_extractor.value_start_head)
        token_ids = torch.tensor([[11, 12, 13, 14]])
        roles = torch.full_like(token_ids, int(SourceRole.USER))
        with torch.no_grad():
            output = model.forward_round(token_ids, source_roles=roles)
        torch.testing.assert_close(
            output.diagnostics["candidate_value_payload_ids"],
            output.diagnostics["candidate_payload_ids"],
        )

    def test_tinystories_stage_trains_backbone_only(self) -> None:
        model = AdaptiveFactMemoryLM(tiny_config()).train()
        token_ids = torch.randint(1, model.config.vocab_size, (2, 16))
        logits = model.forward_tinystories(token_ids)
        self.assertEqual(tuple(logits.shape), (2, 16, model.config.vocab_size))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, model.config.vocab_size),
            token_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if any(
                    marker in name
                    for marker in (
                        "fact_extractor",
                        "memory_controller",
                        "memory_reader",
                        "memory_norm",
                        "memory_fusion",
                    )
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
