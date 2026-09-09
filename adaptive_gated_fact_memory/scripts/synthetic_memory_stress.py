"""Deterministic selective-memory stress data.

The original structured curriculum contains one writable fact and a system
distractor.  That verifies long-distance retrieval but never fills memory.
This generator adds user-role, fact-like *temporary* messages between an early
durable fact and its query.  Fixed memory must write those candidates, whereas
the gated model is supervised to keep the durable fact and reject temporary
content.  Noise count and token delay are independent evaluation axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoTokenizer

from synthetic_memory import CollatedRound, MemoryRound


@dataclass(frozen=True)
class StressMemoryExample:
    rounds: tuple[MemoryRound, ...]
    round_kinds: tuple[str, ...]
    fact_payload: tuple[int, ...]
    answer_ids: tuple[int, ...]
    delay_segments: int
    noise_rounds: int
    name: str
    value: str

    @property
    def query_round_index(self) -> int:
        return len(self.rounds) - 1


class SelectiveMemoryStressGenerator:
    """Create durable-fact -> temporary-noise -> delay -> query examples."""

    NAMES = (
        "Bob", "John", "Sam", "Tom", "Mark", "Luke", "Max", "Jack",
        "Rose", "Mary", "Amy", "Dan", "Ben", "Nora", "Paul", "Mike",
    )
    VALUES = (
        "red", "blue", "green", "black", "white", "gold", "brown", "gray",
        "orange", "silver", "yellow", "lime", "amber", "mint", "rose", "navy",
    )
    TRAIN_DURABLE_TEMPLATES = (
        "Remember: {name} likes {value}.",
        "Keep: {name} likes {value}.",
        "Save: {name} likes {value}.",
    )
    VALID_DURABLE_TEMPLATES = (
        "Store: {name} likes {value}.",
    )
    TRAIN_NOISE_TEMPLATES = (
        "Temporary: {name} likes {value}.",
        "Skip: {name} likes {value}.",
        "Ignore: {name} likes {value}.",
    )
    VALID_NOISE_TEMPLATES = (
        "Discard: {name} likes {value}.",
    )

    def __init__(
        self,
        tokenizer_dir: str | Path,
        filler_path: str | Path,
        *,
        eos_token_id: int = 50_256,
        segment_length: int = 128,
        split: str = "train",
        seed: int = 17,
        delay_buckets: tuple[int, ...] = (1, 2, 4, 8),
        noise_buckets: tuple[int, ...] = (0, 1, 2, 4),
        payload_tokens: int = 12,
        answer_leading_space: bool = False,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True
        )
        if len(self.tokenizer) != 50_257 or self.tokenizer.eos_token_id != eos_token_id:
            raise RuntimeError("unexpected GPT-2 tokenizer for memory stress data")
        self.eos_token_id = int(eos_token_id)
        self.segment_length = int(segment_length)
        self.split = split
        self.seed = int(seed)
        self.delay_buckets = tuple(int(value) for value in delay_buckets)
        self.noise_buckets = tuple(int(value) for value in noise_buckets)
        self.answer_leading_space = bool(answer_leading_space)
        if not self.delay_buckets or min(self.delay_buckets) <= 0:
            raise ValueError("delay buckets must be positive")
        if not self.noise_buckets or min(self.noise_buckets) < 0:
            raise ValueError("noise buckets must be non-negative")

        filler_path = Path(filler_path)
        metadata_path = filler_path.with_suffix(filler_path.suffix + ".json")
        if not filler_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"TinyStories filler cache is missing: {filler_path}")
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.filler = np.memmap(
            filler_path,
            mode="r",
            dtype=np.int32,
            shape=(int(metadata["token_count"]),),
        )
        self.payload_tokens = int(payload_tokens)
        for template in self.TRAIN_DURABLE_TEMPLATES + self.VALID_DURABLE_TEMPLATES:
            encoded = self.encode(template.format(name="Nora", value="orange"))
            if len(encoded) > self.payload_tokens:
                raise RuntimeError(
                    f"durable template uses {len(encoded)} tokens, exceeding "
                    f"payload budget {self.payload_tokens}: {template}"
                )

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(
            self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
        )

    @staticmethod
    def _roles(count: int, role: int) -> tuple[int, ...]:
        return tuple([int(role)] * count)

    def _choice(self, index: int, size: int, salt: int) -> int:
        value = (index + 1) * 1_000_003 + (self.seed + salt) * 97_409
        value ^= value >> 13
        value *= 1_664_525
        value ^= value >> 16
        return int(value % size)

    def _filler(self, index: int, delay_segments: int) -> tuple[int, ...]:
        required = max(self.segment_length, delay_segments * self.segment_length)
        extra = self._choice(index, max(1, self.segment_length // 4), 31)
        required += extra
        max_start = max(1, int(self.filler.shape[0]) - required - 1)
        start = self._choice(index, max_start, 47)
        values = self.filler[start : start + required].astype(np.int64).tolist()
        return tuple(int(value) for value in values)

    def make(
        self,
        index: int,
        *,
        delay_segments: int | None = None,
        noise_rounds: int | None = None,
    ) -> StressMemoryExample:
        if delay_segments is None:
            delay_segments = self.delay_buckets[
                self._choice(index, len(self.delay_buckets), 71)
            ]
        if noise_rounds is None:
            noise_rounds = self.noise_buckets[
                self._choice(index, len(self.noise_buckets), 73)
            ]
        if delay_segments <= 0 or noise_rounds < 0:
            raise ValueError("invalid stress condition")

        name_index = self._choice(index, len(self.NAMES), 101)
        value_index = self._choice(index, len(self.VALUES), 113)
        validation = self.split in {"validation", "valid", "test"}
        if validation:
            name_index = (name_index + 7) % len(self.NAMES)
            value_index = (value_index + 11) % len(self.VALUES)
            durable_templates = self.VALID_DURABLE_TEMPLATES
            noise_templates = self.VALID_NOISE_TEMPLATES
        else:
            durable_templates = self.TRAIN_DURABLE_TEMPLATES
            noise_templates = self.TRAIN_NOISE_TEMPLATES
        name = self.NAMES[name_index]
        value = self.VALUES[value_index]
        durable_template = durable_templates[
            self._choice(index, len(durable_templates), 127)
        ]
        durable_ids = self.encode(durable_template.format(name=name, value=value))

        rounds: list[MemoryRound] = [
            MemoryRound(
                input_ids=durable_ids,
                roles=self._roles(len(durable_ids), 1),
                fact_starts=(0,),
                fact_lengths=(len(durable_ids),),
            )
        ]
        kinds = ["durable"]
        for noise_index in range(noise_rounds):
            # Avoid an exact duplicate of the target while retaining the same
            # relation and vocabulary, so these are genuine fact-like writes.
            noise_name_index = (
                name_index + 1 + self._choice(index + noise_index, len(self.NAMES) - 1, 211)
            ) % len(self.NAMES)
            noise_value_index = (
                value_index + 1 + self._choice(index + noise_index, len(self.VALUES) - 1, 223)
            ) % len(self.VALUES)
            noise_template = noise_templates[
                self._choice(index + noise_index, len(noise_templates), 227)
            ]
            noise_ids = self.encode(
                noise_template.format(
                    name=self.NAMES[noise_name_index],
                    value=self.VALUES[noise_value_index],
                )
            )
            rounds.append(
                MemoryRound(
                    input_ids=noise_ids,
                    roles=self._roles(len(noise_ids), 1),
                )
            )
            kinds.append("noise")

        filler_ids = self._filler(index, int(delay_segments))
        rounds.append(
            MemoryRound(
                input_ids=filler_ids,
                roles=self._roles(len(filler_ids), 3),
            )
        )
        kinds.append("filler")

        query_ids = self.encode(f"What does {name} like?")
        answer_text = (" " if self.answer_leading_space else "") + value
        answer_ids = self.encode(answer_text) + (self.eos_token_id,)
        rounds.append(
            MemoryRound(
                input_ids=query_ids + answer_ids,
                roles=self._roles(len(query_ids), 1)
                + self._roles(len(answer_ids), 2),
                answer_start=len(query_ids),
                answer_ids=answer_ids,
            )
        )
        kinds.append("query")
        return StressMemoryExample(
            rounds=tuple(rounds),
            round_kinds=tuple(kinds),
            fact_payload=durable_ids,
            answer_ids=answer_ids,
            delay_segments=int(delay_segments),
            noise_rounds=int(noise_rounds),
            name=name,
            value=value,
        )

    def batch(
        self,
        indices: Iterable[int],
        device: torch.device,
        *,
        delay_segments: int | None = None,
        noise_rounds: int | None = None,
    ) -> tuple[list[StressMemoryExample], list[CollatedRound]]:
        examples = [
            self.make(
                int(index),
                delay_segments=delay_segments,
                noise_rounds=noise_rounds,
            )
            for index in indices
        ]
        round_count = len(examples[0].rounds)
        if any(len(example.rounds) != round_count for example in examples):
            raise ValueError("all examples in one batch must share a stress condition")

        collated_rounds: list[CollatedRound] = []
        for round_index in range(round_count):
            samples = [example.rounds[round_index] for example in examples]
            max_tokens = max(len(sample.input_ids) for sample in samples)
            batch_size = len(samples)
            input_ids = torch.zeros(
                batch_size, max_tokens, dtype=torch.long, device=device
            )
            roles = torch.zeros_like(input_ids)
            attention_mask = torch.zeros(
                batch_size, max_tokens, dtype=torch.bool, device=device
            )
            fact_start_targets = torch.zeros_like(attention_mask)
            fact_token_write_targets = torch.zeros_like(attention_mask)
            answer_target_mask = torch.zeros_like(attention_mask)
            fact_length_targets: list[list[tuple[int, int]]] = []
            for row, sample in enumerate(samples):
                length = len(sample.input_ids)
                input_ids[row, :length] = torch.tensor(sample.input_ids, device=device)
                roles[row, :length] = torch.tensor(sample.roles, device=device)
                attention_mask[row, :length] = True
                row_lengths: list[tuple[int, int]] = []
                for start, span_length in zip(sample.fact_starts, sample.fact_lengths):
                    fact_start_targets[row, start] = True
                    end = min(length, start + span_length)
                    fact_token_write_targets[row, start:end] = True
                    row_lengths.append((start, span_length))
                fact_length_targets.append(row_lengths)
                for position in range(length - 1):
                    if sample.roles[position + 1] == 2:
                        answer_target_mask[row, position] = True
            collated_rounds.append(
                CollatedRound(
                    input_ids=input_ids,
                    roles=roles,
                    attention_mask=attention_mask,
                    fact_start_targets=fact_start_targets,
                    fact_token_write_targets=fact_token_write_targets,
                    fact_length_targets=fact_length_targets,
                    answer_target_mask=answer_target_mask,
                )
            )
        return examples, collated_rounds
