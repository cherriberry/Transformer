"""Deterministic A -> long B -> return-A data for the memory stage.

The generator deliberately keeps the supervised fact span short and
auditable.  A user fact is followed by a system-role TinyStories distractor
whose length is larger than the local KV budget, then a user query asks for
the earlier value.  TinyStories tokens are used only as natural distractors;
the fact/query vocabulary and all evaluation combinations are generated
independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoTokenizer


@dataclass(frozen=True)
class MemoryRound:
    input_ids: tuple[int, ...]
    roles: tuple[int, ...]
    fact_starts: tuple[int, ...] = ()
    fact_lengths: tuple[int, ...] = ()
    answer_start: int | None = None
    answer_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MemoryExample:
    rounds: tuple[MemoryRound, ...]
    fact_payload: tuple[int, ...]
    answer_ids: tuple[int, ...]
    delay_segments: int
    name: str
    value: str


@dataclass
class CollatedRound:
    input_ids: torch.Tensor
    roles: torch.Tensor
    attention_mask: torch.Tensor
    fact_start_targets: torch.Tensor
    fact_token_write_targets: torch.Tensor
    fact_length_targets: list[list[tuple[int, int]]]
    answer_target_mask: torch.Tensor


class SyntheticMemoryGenerator:
    """Create fixed-shape-friendly synthetic memory conversations."""

    # Keep these lexical inventories small enough for a first curriculum, but
    # use disjoint deterministic slices for train and validation combinations.
    NAMES = (
        "Bob", "John", "Sam", "Tom", "Mark", "Luke", "Max", "Jack",
        "Rose", "Mary", "Amy", "Dan", "Ben", "Nora", "Paul", "Mike",
    )
    VALUES = (
        "red", "blue", "green", "black", "white", "gold", "brown", "gray",
        "orange", "silver", "yellow", "lime", "amber", "mint", "rose", "navy",
    )

    def __init__(
        self,
        tokenizer_dir: str | Path,
        filler_path: str | Path,
        *,
        eos_token_id: int = 50256,
        context: int = 512,
        segment_length: int = 128,
        split: str = "train",
        seed: int = 17,
        delay_buckets: tuple[int, ...] | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True
        )
        if len(self.tokenizer) != 50257 or self.tokenizer.eos_token_id != eos_token_id:
            raise RuntimeError("unexpected GPT-2 tokenizer for synthetic memory data")
        self.eos_token_id = eos_token_id
        self.context = context
        self.segment_length = segment_length
        self.split = split
        self.seed = int(seed)
        if delay_buckets is None:
            delay_buckets = (1, 2, 4, 8, 16)
        if not delay_buckets or any(int(delay) <= 0 for delay in delay_buckets):
            raise ValueError("delay_buckets must contain positive integers")
        self.delay_buckets = tuple(int(delay) for delay in delay_buckets)
        filler_path = Path(filler_path)
        metadata_path = filler_path.with_suffix(filler_path.suffix + ".json")
        if not filler_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"TinyStories filler cache is missing: {filler_path}; run the "
                "pinned TinyStories cache preparation first"
            )
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.filler = np.memmap(
            filler_path,
            mode="r",
            dtype=np.int32,
            shape=(int(metadata["token_count"]),),
        )
        self.fact_template = "{name} likes {value}."
        self.query_template = "What does {name} like?"
        self.ack_template = "Okay."
        # Validate the payload contract once.  All fact spans must fit in the
        # model's copied-token payload budget (12 in the default config).
        for name in self.NAMES[:4]:
            for value in self.VALUES[:4]:
                fact = self.encode(self.fact_template.format(name=name, value=value))
                if len(fact) > 12:
                    raise RuntimeError(f"fact template exceeds payload budget: {fact}")

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(
            self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
        )

    @property
    def delays(self) -> tuple[int, ...]:
        return self.delay_buckets

    def _choice(self, index: int, size: int, salt: int) -> int:
        # A cheap integer hash gives stable but non-periodic combinations while
        # avoiding global RNG state in dataloader workers.
        value = (index + 1) * 1_000_003 + (self.seed + salt) * 97_409
        value ^= value >> 13
        value *= 1_664_525
        value ^= value >> 16
        return int(value % size)

    def _filler(self, index: int, delay_segments: int) -> tuple[int, ...]:
        required = max(self.segment_length, delay_segments * self.segment_length)
        # Leave a little variation while retaining the requested delay bucket.
        extra = self._choice(index, max(1, self.segment_length // 4), 31)
        required += extra
        max_start = max(1, int(self.filler.shape[0]) - required - 1)
        start = self._choice(index, max_start, 47)
        values = self.filler[start : start + required].astype(np.int64).tolist()
        return tuple(int(value) for value in values)

    def _roles(self, count: int, role: int) -> tuple[int, ...]:
        return tuple([int(role)] * count)

    def make(self, index: int, *, delay_segments: int | None = None) -> MemoryExample:
        if delay_segments is None:
            delay_segments = self.delays[self._choice(index, len(self.delays), 71)]
        if delay_segments <= 0:
            raise ValueError("delay_segments must be positive")

        # Validation shifts the lexical pairing so combinations are not exact
        # copies of training examples, while preserving shared query semantics.
        name_index = self._choice(index, len(self.NAMES), 101)
        value_index = self._choice(index, len(self.VALUES), 113)
        if self.split in {"validation", "valid", "test"}:
            name_index = (name_index + 7) % len(self.NAMES)
            value_index = (value_index + 11) % len(self.VALUES)
        name = self.NAMES[name_index]
        value = self.VALUES[value_index]

        fact_ids = self.encode(self.fact_template.format(name=name, value=value))
        query_ids = self.encode(self.query_template.format(name=name))
        ack_ids = self.encode(self.ack_template) + (self.eos_token_id,)
        answer_ids = self.encode(value) + (self.eos_token_id,)
        filler_ids = self._filler(index, delay_segments)

        # Round 0: the only supervised user fact.  The complete user message
        # is exactly the fact span, so its start is unambiguous at token 0.
        round_a = MemoryRound(
            input_ids=fact_ids + ack_ids,
            roles=self._roles(len(fact_ids), 1) + self._roles(len(ack_ids), 2),
            fact_starts=(0,),
            fact_lengths=(len(fact_ids),),
        )
        # Round B is system-role distractor text.  It pushes A out of the
        # recent local cache without creating additional user facts.
        round_b = MemoryRound(
            input_ids=filler_ids,
            roles=self._roles(len(filler_ids), 3),
        )
        # The answer is teacher-forced during training.  Its first token is
        # predicted from the final query token, exactly as at inference.
        round_q = MemoryRound(
            input_ids=query_ids + answer_ids,
            roles=self._roles(len(query_ids), 1) + self._roles(len(answer_ids), 2),
            answer_start=len(query_ids),
            answer_ids=answer_ids,
        )
        return MemoryExample(
            rounds=(round_a, round_b, round_q),
            fact_payload=fact_ids,
            answer_ids=answer_ids,
            delay_segments=delay_segments,
            name=name,
            value=value,
        )

    def batch(
        self,
        indices: Iterable[int],
        device: torch.device,
        *,
        delay_segments: int | None = None,
    ) -> tuple[list[MemoryExample], list[CollatedRound]]:
        examples = [self.make(int(index), delay_segments=delay_segments) for index in indices]
        rounds: list[CollatedRound] = []
        for round_index in range(3):
            samples = [example.rounds[round_index] for example in examples]
            max_tokens = max(len(sample.input_ids) for sample in samples)
            batch = len(samples)
            input_ids = torch.zeros(batch, max_tokens, dtype=torch.long, device=device)
            roles = torch.zeros_like(input_ids)
            mask = torch.zeros(batch, max_tokens, dtype=torch.bool, device=device)
            start_targets = torch.zeros(batch, max_tokens, dtype=torch.bool, device=device)
            token_write_targets = torch.zeros(
                batch, max_tokens, dtype=torch.bool, device=device
            )
            length_targets: list[list[tuple[int, int]]] = []
            answer_mask = torch.zeros(batch, max_tokens, dtype=torch.bool, device=device)
            for row, sample in enumerate(samples):
                length = len(sample.input_ids)
                input_ids[row, :length] = torch.tensor(sample.input_ids, device=device)
                roles[row, :length] = torch.tensor(sample.roles, device=device)
                mask[row, :length] = True
                row_lengths: list[tuple[int, int]] = []
                for start, span_length in zip(sample.fact_starts, sample.fact_lengths):
                    start_targets[row, start] = True
                    # The extractor writes a span-level representation.  Mark
                    # every token in the labelled fact as positive so the
                    # averaged candidate write logit remains above the
                    # curriculum threshold; supervising only the first token
                    # would encourage the span average toward zero.
                    end = min(length, start + span_length)
                    token_write_targets[row, start:end] = True
                    row_lengths.append((start, span_length))
                length_targets.append(row_lengths)
                # A logit at position i predicts input_ids[i+1]; select only
                # positions whose target token is an assistant token.
                for position in range(length - 1):
                    if sample.roles[position + 1] == 2:
                        answer_mask[row, position] = True
            rounds.append(
                CollatedRound(
                    input_ids=input_ids,
                    roles=roles,
                    attention_mask=mask,
                    fact_start_targets=start_targets,
                    fact_token_write_targets=token_write_targets,
                    fact_length_targets=length_targets,
                    answer_target_mask=answer_mask,
                )
            )
        return examples, rounds
