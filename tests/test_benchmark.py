from __future__ import annotations

import torch
import torch.nn as nn

from universal_benchmark import UniversalBenchmark


class IdentityModel(nn.Module):
    def __init__(self, dim: int, seq_len: int, **_: object) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class FailsAtLength(nn.Module):
    def __init__(self, dim: int, seq_len: int, **_: object) -> None:
        super().__init__()
        self.fail = seq_len == 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("intentional failure")
        return x


def test_benchmark_returns_statistics_and_raw_runs() -> None:
    result = UniversalBenchmark("test").benchmark_model(
        IdentityModel, {"dim": 8}, [4, 6], num_runs=2, warmup_runs=1
    )
    assert len(result.records) == 2
    for record in result.records:
        assert record["status"] == "ok"
        assert len(record["run_times_seconds"]) == 2
        assert record["mean_seconds"] >= 0
        assert record["median_seconds"] >= 0
        assert record["std_seconds"] >= 0


def test_failure_keeps_sequence_alignment() -> None:
    lengths = [5, 7, 9]
    result = UniversalBenchmark("test").benchmark_model(
        FailsAtLength, {"dim": 8}, lengths, num_runs=1, warmup_runs=1
    )
    assert [record["seq_len"] for record in result.records] == lengths
    assert [record["status"] for record in result.records] == ["ok", "error", "ok"]
    assert result.times[1] is None
    assert "intentional failure" in result.records[1]["error"]


def test_legacy_tuple_unpacking_still_works() -> None:
    result = UniversalBenchmark("test").benchmark_model(
        IdentityModel, {"dim": 8}, [4], num_runs=1, warmup_runs=0
    )
    times, memories = result
    assert times == result.times
    assert memories == result.memories
