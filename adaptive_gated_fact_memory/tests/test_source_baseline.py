from __future__ import annotations

import unittest

import torch

from adaptive_fact_memory import StreamingSWASourcePolicy


class StreamingSourceBaselineTests(unittest.TestCase):
    def test_source_policy_keeps_sinks_and_recent_keys(self) -> None:
        policy = StreamingSWASourcePolicy(window_size=8, sink_tokens=2)
        # One layer, [batch, heads, sequence, head_dim].  Values identify the
        # original position, making the retained positions auditable.
        keys = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1)
        values = keys + 100
        trimmed = policy([[keys, values]])
        self.assertEqual(trimmed[0][0].shape[2], 8)
        torch.testing.assert_close(
            trimmed[0][0].flatten(),
            torch.tensor([0, 1, 4, 5, 6, 7, 8, 9], dtype=torch.float32),
        )
        torch.testing.assert_close(
            trimmed[0][1].flatten(),
            torch.tensor([100, 101, 104, 105, 106, 107, 108, 109], dtype=torch.float32),
        )

    def test_source_policy_noop_below_budget(self) -> None:
        policy = StreamingSWASourcePolicy(window_size=8, sink_tokens=2)
        keys = torch.randn(1, 2, 7, 4)
        values = torch.randn(1, 2, 7, 4)
        result = policy([[keys, values]])
        self.assertIs(result[0][0], keys)
        self.assertIs(result[0][1], values)


if __name__ == "__main__":
    unittest.main()
