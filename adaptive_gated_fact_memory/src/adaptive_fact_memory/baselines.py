"""Adapters that delegate baseline mechanism logic to frozen upstream code.

The source-backed policy in this module contains no alternative
sliding-window algorithm.  It delegates cache trimming to
``streaming_llm.kv_cache.StartRecentKVCache`` from the downloaded upstream
revision.  The wrapper only supplies project metadata and a stable interface
for later TinyLM/Hugging Face runners.

The upstream object operates on Hugging Face-style ``past_key_values``.  The
current adaptive TinyLM uses a different explicit ``LocalKVCache`` state, so
this policy is not silently wired into that model; a runner must make the
format conversion explicit and report it as a protocol adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SOURCE_URL = "https://github.com/mit-han-lab/streaming-llm"
SOURCE_REVISION = "2e5042606d69933d88fbf909bd77907456b9b4dd"


def _upstream_cache_class():
    """Load the frozen upstream class without vendoring or reimplementing it."""

    reference_root = Path(__file__).resolve().parents[2] / "references" / "streaming-llm"
    if not reference_root.is_dir():
        raise FileNotFoundError(
            "StreamingLLM reference is missing at "
            f"{reference_root}; run scripts/fetch_references.sh first"
        )

    # The upstream checkout is intentionally kept outside the package import
    # tree and is ignored by the parent Git repository.
    import sys

    root_text = str(reference_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from streaming_llm.kv_cache import StartRecentKVCache

    return StartRecentKVCache


class StreamingSWASourcePolicy:
    """Source-backed ``start sinks + recent window`` cache policy.

    Parameters use the paper's notation: ``window_size`` is the total number
    of retained keys and ``sink_tokens`` is the number of preserved initial
    keys.  The upstream implementation receives the corresponding
    ``start_size`` and ``recent_size`` values.
    """

    source_url = SOURCE_URL
    source_revision = SOURCE_REVISION

    def __init__(
        self,
        *,
        window_size: int = 128,
        sink_tokens: int = 4,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if sink_tokens < 0 or sink_tokens >= window_size:
            raise ValueError("sink_tokens must satisfy 0 <= sink_tokens < window_size")
        if k_seq_dim not in (1, 2, 3) or v_seq_dim not in (1, 2, 3):
            raise ValueError("k_seq_dim and v_seq_dim must be 1, 2, or 3")

        self.window_size = int(window_size)
        self.sink_tokens = int(sink_tokens)
        self.recent_window = self.window_size - self.sink_tokens
        cache_class = _upstream_cache_class()
        self._upstream = cache_class(
            start_size=self.sink_tokens,
            recent_size=self.recent_window,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )

    def __call__(self, past_key_values: Any) -> Any:
        """Trim an existing cache using the upstream implementation."""

        return self._upstream(past_key_values)

    def evict_for_space(self, past_key_values: Any, num_coming: int) -> Any:
        """Delegate pre-allocation eviction to the upstream implementation."""

        if num_coming < 0:
            raise ValueError("num_coming must be non-negative")
        return self._upstream.evict_for_space(past_key_values, int(num_coming))

    def evict_range(self, past_key_values: Any, start: int, end: int) -> Any:
        """Delegate range eviction to the upstream implementation."""

        return self._upstream.evict_range(past_key_values, int(start), int(end))

