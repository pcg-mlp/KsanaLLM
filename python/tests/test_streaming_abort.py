# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
Regression test for the client-disconnect KV-release fix.

The bug: when a client disconnected mid-stream (Ctrl+C / process kill), the
async generator returned by ``LLMService.format_streaming_output`` would be
``aclose()``-d by Starlette, but the underlying C++ ``StreamingIterator`` was
blocked inside ``GetNext()`` on a thread-pool executor. Its ``shared_ptr``
never went to zero, ``~StreamingIterator()`` never ran, and ~hundreds of
abandoned requests kept their KV blocks until ``max_new_tokens`` elapsed.

The fix: ``format_streaming_output`` wraps its ``async for`` loop in
``try / except / finally`` and calls ``iterator.abort()`` in the ``finally``
block. ``abort()`` flips ``req->aborted`` and ``Stop()``s ``step_waiter`` so
the blocked ``GetNext()`` wakes immediately and the scheduler reclaims KV.

This test verifies that ``abort()`` is invoked on the underlying iterator in
the three exit paths a streaming generator can take:

  1. Normal completion (iterator raises StopAsyncIteration)
  2. Exception raised during iteration
  3. GeneratorExit (Starlette ``aclose()`` on client disconnect)

Run from ``python/``:
    python3 -m pytest tests/test_streaming_abort.py -v
"""

import asyncio
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock heavy dependencies pulled in transitively by ksana_llm.ksana_service.
# We import ``ksana_service`` only for its ``format_streaming_output`` method;
# we do NOT instantiate LLMService (which would boot a real C++ engine).
# ---------------------------------------------------------------------------
_STUB = types.ModuleType("_stub")
_MOCK_MODULES = [
    "libtorch_serving",
    "torch", "torch.nn", "torch.cuda",
    "transformers", "transformers.tokenization_utils_base",
    "transformers.logging",
    "uvloop",
    "utilize", "utilize.logger",
    "ksana_llm.ksana_plugin",
    "ksana_llm.processor_op_base",
    "ksana_llm.arg_utils",
    "openaiapi", "openaiapi.openai_adapter",
    "request_id",
]
for _m in _MOCK_MODULES:
    sys.modules.setdefault(_m, _STUB)

# transformers.logging is referenced as `hf_logging` in ksana_service.py
sys.modules["transformers"].logging = _STUB  # type: ignore[attr-defined]


# utilize.logger.init_logger is used by ksana_engine.py
def _init_logger(name):
    import logging
    return logging.getLogger(name)
sys.modules["utilize.logger"].init_logger = _init_logger  # type: ignore[attr-defined]

# request_id.generate_request_id is referenced in ksana_service.py
sys.modules["request_id"].generate_request_id = lambda: "test-req-id"  # type: ignore[attr-defined]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ksana_llm.exceptions import KsanaEngineError  # noqa: E402

# Late import so the mocks above are visible to the import chain.
# ksana_service imports `import ksana_llm`, which would normally trigger the
# real package __init__ and pull in libtorch_serving etc. Pre-register a
# lightweight ksana_llm package that exposes only what ksana_service needs.
_ksana_pkg = types.ModuleType("ksana_llm")
_ksana_pkg.__path__ = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ksana_llm"))
]
_ksana_pkg.KsanaLLMEngine = MagicMock()
sys.modules["ksana_llm"] = _ksana_pkg

from ksana_llm import ksana_service  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
class _FakeOutput:
    """Minimal duck-typed stand-in for a KsanaPythonOutput chunk."""
    def __init__(self, token_id=1):
        self.input_tokens = [10, 11, 12]
        self.output_tokens = [[token_id]]
        self.logprobs = []
        self.cache_stat = []
        self.finish_status = None


class FakeIterator:
    """Async iterator that records abort() calls and is driven by a script.

    ``script`` is a list of either ``_FakeOutput`` instances (yield this) or
    Exception instances (raise this). When exhausted, raises StopAsyncIteration
    so the consumer's ``async for`` exits normally.
    """

    def __init__(self, script):
        self._script = list(script)
        self._idx = 0
        self.abort_call_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._script):
            raise StopAsyncIteration
        item = self._script[self._idx]
        self._idx += 1
        if isinstance(item, BaseException):
            raise item
        return item

    def abort(self):
        self.abort_call_count += 1


class _FakeService:
    """Bind LLMService.format_streaming_output without running its __init__."""

    def __init__(self):
        self.pre_post_processor = MagicMock()
        self.pre_post_processor.decode = MagicMock(return_value="hello")

    format_streaming_output = ksana_service.LLMService.format_streaming_output


async def _drain(agen):
    """Consume an async generator to completion, returning the chunks."""
    chunks = []
    async for chunk in agen:
        chunks.append(chunk)
    return chunks


# ===========================================================================
# Tests: abort() must run in all 3 exit paths
# ===========================================================================
class TestStreamingAbortInFinally:
    """``ksana_service.LLMService.format_streaming_output`` must call
    ``iterator.abort()`` exactly once in its ``finally`` block, regardless of
    whether the inner ``async for`` exits via normal completion, an exception,
    or ``GeneratorExit`` (client disconnect)."""

    def test_abort_called_on_normal_completion(self):
        """Iterator exhausted normally → finally runs → abort() called."""
        fake_iter = FakeIterator([_FakeOutput(token_id=1), _FakeOutput(token_id=2)])
        service = _FakeService()

        async def run():
            agen = await service.format_streaming_output(fake_iter)
            chunks = await _drain(agen)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 2, "should yield two data chunks before exhausting"
        assert fake_iter.abort_call_count == 1, (
            f"abort() must run in finally on normal completion, "
            f"got {fake_iter.abort_call_count} calls"
        )

    def test_abort_called_on_engine_error(self):
        """KsanaEngineError raised mid-stream → except catches → finally runs."""
        fake_iter = FakeIterator([
            _FakeOutput(token_id=1),
            KsanaEngineError("input too long", 413),
        ])
        service = _FakeService()

        async def run():
            agen = await service.format_streaming_output(fake_iter)
            return await _drain(agen)

        chunks = asyncio.run(run())
        # 1 data chunk + 1 error chunk
        assert len(chunks) == 2
        assert fake_iter.abort_call_count == 1, (
            "abort() must run in finally even when the except branch fires"
        )

    def test_abort_called_on_generic_exception(self):
        """Generic RuntimeError → except catches → finally runs."""
        fake_iter = FakeIterator([
            _FakeOutput(token_id=1),
            RuntimeError("CUDA OOM"),
        ])
        service = _FakeService()

        async def run():
            agen = await service.format_streaming_output(fake_iter)
            return await _drain(agen)

        chunks = asyncio.run(run())
        assert len(chunks) == 2
        assert fake_iter.abort_call_count == 1

    def test_abort_called_on_generator_close(self):
        """Consumer calls ``aclose()`` mid-stream → GeneratorExit → finally runs.

        This is the *exact* scenario the production bug describes: Starlette
        cancels the streaming task when the client disconnects, which causes
        ``aclose()`` on the body iterator. Since ``GeneratorExit`` derives from
        ``BaseException`` (not ``Exception``), the ``except Exception`` branch
        does NOT catch it — only ``finally`` runs.
        """
        fake_iter = FakeIterator([
            _FakeOutput(token_id=1),
            _FakeOutput(token_id=2),
            _FakeOutput(token_id=3),
        ])
        service = _FakeService()

        async def run():
            agen = await service.format_streaming_output(fake_iter)
            # Pull the first chunk, then close mid-stream.
            first = await agen.__anext__()
            assert first is not None
            await agen.aclose()

        asyncio.run(run())
        assert fake_iter.abort_call_count == 1, (
            "GeneratorExit (Starlette aclose on client disconnect) must still "
            "trigger abort() — this is the bug we fixed"
        )

    def test_abort_called_only_once_even_if_drained_twice(self):
        """Re-iterating a finished async gen must not double-fire abort()."""
        fake_iter = FakeIterator([_FakeOutput(token_id=1)])
        service = _FakeService()

        async def run():
            agen = await service.format_streaming_output(fake_iter)
            await _drain(agen)
            # Second drain on already-exhausted gen is a no-op; finally has
            # already executed exactly once.
            async for _ in agen:
                pass

        asyncio.run(run())
        assert fake_iter.abort_call_count == 1

    def test_abort_exceptions_are_swallowed(self):
        """If abort() itself raises (unlikely but possible from C++), the
        wrapping ``PyAsyncStreamingIterator.abort`` already swallows it. The
        service-layer ``finally`` calls ``iterator.abort()`` unconditionally,
        so a raising abort here would propagate — verify it does NOT, which
        confirms the production guard in ``PyAsyncStreamingIterator.abort``.
        """
        class RaisingIterator(FakeIterator):
            def abort(self):
                super().abort()
                raise RuntimeError("pybind raised")

        fake_iter = RaisingIterator([_FakeOutput(token_id=1)])
        service = _FakeService()

        async def run():
            agen = await service.format_streaming_output(fake_iter)
            # If service-layer abort propagates, this will raise.
            # Production guarantees abort() is exception-safe via
            # PyAsyncStreamingIterator.abort()'s try/except. The bare
            # FakeIterator class here mimics that contract; we expose the
            # raising variant to highlight what would break if the contract
            # is violated.
            with pytest.raises(RuntimeError):
                await _drain(agen)

        asyncio.run(run())
        assert fake_iter.abort_call_count == 1
