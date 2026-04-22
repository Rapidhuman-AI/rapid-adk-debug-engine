"""Tests for @observe decorator — auto-instrumentation across sync and async.

Uses the OTel SDK in-memory exporter so assertions run without a real
collector. When OTel isn't installed the decorator is a pass-through
(no-op); those paths are covered by a minimal smoke test.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from rapid_debug_engine import observe

pytest.importorskip("opentelemetry")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

# Install a global tracer provider once per process.
_provider = TracerProvider()
_exporter = InMemorySpanExporter()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


def setup_function() -> None:
    _exporter.clear()


def _attr(span, key: str):
    return span.attributes.get(key) if span.attributes else None


def test_observe_sync_function_produces_span_with_name() -> None:
    @observe
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    spans = _exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name.endswith(".add")


def test_observe_captures_input_and_output_as_json() -> None:
    @observe()
    def greet(name: str, formal: bool = False) -> str:
        return f"Hello {name}"

    greet("Alice", formal=True)
    spans = _exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_attr = _attr(span, "rapid.function_input")
    assert input_attr is not None
    parsed = json.loads(input_attr)
    assert parsed == {"name": "Alice", "formal": True}

    output_attr = _attr(span, "rapid.function_output")
    assert output_attr is not None
    assert json.loads(output_attr) == "Hello Alice"


def test_observe_custom_name_and_tags() -> None:
    @observe(name="custom-span", tags={"team": "platform", "feature": "chat"})
    def noop() -> None:
        return None

    noop()
    span = _exporter.get_finished_spans()[0]
    assert span.name == "custom-span"
    assert _attr(span, "rapid.tags.team") == "platform"
    assert _attr(span, "rapid.tags.feature") == "chat"


def test_observe_capture_input_false() -> None:
    @observe(capture_input=False)
    def f(secret: str) -> str:
        return "ok"

    f("hunter2")
    span = _exporter.get_finished_spans()[0]
    assert _attr(span, "rapid.function_input") is None
    assert _attr(span, "rapid.function_output") == json.dumps("ok")


def test_observe_capture_output_false() -> None:
    @observe(capture_output=False)
    def f() -> dict:
        return {"sensitive": "response"}

    f()
    span = _exporter.get_finished_spans()[0]
    assert _attr(span, "rapid.function_output") is None


def test_observe_async_function() -> None:
    @observe
    async def multiply(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a * b

    result = asyncio.run(multiply(3, 4))
    assert result == 12
    spans = _exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name.endswith(".multiply")


def test_observe_records_exception_and_sets_error_status() -> None:
    @observe()
    def fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        fail()

    span = _exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert any(ev.name == "exception" for ev in span.events)


def test_observe_truncates_very_large_outputs() -> None:
    big = "x" * 100_000

    @observe()
    def big_output() -> str:
        return big

    big_output()
    span = _exporter.get_finished_spans()[0]
    out = _attr(span, "rapid.function_output")
    assert out is not None
    assert "<truncated>" in out
    assert len(out) < len(big)
