"""@observe decorator — auto-instrumentation for agent-side callables.

Wraps any sync or async function so each invocation becomes an OTel span,
with input/output captured as span attributes the Debug Engine ingest
pipeline persists. Closes the parity gap with Langfuse's `@observe` and
Langwatch's OTel-native auto-instrumentation.

Usage (identical to Langfuse's API so migration is mechanical):

    from rapid_debug_engine import observe

    @observe()
    async def generate_summary(prompt: str) -> str:
        return await llm.generate(prompt)

    @observe(name="custom-span-name", capture_input=False)
    def classify(payload: dict) -> str:
        ...

The produced span carries:
  - name: the decorator's `name=` or the function's qualname
  - rapid.function_input: JSON-serialized args/kwargs (unless capture_input=False)
  - rapid.function_output: JSON-serialized return value (unless capture_output=False)
  - rapid.tags.*: any tags passed via the decorator kwarg
  - gen_ai.*: if the function sets them via enrich_span on the current span
  - exception event if the function raises — span also gets statusCode=ERROR

Degrades to a no-op wrapper when OpenTelemetry is not installed. The
function still runs identically; nothing gets traced.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import Any, Awaitable, Callable, TypeVar, cast, overload

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status, StatusCode

    _otel_available = True
except ImportError:  # pragma: no cover — optional extra
    _otel_trace = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]
    _otel_available = False


_MAX_CAPTURE_BYTES = 32_000  # cap large payloads so spans stay ingestible


def _truncate(s: str) -> str:
    if len(s) <= _MAX_CAPTURE_BYTES:
        return s
    return s[:_MAX_CAPTURE_BYTES] + "...<truncated>"


def _safe_json(value: Any) -> str:
    try:
        return _truncate(json.dumps(value, default=str))
    except Exception:  # noqa: BLE001 — never fail the wrapped function
        return _truncate(repr(value))


def _capture_input(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return _safe_json(dict(bound.arguments))
    except Exception:  # noqa: BLE001
        return _safe_json({"args": args, "kwargs": kwargs})


def _apply_tags(span: Any, tags: dict[str, Any] | None) -> None:
    if not tags:
        return
    for k, v in tags.items():
        try:
            span.set_attribute(f"rapid.tags.{k}", v)
        except Exception:  # noqa: BLE001
            pass


@overload
def observe(__func: F) -> F: ...


@overload
def observe(
    *,
    name: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
    tags: dict[str, Any] | None = None,
) -> Callable[[F], F]: ...


def observe(
    __func: F | None = None,
    *,
    name: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
    tags: dict[str, Any] | None = None,
) -> Any:
    """Wrap a callable so its invocation becomes an OTel span.

    Can be used with or without arguments:

        @observe
        def f(...): ...

        @observe(name="custom", tags={"team": "platform"})
        async def g(...): ...
    """

    def decorate(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        # Fall back to a no-op wrapper when OTel is absent — keeps the SDK
        # usable in services that don't install the `otel` extra.
        if not _otel_available or _otel_trace is None:
            return func

        tracer = _otel_trace.get_tracer(__name__)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(span_name) as span:
                    _apply_tags(span, tags)
                    if capture_input:
                        try:
                            span.set_attribute(
                                "rapid.function_input",
                                _capture_input(func, args, kwargs),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        result = await cast(
                            Callable[..., Awaitable[Any]], func
                        )(*args, **kwargs)
                    except BaseException as exc:
                        try:
                            span.record_exception(exc)
                            if Status is not None and StatusCode is not None:
                                span.set_status(Status(StatusCode.ERROR, str(exc)))
                        except Exception:  # noqa: BLE001
                            pass
                        raise
                    if capture_output:
                        try:
                            span.set_attribute(
                                "rapid.function_output", _safe_json(result)
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    return result

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                _apply_tags(span, tags)
                if capture_input:
                    try:
                        span.set_attribute(
                            "rapid.function_input",
                            _capture_input(func, args, kwargs),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    try:
                        span.record_exception(exc)
                        if Status is not None and StatusCode is not None:
                            span.set_status(Status(StatusCode.ERROR, str(exc)))
                    except Exception:  # noqa: BLE001
                        pass
                    raise
                if capture_output:
                    try:
                        span.set_attribute(
                            "rapid.function_output", _safe_json(result)
                        )
                    except Exception:  # noqa: BLE001
                        pass
                return result

        return cast(F, sync_wrapper)

    # Usage: @observe (no parens) — __func is the decorated function.
    if __func is not None and callable(__func):
        return decorate(__func)

    # Usage: @observe(...) — return the decorator.
    return decorate


__all__ = ["observe"]
