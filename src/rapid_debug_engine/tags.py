"""Helpers to enrich OpenTelemetry spans with rapid.* attributes.

The Debug Engine ingest pipeline reads these attributes from incoming OTLP
spans to stitch traces back to the agent registry. Without them, spans
arrive as orphans and can only be grouped by service name.

OpenTelemetry is an optional dependency — if the SDK is not installed,
these functions degrade to no-ops so the agent service still boots.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as _otel_trace

    _otel_available = True
except ImportError:  # pragma: no cover — optional extra
    _otel_trace = None  # type: ignore[assignment]
    _otel_available = False


def enrich_span(
    *,
    agent_id: str | None = None,
    agent_category: str | None = None,
    agent_name: str | None = None,
    module_id: str | None = None,
    screen_id: str | None = None,
    deployment_id: str | None = None,
    config_version_id: str | None = None,
    tenant_id: str | None = None,
) -> None:
    """Attach rapid.* attributes to the current active span.

    Safe to call from anywhere in the request path. Silently no-ops if
    OpenTelemetry is not installed or there is no active span.

    `agent_id` is the server-assigned Debug Engine cuid. If you don't have
    that handy in the route handler, pass `agent_category` + `agent_name`
    instead — the Debug Engine mapper resolves them at ingest time using
    the (deployment_id, service_name, category, name) registry tuple.
    """
    if not _otel_available or _otel_trace is None:
        return

    span = _otel_trace.get_current_span()
    if span is None:  # pragma: no cover
        return

    attrs: dict[str, Any] = {}
    if agent_id is not None:
        attrs["rapid.agent_id"] = agent_id
    if agent_category is not None:
        attrs["rapid.agent_category"] = agent_category
    if agent_name is not None:
        attrs["rapid.agent_name"] = agent_name
    if module_id is not None:
        attrs["rapid.module_id"] = module_id
    if screen_id is not None:
        attrs["rapid.screen_id"] = screen_id
    if deployment_id is not None:
        attrs["rapid.deployment_id"] = deployment_id
    if config_version_id is not None:
        attrs["rapid.config_version_id"] = config_version_id
    if tenant_id is not None:
        attrs["rapid.tenant_id"] = tenant_id

    for key, value in attrs.items():
        span.set_attribute(key, value)


def set_trace_agent_context(
    *,
    agent_id: str,
    deployment_id: str,
    service_name: str,
) -> None:
    """Convenience: set the minimum trace-level context for an agent invocation.

    Called from the top of an agent route handler to ensure every child span
    in the request inherits the identifiers the Debug Engine needs.
    """
    enrich_span(
        agent_id=agent_id,
        deployment_id=deployment_id,
    )
    if not _otel_available or _otel_trace is None:
        return
    span = _otel_trace.get_current_span()
    if span is None:  # pragma: no cover
        return
    span.set_attribute("rapid.service_name", service_name)
