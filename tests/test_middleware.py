"""Tests for ObservatoryMiddleware — verifies route → agent id resolution.

The middleware degrades gracefully when no OpenTelemetry SDK is installed
(tags.py enrichment functions become no-ops), so we only need to assert the
resolution logic itself: given a path, does it look up the right agent id
from the client cache?
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rapid_observatory.middleware import ObservatoryMiddleware


class FakeClient:
    """Stand-in for ObservatoryClient — just enough surface for the middleware."""

    def __init__(self) -> None:
        self._lookup: dict[tuple[str, str], str] = {}
        self.deployment_id = "acme"
        self.service_name = "rapid-adk-requirements"

    def set(self, category: str, name: str, agent_id: str) -> None:
        self._lookup[(category, name)] = agent_id

    def get_agent_id(self, category: str, name: str) -> str | None:
        return self._lookup.get((category, name))


def test_resolve_agent_from_named_capture() -> None:
    client = FakeClient()
    client.set("requirements", "apis", "agt_123")

    middleware = ObservatoryMiddleware(
        app=MagicMock(),
        client=client,  # type: ignore[arg-type]
        route_map={
            r"/api/requirements/(?P<name>[^/]+)$": ("requirements", None),
        },
    )

    resolved = middleware._resolve_agent("/api/requirements/apis")
    assert resolved == ("requirements", "apis")


def test_resolve_agent_explicit_name_override() -> None:
    client = FakeClient()
    middleware = ObservatoryMiddleware(
        app=MagicMock(),
        client=client,  # type: ignore[arg-type]
        route_map={
            r"^/health$": ("utility", "health-probe"),
        },
    )
    resolved = middleware._resolve_agent("/health")
    assert resolved == ("utility", "health-probe")


def test_resolve_agent_returns_none_for_unmatched_paths() -> None:
    middleware = ObservatoryMiddleware(
        app=MagicMock(),
        client=FakeClient(),  # type: ignore[arg-type]
        route_map={
            r"/api/requirements/(?P<name>[^/]+)$": ("requirements", None),
        },
    )
    assert middleware._resolve_agent("/unrelated/path") is None


def test_lazy_import_of_middleware_from_package() -> None:
    """Importing ObservatoryMiddleware from the package root must not fail
    just because starlette wasn't needed. This exercises the __getattr__
    shim in __init__.py."""
    import rapid_observatory

    attr = rapid_observatory.ObservatoryMiddleware
    assert attr is ObservatoryMiddleware

    with pytest.raises(AttributeError):
        rapid_observatory.DoesNotExist  # type: ignore[attr-defined]
