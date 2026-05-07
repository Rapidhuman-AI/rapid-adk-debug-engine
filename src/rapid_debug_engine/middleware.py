"""FastAPI middleware helpers for the Debug Engine SDK.

Given a route like `POST /api/requirements/forapis`, the middleware looks up
the agent registered for that (category, name) pair and enriches every span
in the request with the Debug Engine identifiers. This means the existing
`services/otel_observability.py` instrumentation keeps working unchanged —
the SDK just layers stitching metadata on top.

Usage in `rapid-adk-requirements/main.py`:

    from rapid_debug_engine import DebugEngineClient
    from rapid_debug_engine.middleware import DebugEngineMiddleware

    debug_engine = DebugEngineClient(...)
    app.add_middleware(
        DebugEngineMiddleware,
        client=debug_engine,
        # Map HTTP path patterns to (category, name). The keys are regex
        # fragments matched against request.url.path.
        route_map={
            r"/api/requirements/(?P<name>[^/]+)$": ("requirements", None),
            r"/api/architecture/(?P<name>[^/]+)$": ("architecture", None),
            r"/api/transformation/(?P<name>[^/]+)$": ("transformation", None),
        },
    )
"""

from __future__ import annotations

import logging
import re
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .client import DebugEngineClient
from .tags import enrich_span, set_trace_agent_context

logger = logging.getLogger(__name__)

RouteMap = dict[str, tuple[str, str | None]]


class DebugEngineMiddleware(BaseHTTPMiddleware):
    """Stitches rapid.* attributes onto every span in a request.

    The decision of which agent the request belongs to comes from a
    user-supplied `route_map` that maps regex fragments against the path to
    (category, name_override) tuples. If `name_override` is None, the
    middleware extracts `name` from a named capture group in the regex.

    Compiled regexes are cached on construction, so the per-request cost is
    O(routes) string matches — fine for the double-digit route counts in
    `rapid-adk-requirements`/`rapid-adk-transformation`.
    """

    def __init__(
        self,
        app: Any,
        client: DebugEngineClient,
        route_map: RouteMap,
    ) -> None:
        super().__init__(app)
        self._client = client
        self._compiled = [(re.compile(pattern), target) for pattern, target in route_map.items()]

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        self._try_enrich(request)
        return await call_next(request)

    def _try_enrich(self, request: Request) -> None:
        agent_resolution = self._resolve_agent(request.url.path)
        if agent_resolution is None:
            return
        category, name = agent_resolution
        agent_id = self._client.get_agent_id(category, name)
        if agent_id is None:
            logger.debug(
                "debug_engine: no cached agent for (%s, %s); skipping enrichment",
                category,
                name,
            )
            return
        set_trace_agent_context(
            agent_id=agent_id,
            deployment_id=self._client.deployment_id,
            service_name=self._client.service_name,
        )
        enrich_span(
            agent_id=agent_id,
            deployment_id=self._client.deployment_id,
        )

    def _resolve_agent(self, path: str) -> tuple[str, str] | None:
        for pattern, (category, explicit_name) in self._compiled:
            match = pattern.search(path)
            if match is None:
                continue
            if explicit_name is not None:
                return category, explicit_name
            name = match.groupdict().get("name")
            if name:
                return category, name
        return None
