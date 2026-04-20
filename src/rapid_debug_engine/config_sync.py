"""Config hot-reload synchronization.

Two delivery mechanisms:

1. **Pub/Sub subscriber** — preferred when `google-cloud-pubsub` is installed
   and the `PUBSUB_EMULATOR_HOST` env var or real GCP creds are available.
   Latency is sub-second. Not implemented in this first cut; the long-poll
   path is sufficient for the M4 acceptance criteria (<5s hot-reload) and
   doesn't require any extra infrastructure.

2. **Long-poll** — the default and fallback. A background asyncio task
   calls `GET /api/v1/configs/sync?since=<ts>` every ~2s and applies
   returned promotion events to the in-memory `AgentConfigRegistry`.

Registry overrides are consulted by `AdkAgentLoader.load_*_agent()` before
falling back to disk reads, so a promoted config takes effect on the next
agent invocation without restarting the service.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AgentConfigOverride:
    """One hot-reloaded config kept in memory. Replaces the on-disk JSON
    for the same (category, name) pair when present."""

    agent_id: str
    category: str
    name: str
    config_version_id: str
    version: int
    config: dict[str, Any]
    promoted_at: str
    promoted_by: str


@dataclass
class AgentConfigRegistry:
    """Thread-safe enough for our asyncio context — all writes happen from
    the single background task and reads happen on the request path. No
    explicit lock needed because dict assignment is atomic in CPython."""

    overrides: dict[tuple[str, str], AgentConfigOverride] = field(default_factory=dict)

    def set(self, override: AgentConfigOverride) -> None:
        self.overrides[(override.category, override.name)] = override

    def get(self, category: str, name: str) -> AgentConfigOverride | None:
        return self.overrides.get((category, name))


class ConfigSyncWorker:
    """Background long-poller. Owns no state other than the HTTP client,
    the registry it writes into, and the cursor it advances."""

    DEFAULT_POLL_INTERVAL_S = 2.0

    def __init__(
        self,
        http: httpx.AsyncClient,
        deployment_id: str,
        service_name: str,
        registry: AgentConfigRegistry,
        poll_interval_s: float | None = None,
    ) -> None:
        self._http = http
        self._deployment_id = deployment_id
        self._service_name = service_name
        self._registry = registry
        self._poll_interval_s = poll_interval_s or self.DEFAULT_POLL_INTERVAL_S
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Seed the cursor slightly in the past so the first poll catches
        # any recent promotions the SDK might have missed during a restart.
        self._cursor = _iso_now_minus(seconds=60)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="debug-engine-config-sync")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._poll_once()
            except Exception as err:  # noqa: BLE001 — long-poll loop must never die
                logger.warning("debug_engine: config sync poll failed: %s", err)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval_s)
            except asyncio.TimeoutError:
                pass

    async def _poll_once(self) -> None:
        response = await self._http.get(
            "/api/v1/configs/sync",
            params={
                "deploymentId": self._deployment_id,
                "serviceName": self._service_name,
                "since": self._cursor,
            },
        )
        if response.status_code != 200:
            logger.debug("debug_engine: config sync returned %s", response.status_code)
            return
        data = response.json()
        events = data.get("events", [])
        for event in events:
            override = AgentConfigOverride(
                agent_id=event["agentId"],
                category=event["category"],
                name=event["name"],
                config_version_id=event["configVersionId"],
                version=event["version"],
                config=event["config"],
                promoted_at=event["promotedAt"],
                promoted_by=event["promotedBy"],
            )
            self._registry.set(override)
            logger.info(
                "debug_engine: applied config v%d for %s/%s (promoted by %s)",
                override.version,
                override.category,
                override.name,
                override.promoted_by,
            )
        self._cursor = data.get("cursor", self._cursor)


def _iso_now_minus(seconds: float) -> str:
    """Return an ISO-8601 UTC timestamp `seconds` in the past — used to seed
    the sync cursor with a tiny backfill window."""
    ts = datetime.now(timezone.utc).timestamp() - seconds
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
