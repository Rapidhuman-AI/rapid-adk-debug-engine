"""ObservatoryClient — the public surface the agent services use."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from .config_sync import AgentConfigOverride, AgentConfigRegistry, ConfigSyncWorker
from .registry import AgentRegistration, discover_agents

logger = logging.getLogger(__name__)

DEFAULT_AGENTCONFIG_DIR = Path("agentconfig/agents")
DEFAULT_MODULES_FILE = Path("agentconfig/modules.json")


class ObservatoryClient:
    """Async client for the AI Agent Observatory REST API.

    Usage (M1 scope):

        observatory = ObservatoryClient(
            base_url="http://localhost:8080",
            api_key="...",
            deployment_id="acme",
            service_name="rapid-adk-requirements",
        )
        await observatory.register_on_startup()
        ...
        await observatory.stop()

    M4 adds `start_config_sync`, M7 adds `get_shadow_experiment`, M8 adds
    `record_guardrail_event`. Their stubs are included below so the import
    surface is stable across milestones.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        deployment_id: str,
        service_name: str,
        environment: str = "development",
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._deployment_id = deployment_id
        self._service_name = service_name
        self._environment = environment
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "content-type": "application/json",
                "x-internal-api-key": api_key,
            },
            timeout=timeout,
        )
        # Populated after register_on_startup — maps (category, name) → server agent_id
        self._agent_id_cache: dict[tuple[str, str], str] = {}
        # Hot-reload registry — written by the long-poll worker, read by
        # AdkAgentLoader overrides on every agent invocation.
        self._config_registry = AgentConfigRegistry()
        self._sync_worker: ConfigSyncWorker | None = None

    @property
    def deployment_id(self) -> str:
        return self._deployment_id

    @property
    def service_name(self) -> str:
        return self._service_name

    def get_agent_id(self, category: str, name: str) -> str | None:
        """Resolve a (category, name) pair to the server-assigned agent id.

        Used by the shadow-aware decorator and span enrichment helpers to
        know which agent a given route is invoking.
        """
        return self._agent_id_cache.get((category, name))

    async def register_on_startup(
        self,
        agentconfig_dir: Path = DEFAULT_AGENTCONFIG_DIR,
        modules_file: Path = DEFAULT_MODULES_FILE,
    ) -> list[dict[str, Any]]:
        """Walk the agent config directory and POST to /api/v1/agents/register.

        Idempotent: safe to call on every boot. The Observatory upserts by
        (deployment_id, service_name, category, name) and only creates a new
        immutable AgentConfig version when configHash changes.

        Returns the server-side agent list (also cached in self._agent_id_cache).
        """
        registrations = discover_agents(agentconfig_dir, modules_file)
        if not registrations:
            logger.warning("observatory: no agents discovered — nothing to register")
            return []

        payload = self._build_registration_payload(registrations)

        try:
            response = await self._http.post("/api/v1/agents/register", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as err:
            logger.error("observatory: registration failed: %s", err)
            return []

        data = response.json()
        agents = data.get("agents", [])
        for agent in agents:
            key = (agent["category"], agent["name"])
            self._agent_id_cache[key] = agent["id"]

        logger.info(
            "observatory: registered %d agents with deployment=%s service=%s",
            len(agents),
            self._deployment_id,
            self._service_name,
        )
        return agents

    def _build_registration_payload(
        self, registrations: list[AgentRegistration]
    ) -> dict[str, Any]:
        return {
            "deploymentId": self._deployment_id,
            "serviceName": self._service_name,
            "environment": self._environment,
            "agents": [r.to_payload() for r in registrations],
        }

    # ─── Hot-reload (M4) ─────────────────────────────────────────────────────
    async def start_config_sync(self, poll_interval_s: float | None = None) -> None:
        """Start the background config-sync worker.

        Polls `/api/v1/configs/sync` every ~2s by default. Returned promotion
        events are applied to an in-memory `AgentConfigRegistry` that
        `AdkAgentLoader` can consult before falling back to disk.
        """
        if self._sync_worker is not None:
            return
        self._sync_worker = ConfigSyncWorker(
            http=self._http,
            deployment_id=self._deployment_id,
            service_name=self._service_name,
            registry=self._config_registry,
            poll_interval_s=poll_interval_s,
        )
        await self._sync_worker.start()
        logger.info("observatory: config sync started (long-poll)")

    def get_active_config_override(
        self, category: str, name: str
    ) -> AgentConfigOverride | None:
        """Called by AdkAgentLoader to check for a hot-reloaded config
        before loading from disk. Returns None when no override is active."""
        return self._config_registry.get(category, name)

    # ─── M7 stubs ────────────────────────────────────────────────────────────
    def get_shadow_experiment(self, agent_id: str) -> None:
        """M7: look up the active ShadowExperiment for this agent, if any."""
        _ = agent_id
        return None

    # ─── Guardrails (M8) ─────────────────────────────────────────────────────
    async def record_guardrail_event(
        self,
        trace_id: str,
        agent_id: str,
        schema_name: str,
        result: dict[str, Any],
        attempts: list[dict[str, Any]],
    ) -> None:
        """POST a Pydantic validation outcome to /api/v1/ingest/guardrails.

        Fire-and-forget by convention — guardrail ingest must never fail a
        user request. `result` is expected to carry a `status` key
        (`pass | fail | repair`) plus an optional `errors` dict from
        ValidationError.errors(). `attempts` is the list of retry payloads
        from validated_runner; we only persist the count.
        """
        status = str(result.get("status", "fail")).lower()
        if status not in {"pass", "fail", "repair"}:
            status = "fail"

        # Backend Zod schema treats these as `.optional()` — omit them
        # entirely when empty rather than sending null, which would fail
        # validation with "Expected object, received null".
        payload: dict[str, Any] = {
            "traceId": trace_id,
            "agentId": agent_id,
            "schemaName": schema_name,
            "status": status,
            "attempts": len(attempts) if attempts else 1,
        }
        errors = result.get("errors")
        if errors:
            payload["validatorErrors"] = errors
        raw_input = result.get("input")
        if raw_input is not None:
            payload["rawInput"] = raw_input
        raw_output = result.get("output")
        if raw_output is not None:
            payload["rawOutput"] = raw_output
        try:
            response = await self._http.post("/api/v1/ingest/guardrails", json=payload)
            if response.status_code >= 400:
                logger.debug(
                    "observatory: guardrail ingest returned %s: %s",
                    response.status_code,
                    response.text[:200],
                )
        except Exception as err:  # noqa: BLE001 — must never raise in hot path
            logger.debug("observatory: guardrail ingest failed: %s", err)

    async def stop(self) -> None:
        """Clean shutdown. Call from FastAPI lifespan teardown."""
        if self._sync_worker is not None:
            await self._sync_worker.stop()
            self._sync_worker = None
        await self._http.aclose()
