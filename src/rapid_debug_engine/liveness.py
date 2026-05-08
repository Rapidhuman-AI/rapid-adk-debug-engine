"""Liveness + agent-introspection HTTP surface for the Debug Engine SDK.

Each ADK that uses this SDK can expose two endpoints to let the Debug Engine
backend confirm reachability and pull the canonical list of agents currently
present on this service:

    GET /health        — small JSON payload, 200 once the SDK has booted.
    GET /agents/list   — `{ service, agents: [AgentDefinition, ...] }` describing
                         every agent the service currently has, including
                         single-vs-multi-agent composition when a runtime
                         Google ADK agent tree is supplied.

Wire-up (one line) in an ADK's FastAPI app:

    debug_engine.register_liveness_routes(app, agent_tree_provider=lambda: ROOT_AGENT)

If the ADK has no stable runtime root agent, omit `agent_tree_provider` and
the static `agentconfig/agents/<category>/<name>.json` discovery is used as
the fallback (every agent reports as `kind="single"`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .registry import AgentRegistration, discover_agents

logger = logging.getLogger(__name__)

ORCHESTRATOR_CLASS_NAMES = frozenset(
    {"SequentialAgent", "ParallelAgent", "LoopAgent"}
)


@dataclass
class AgentDefinition:
    """Wire shape returned by GET /agents/list.

    `kind="single"` means a leaf LLM agent; `kind="multi"` means an
    orchestrator (Sequential/Parallel/Loop) whose `sub_agents` carry the
    children. `output_structure` is either a Pydantic JSON schema dict
    (when the ADK runtime exposes `output_schema`) or the raw string from
    the agent's JSON config (static fallback).
    """

    category: str
    name: str
    kind: str  # "single" | "multi"
    model: str | None = None
    system_message: str | None = None
    tools: list[str] = field(default_factory=list)
    output_structure: Any | None = None
    sub_agents: list["AgentDefinition"] = field(default_factory=list)
    orchestrator: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Runtime introspection ──────────────────────────────────────────────────


def introspect_agent_tree(root: Any) -> list[AgentDefinition]:
    """Walk a Google ADK agent tree and return its definition list.

    Duck-typed: never imports `google.adk` so the SDK stays usable in
    services that haven't installed it. The `root` may be a single agent or
    an iterable of top-level agents.
    """
    if root is None:
        return []
    if isinstance(root, (list, tuple, set)):
        return [_describe(a) for a in root]
    return [_describe(root)]


def _describe(agent: Any) -> AgentDefinition:
    cls_name = type(agent).__name__
    raw_subs = getattr(agent, "sub_agents", None) or []
    sub_list = list(raw_subs) if isinstance(raw_subs, Iterable) else []
    is_orchestrator = cls_name in ORCHESTRATOR_CLASS_NAMES
    is_multi = is_orchestrator or bool(sub_list)

    return AgentDefinition(
        category=str(getattr(agent, "category", None) or _category_for(cls_name)),
        name=str(getattr(agent, "name", None) or cls_name),
        kind="multi" if is_multi else "single",
        model=_extract_model(agent),
        system_message=_extract_instruction(agent),
        tools=[_tool_name(t) for t in getattr(agent, "tools", None) or []],
        output_structure=_extract_output_schema(agent),
        sub_agents=[_describe(c) for c in sub_list],
        orchestrator=cls_name if is_orchestrator else None,
    )


def _category_for(cls_name: str) -> str:
    if cls_name in ORCHESTRATOR_CLASS_NAMES:
        return "orchestrator"
    return "agent"


def _extract_model(agent: Any) -> str | None:
    model = getattr(agent, "model", None)
    if model is None:
        return None
    if isinstance(model, str):
        return model
    return getattr(model, "name", None) or getattr(model, "model", None) or str(model)


def _extract_instruction(agent: Any) -> str | None:
    for attr in ("instruction", "system_message", "system_instruction"):
        value = getattr(agent, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                resolved = value({})
            except Exception:  # noqa: BLE001 — never break introspection
                continue
            if isinstance(resolved, str):
                return resolved
        if isinstance(value, str):
            return value
    return None


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    fn_name = getattr(tool, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name
    return type(tool).__name__


def _extract_output_schema(agent: Any) -> Any | None:
    schema = getattr(agent, "output_schema", None)
    if schema is None:
        return None
    json_schema = getattr(schema, "model_json_schema", None)
    if callable(json_schema):
        try:
            return json_schema()
        except Exception:  # noqa: BLE001
            return None
    if isinstance(schema, dict):
        return schema
    return None


# ─── Static-config fallback ─────────────────────────────────────────────────


def _registration_to_definition(reg: AgentRegistration) -> AgentDefinition:
    """Convert a disk-discovered AgentRegistration to an AgentDefinition.

    The `_extract_config` step in registry.py merges `outputstructure` into
    `prompt`; reverse that here so the Debug Engine UI shows a clean system
    prompt and a separate output_structure field.
    """
    config = reg.config or {}
    output_structure = (config.get("meta") or {}).get("outputStructure")
    prompt = config.get("prompt") or ""
    if output_structure and isinstance(prompt, str):
        suffix = f"\n\nOutput structure: {output_structure}"
        if prompt.endswith(suffix):
            prompt = prompt[: -len(suffix)]
    tools = config.get("tools") or []
    return AgentDefinition(
        category=reg.category,
        name=reg.name,
        kind="single",
        model=config.get("model"),
        system_message=prompt or None,
        tools=[str(t) for t in tools],
        output_structure=output_structure,
        sub_agents=[],
        orchestrator=None,
    )


def _static_definitions(
    service_name: str,
    agentconfig_dir: Path,
    modules_file: Path,
) -> list[AgentDefinition]:
    registrations = discover_agents(agentconfig_dir, modules_file, service_name)
    return [_registration_to_definition(r) for r in registrations]


# ─── FastAPI router ─────────────────────────────────────────────────────────


def build_liveness_router(
    *,
    service_name: str,
    sdk_version: str,
    agent_tree_provider: Callable[[], Any] | None = None,
    agentconfig_dir: Path,
    modules_file: Path,
) -> Any:
    """Build the FastAPI router that serves /health + /agents/list.

    Returns an `APIRouter`. Imports `fastapi` lazily so the SDK stays
    importable in services that haven't installed the FastAPI extra.
    """
    try:
        from fastapi import APIRouter
    except ImportError as err:  # pragma: no cover — install error path
        raise RuntimeError(
            "rapid-debug-engine: register_liveness_routes requires FastAPI. "
            "Install with `pip install rapid-debug-engine-sdk[fastapi]`."
        ) from err

    started_at = time.monotonic()
    router = APIRouter()

    @router.get("/health")
    async def _health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": service_name,
            "uptime_s": round(time.monotonic() - started_at, 3),
            "version": sdk_version,
        }

    @router.get("/agents/list")
    async def _agents_list() -> dict[str, Any]:
        agents = _resolve_agents(
            service_name=service_name,
            agent_tree_provider=agent_tree_provider,
            agentconfig_dir=agentconfig_dir,
            modules_file=modules_file,
        )
        return {
            "service": service_name,
            "agents": [a.to_dict() for a in agents],
        }

    return router


def _resolve_agents(
    *,
    service_name: str,
    agent_tree_provider: Callable[[], Any] | None,
    agentconfig_dir: Path,
    modules_file: Path,
) -> list[AgentDefinition]:
    if agent_tree_provider is not None:
        try:
            root = agent_tree_provider()
        except Exception:  # noqa: BLE001
            logger.exception("debug_engine: agent_tree_provider raised; falling back to static")
        else:
            try:
                agents = introspect_agent_tree(root)
            except Exception:  # noqa: BLE001
                logger.exception("debug_engine: introspection failed; falling back to static")
            else:
                if agents:
                    return agents
    return _static_definitions(service_name, agentconfig_dir, modules_file)
