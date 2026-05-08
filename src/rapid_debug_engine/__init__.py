"""rapid-debug-engine-sdk — Python SDK for the Rapid Debug Engine.

Installed into `rapid-adk-requirements` and `rapid-adk-transformation` to
register agents, sync configs, and enrich OpenTelemetry spans with the
identifiers the Debug Engine backend uses to stitch traces to the registry.
"""

from .client import DebugEngineClient
from .config_sync import AgentConfigOverride, AgentConfigRegistry
from .liveness import AgentDefinition, introspect_agent_tree
from .observe import observe
from .registry import AgentRegistration, ModulesFile
from .shadow import (
    ShadowExperimentContext,
    get_active_experiment,
    set_active_experiment,
    shadow_aware,
)
from .tags import enrich_span, set_trace_agent_context

__all__ = [
    "DebugEngineClient",
    "AgentConfigOverride",
    "AgentConfigRegistry",
    "AgentDefinition",
    "AgentRegistration",
    "ModulesFile",
    "introspect_agent_tree",
    "observe",
    "ShadowExperimentContext",
    "get_active_experiment",
    "set_active_experiment",
    "shadow_aware",
    "enrich_span",
    "set_trace_agent_context",
]


def __getattr__(name: str):
    """Lazy re-export of DebugEngineMiddleware — only imported when requested.

    Starlette (FastAPI's base) is an optional dependency for the SDK; users
    who don't need the middleware helper shouldn't have to install it.
    """
    if name == "DebugEngineMiddleware":
        from .middleware import DebugEngineMiddleware

        return DebugEngineMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"
