"""rapid-observatory-sdk — Python SDK for the AI Agent Observatory.

Installed into `rapid-adk-requirements` and `rapid-adk-transformation` to
register agents, sync configs, and enrich OpenTelemetry spans with the
identifiers the Observatory backend uses to stitch traces to the registry.
"""

from .client import ObservatoryClient
from .config_sync import AgentConfigOverride, AgentConfigRegistry
from .registry import AgentRegistration, ModulesFile
from .shadow import (
    ShadowExperimentContext,
    get_active_experiment,
    set_active_experiment,
    shadow_aware,
)
from .tags import enrich_span, set_trace_agent_context

__all__ = [
    "ObservatoryClient",
    "AgentConfigOverride",
    "AgentConfigRegistry",
    "AgentRegistration",
    "ModulesFile",
    "ShadowExperimentContext",
    "get_active_experiment",
    "set_active_experiment",
    "shadow_aware",
    "enrich_span",
    "set_trace_agent_context",
]


def __getattr__(name: str):
    """Lazy re-export of ObservatoryMiddleware — only imported when requested.

    Starlette (FastAPI's base) is an optional dependency for the SDK; users
    who don't need the middleware helper shouldn't have to install it.
    """
    if name == "ObservatoryMiddleware":
        from .middleware import ObservatoryMiddleware

        return ObservatoryMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"
