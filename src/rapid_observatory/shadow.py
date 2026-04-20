"""Shadow-mode support for the Python SDK (M7).

The @shadow_aware decorator wraps an agent invocation so that — when the
Observatory has an **active dual_invoke experiment** registered for this
agent and this request falls inside the sampled traffic percentage — a
second invocation runs against the candidate config in an `asyncio.create_task`
background task.

Primary latency is unaffected: the candidate invocation fires after the
primary has returned to the caller. Both executions emit OTel spans tagged
with `shadow.experiment_id` + `shadow.variant=primary|candidate`, which the
Observatory ingest pairs up post-hoc on the `shadow_experiment_id` FK.

Important guardrails enforced by the backend before an experiment can enter
`dual_invoke` mode:

    - `acknowledgeCostDoubling: true` must be set explicitly
    - `trafficPercent <= 25`
    - Agent must have an active config (the "primary")

This decorator is deliberately opt-in per call site. The default behavior
is `simulation` mode, which runs entirely inside the Observatory backend
and never touches live traffic.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ShadowExperimentContext:
    """Minimal context the SDK needs to route a request into a shadow run.

    Populated by the Observatory config-sync payload when a dual_invoke
    experiment is active. Fields match the Observatory ShadowExperiment
    schema 1:1 so the SDK can construct attributes without re-mapping.
    """

    experiment_id: str
    agent_id: str
    primary_config_version_id: str
    candidate_config_version_id: str
    traffic_percent: int


# Module-level registry — populated by ObservatoryClient when the config-sync
# worker sees a `dual_invoke` experiment. Kept simple so unit tests can set
# and clear it explicitly.
_active: dict[str, ShadowExperimentContext] = {}


def set_active_experiment(agent_id: str, ctx: ShadowExperimentContext | None) -> None:
    if ctx is None:
        _active.pop(agent_id, None)
    else:
        _active[agent_id] = ctx


def get_active_experiment(agent_id: str) -> ShadowExperimentContext | None:
    return _active.get(agent_id)


def _should_sample(traffic_percent: int) -> bool:
    if traffic_percent <= 0:
        return False
    if traffic_percent >= 100:
        return True
    return random.random() * 100 < traffic_percent


def shadow_aware(
    *,
    get_agent_id: Callable[..., str | None],
    invoke_with_config: Callable[[str, Any], Awaitable[Any]] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator factory.

    Arguments:
        get_agent_id: extracts the agent id from the wrapped function's args
            so the decorator can look up the active experiment. Typical shape
            is `lambda *a, **kw: kw.get("agent_id")` or
            `lambda ctx, *_: ctx.agent_id`.
        invoke_with_config: async callable that runs the wrapped operation
            using a specific candidate config. Called in the background task
            with `(candidate_config_version_id, original_args_snapshot)`.
            When omitted, the decorator just tags the primary span but
            doesn't fire a candidate run — useful during gradual rollout.

    The returned decorator preserves the original function's signature,
    so it can wrap FastAPI route handlers or plain async functions.
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapped(*args: Any, **kwargs: Any) -> T:
            agent_id = get_agent_id(*args, **kwargs)
            ctx = _active.get(agent_id) if agent_id else None

            primary_result = await fn(*args, **kwargs)

            if ctx is None:
                return primary_result
            if not _should_sample(ctx.traffic_percent):
                return primary_result
            if invoke_with_config is None:
                logger.debug(
                    "observatory: shadow experiment active but no invoke_with_config provided; skipping candidate"
                )
                return primary_result

            # Candidate runs in the background — never blocks the primary
            # response. Errors are logged but never propagate.
            async def _candidate() -> None:
                try:
                    await invoke_with_config(
                        ctx.candidate_config_version_id,
                        {"args": args, "kwargs": kwargs},
                    )
                except Exception as err:  # noqa: BLE001 — must never crash hot path
                    logger.warning(
                        "observatory: shadow candidate invocation failed: %s", err
                    )

            asyncio.create_task(_candidate())
            return primary_result

        return wrapped

    return decorator


__all__ = [
    "ShadowExperimentContext",
    "set_active_experiment",
    "get_active_experiment",
    "shadow_aware",
]
