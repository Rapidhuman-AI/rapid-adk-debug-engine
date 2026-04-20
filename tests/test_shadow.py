"""Tests for the @shadow_aware decorator."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rapid_debug_engine.shadow import (
    ShadowExperimentContext,
    set_active_experiment,
    shadow_aware,
)


@pytest.mark.asyncio
async def test_shadow_aware_returns_primary_when_no_experiment_active() -> None:
    calls: list[str] = []

    @shadow_aware(get_agent_id=lambda agent_id, **_: agent_id)
    async def primary(agent_id: str, payload: str) -> str:
        calls.append(f"primary:{payload}")
        return f"ok:{payload}"

    result = await primary("agt_1", payload="hello")
    assert result == "ok:hello"
    assert calls == ["primary:hello"]


@pytest.mark.asyncio
async def test_shadow_aware_fires_candidate_when_experiment_active() -> None:
    set_active_experiment(
        "agt_1",
        ShadowExperimentContext(
            experiment_id="exp_1",
            agent_id="agt_1",
            primary_config_version_id="cfg_v1",
            candidate_config_version_id="cfg_v2",
            traffic_percent=100,
        ),
    )
    candidate_args: list[Any] = []

    async def candidate(config_version_id: str, snapshot: dict[str, Any]) -> None:
        candidate_args.append((config_version_id, snapshot))

    @shadow_aware(
        get_agent_id=lambda agent_id, **_: agent_id,
        invoke_with_config=candidate,
    )
    async def primary(agent_id: str, payload: str) -> str:
        return f"primary:{payload}"

    try:
        result = await primary("agt_1", payload="x")
        assert result == "primary:x"
        # Candidate runs in the background — yield to let it complete.
        await asyncio.sleep(0.01)
        assert len(candidate_args) == 1
        config_version_id, snapshot = candidate_args[0]
        assert config_version_id == "cfg_v2"
        assert snapshot["kwargs"]["payload"] == "x"
    finally:
        set_active_experiment("agt_1", None)


@pytest.mark.asyncio
async def test_shadow_aware_skips_candidate_when_sampling_excludes_request() -> None:
    set_active_experiment(
        "agt_1",
        ShadowExperimentContext(
            experiment_id="exp_1",
            agent_id="agt_1",
            primary_config_version_id="cfg_v1",
            candidate_config_version_id="cfg_v2",
            traffic_percent=0,  # 0% traffic → candidate must never fire
        ),
    )
    fires: list[str] = []

    async def candidate(config_version_id: str, _: Any) -> None:
        fires.append(config_version_id)

    @shadow_aware(
        get_agent_id=lambda agent_id, **_: agent_id,
        invoke_with_config=candidate,
    )
    async def primary(agent_id: str) -> str:
        return "p"

    try:
        for _ in range(5):
            await primary("agt_1")
        await asyncio.sleep(0.01)
        assert fires == []
    finally:
        set_active_experiment("agt_1", None)


@pytest.mark.asyncio
async def test_shadow_aware_swallows_candidate_errors() -> None:
    set_active_experiment(
        "agt_1",
        ShadowExperimentContext(
            experiment_id="exp_1",
            agent_id="agt_1",
            primary_config_version_id="cfg_v1",
            candidate_config_version_id="cfg_v2",
            traffic_percent=100,
        ),
    )

    async def candidate(_: str, __: Any) -> None:
        raise RuntimeError("boom")

    @shadow_aware(
        get_agent_id=lambda agent_id, **_: agent_id,
        invoke_with_config=candidate,
    )
    async def primary(agent_id: str) -> str:
        return "p"

    try:
        result = await primary("agt_1")
        assert result == "p"  # primary unaffected
        await asyncio.sleep(0.01)
    finally:
        set_active_experiment("agt_1", None)
