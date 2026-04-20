"""Tests for the long-poll ConfigSyncWorker + AgentConfigRegistry integration."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from rapid_observatory.config_sync import (
    AgentConfigRegistry,
    ConfigSyncWorker,
)


@pytest.mark.asyncio
async def test_sync_worker_applies_promotion_events_to_registry(httpx_mock: HTTPXMock) -> None:
    event: dict[str, Any] = {
        "agentId": "agt_abc",
        "category": "requirements",
        "name": "apis",
        "configVersionId": "cfg_v2",
        "version": 2,
        "config": {
            "model": "gemini-2.5-flash",
            "temperature": 0.1,
            "maxTokens": 2048,
            "prompt": "Updated prompt",
            "tools": [],
            "meta": {},
        },
        "promotedAt": "2026-04-10T10:00:00Z",
        "promotedBy": "alice@example.com",
    }
    httpx_mock.add_response(
        method="GET",
        url=httpx.URL("http://observatory/api/v1/configs/sync", params={}),
        json={"events": [event], "cursor": "2026-04-10T10:00:00Z"},
        match_headers={},
    )
    # Subsequent polls during the worker lifetime get no events.
    httpx_mock.add_response(
        method="GET",
        url=httpx.URL("http://observatory/api/v1/configs/sync", params={}),
        json={"events": [], "cursor": "2026-04-10T10:00:00Z"},
    )

    http = httpx.AsyncClient(base_url="http://observatory")
    registry = AgentConfigRegistry()
    worker = ConfigSyncWorker(
        http=http,
        deployment_id="acme",
        service_name="rapid-adk-requirements",
        registry=registry,
        poll_interval_s=0.05,
    )

    await worker.start()
    # Give the background task a couple of ticks to run.
    for _ in range(20):
        await asyncio.sleep(0.02)
        if registry.get("requirements", "apis") is not None:
            break
    await worker.stop()
    await http.aclose()

    override = registry.get("requirements", "apis")
    assert override is not None
    assert override.version == 2
    assert override.agent_id == "agt_abc"
    assert override.config["temperature"] == 0.1
    assert override.promoted_by == "alice@example.com"


@pytest.mark.asyncio
async def test_sync_worker_survives_http_errors(httpx_mock: HTTPXMock) -> None:
    """A flaky backend must not kill the long-poll loop."""
    # First call: 500. Second call onward: empty success.
    httpx_mock.add_response(
        method="GET",
        url=httpx.URL("http://observatory/api/v1/configs/sync", params={}),
        status_code=500,
    )
    httpx_mock.add_response(
        method="GET",
        url=httpx.URL("http://observatory/api/v1/configs/sync", params={}),
        json={"events": [], "cursor": "2026-04-10T10:00:00Z"},
    )

    http = httpx.AsyncClient(base_url="http://observatory")
    registry = AgentConfigRegistry()
    worker = ConfigSyncWorker(
        http=http,
        deployment_id="acme",
        service_name="svc",
        registry=registry,
        poll_interval_s=0.02,
    )
    await worker.start()
    # Let several ticks pass — first polls 500, then 200, then no more mocks.
    await asyncio.sleep(0.15)
    await worker.stop()
    await http.aclose()
    # Registry stayed empty; no override written on the error path.
    assert registry.get("requirements", "apis") is None


@pytest.mark.asyncio
async def test_registry_set_and_get() -> None:
    from rapid_observatory.config_sync import AgentConfigOverride

    registry = AgentConfigRegistry()
    assert registry.get("x", "y") is None

    override = AgentConfigOverride(
        agent_id="agt",
        category="x",
        name="y",
        config_version_id="v",
        version=3,
        config={"model": "m", "temperature": 0.5, "maxTokens": 100, "prompt": "p", "tools": [], "meta": {}},
        promoted_at="2026-04-10T00:00:00Z",
        promoted_by="dev",
    )
    registry.set(override)
    assert registry.get("x", "y") is override
