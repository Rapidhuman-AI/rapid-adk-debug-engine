"""Tests for DebugEngineClient.register_on_startup() using pytest-httpx."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from rapid_debug_engine import DebugEngineClient


def _seed_agentconfig(root: Path) -> Path:
    agents_dir = root / "agentconfig" / "agents"
    (agents_dir / "requirements").mkdir(parents=True)
    (agents_dir / "requirements" / "apis.json").write_text(
        json.dumps(
            {
                "name": "apis",
                "role": "AssistantAgent",
                "system_message": "APIs agent",
                "outputstructure": "{apis: []}",
            }
        ),
        encoding="utf-8",
    )
    return agents_dir


@pytest.mark.asyncio
async def test_register_on_startup_posts_and_caches_ids(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    agents_dir = _seed_agentconfig(tmp_path)

    httpx_mock.add_response(
        method="POST",
        url="http://debug-engine/api/v1/agents/register",
        json={
            "deploymentId": "acme",
            "agents": [
                {
                    "id": "agt_123",
                    "deploymentId": "acme",
                    "serviceName": "rapid-adk-requirements",
                    "category": "requirements",
                    "name": "apis",
                    "activeConfigId": "cfg_1",
                    "moduleId": None,
                    "screenId": None,
                    "createdAt": "2026-04-10T00:00:00Z",
                    "updatedAt": "2026-04-10T00:00:00Z",
                }
            ],
        },
    )

    client = DebugEngineClient(
        base_url="http://debug-engine",
        api_key="key",
        deployment_id="acme",
        service_name="rapid-adk-requirements",
    )
    registered = await client.register_on_startup(agentconfig_dir=agents_dir)

    assert len(registered) == 1
    assert client.get_agent_id("requirements", "apis") == "agt_123"

    request = httpx_mock.get_request()
    assert request is not None
    body = json.loads(request.content)
    assert body["deploymentId"] == "acme"
    assert body["serviceName"] == "rapid-adk-requirements"
    assert len(body["agents"]) == 1
    assert body["agents"][0]["name"] == "apis"

    await client.stop()


@pytest.mark.asyncio
async def test_register_on_startup_survives_http_error(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    agents_dir = _seed_agentconfig(tmp_path)
    httpx_mock.add_response(
        method="POST",
        url="http://debug-engine/api/v1/agents/register",
        status_code=500,
        json={"error": "boom"},
    )

    client = DebugEngineClient(
        base_url="http://debug-engine",
        api_key="key",
        deployment_id="acme",
        service_name="rapid-adk-requirements",
    )
    # Must not raise — the agent service should still boot even if the
    # Debug Engine is unreachable.
    result = await client.register_on_startup(agentconfig_dir=agents_dir)
    assert result == []
    await client.stop()


@pytest.mark.asyncio
async def test_register_on_startup_noop_when_no_agents(tmp_path: Path) -> None:
    client = DebugEngineClient(
        base_url="http://debug-engine",
        api_key="key",
        deployment_id="acme",
        service_name="svc",
    )
    result = await client.register_on_startup(agentconfig_dir=tmp_path / "nothing")
    assert result == []
    await client.stop()
