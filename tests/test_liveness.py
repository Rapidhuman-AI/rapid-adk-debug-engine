"""Tests for liveness router + agent introspection.

These exercise the duck-typed runtime introspection (no `google.adk`
import — fake classes named `LlmAgent`/`SequentialAgent`/etc. are enough
because the SDK identifies orchestrators by class name) and the FastAPI
router that wraps them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_debug_engine import DebugEngineClient, introspect_agent_tree
from rapid_debug_engine.liveness import (
    AgentDefinition,
    _registration_to_definition,
    build_liveness_router,
)
from rapid_debug_engine.registry import AgentRegistration


# ─── Fake Google ADK agent shapes (duck-typed, no real ADK install) ────────


class LlmAgent:  # noqa: D101 — test fixture
    def __init__(
        self,
        name: str,
        model: str = "gemini-2.5-flash",
        instruction: str = "You are helpful.",
        tools: list[Any] | None = None,
        output_schema: Any | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.instruction = instruction
        self.tools = tools or []
        self.output_schema = output_schema


class SequentialAgent:  # noqa: D101
    def __init__(self, name: str, sub_agents: list[Any]) -> None:
        self.name = name
        self.sub_agents = sub_agents


class ParallelAgent:  # noqa: D101
    def __init__(self, name: str, sub_agents: list[Any]) -> None:
        self.name = name
        self.sub_agents = sub_agents


class LoopAgent:  # noqa: D101
    def __init__(self, name: str, sub_agents: list[Any], max_iterations: int = 3) -> None:
        self.name = name
        self.sub_agents = sub_agents
        self.max_iterations = max_iterations


class _NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


# ─── Introspection tests ────────────────────────────────────────────────────


def test_single_llm_agent_is_kind_single() -> None:
    agent = LlmAgent(name="apis", instruction="Summarise APIs.")

    [definition] = introspect_agent_tree(agent)

    assert definition.name == "apis"
    assert definition.kind == "single"
    assert definition.model == "gemini-2.5-flash"
    assert definition.system_message == "Summarise APIs."
    assert definition.sub_agents == []
    assert definition.orchestrator is None


def test_tools_are_extracted_by_name_and_callable() -> None:
    def search(query: str) -> str:  # callable tool
        return query

    tool_obj = _NamedTool("calculator")
    agent = LlmAgent(name="root", tools=[tool_obj, search, "raw_string_tool"])

    [definition] = introspect_agent_tree(agent)

    assert definition.tools == ["calculator", "search", "raw_string_tool"]


def test_sequential_agent_with_subagents_is_multi() -> None:
    child_a = LlmAgent(name="extract", instruction="Extract entities.")
    child_b = LlmAgent(name="enrich", instruction="Enrich entities.")
    root = SequentialAgent(name="entity_pipeline", sub_agents=[child_a, child_b])

    [definition] = introspect_agent_tree(root)

    assert definition.kind == "multi"
    assert definition.orchestrator == "SequentialAgent"
    assert definition.name == "entity_pipeline"
    assert len(definition.sub_agents) == 2
    assert {s.name for s in definition.sub_agents} == {"extract", "enrich"}
    assert all(s.kind == "single" for s in definition.sub_agents)


def test_parallel_agent_recurses_into_nested_orchestrators() -> None:
    leaf_a = LlmAgent(name="a")
    leaf_b = LlmAgent(name="b")
    inner = ParallelAgent(name="fanout", sub_agents=[leaf_a, leaf_b])
    leaf_c = LlmAgent(name="c")
    outer = SequentialAgent(name="pipeline", sub_agents=[inner, leaf_c])

    [definition] = introspect_agent_tree(outer)

    assert definition.kind == "multi"
    assert definition.orchestrator == "SequentialAgent"
    assert len(definition.sub_agents) == 2
    nested_parallel = next(s for s in definition.sub_agents if s.name == "fanout")
    assert nested_parallel.kind == "multi"
    assert nested_parallel.orchestrator == "ParallelAgent"
    assert {s.name for s in nested_parallel.sub_agents} == {"a", "b"}


def test_loop_agent_classified_as_multi() -> None:
    leaf = LlmAgent(name="iterate")
    root = LoopAgent(name="retry_loop", sub_agents=[leaf])

    [definition] = introspect_agent_tree(root)

    assert definition.kind == "multi"
    assert definition.orchestrator == "LoopAgent"
    assert len(definition.sub_agents) == 1


def test_introspect_handles_iterable_root() -> None:
    a = LlmAgent(name="one")
    b = LlmAgent(name="two")

    definitions = introspect_agent_tree([a, b])

    assert {d.name for d in definitions} == {"one", "two"}


def test_introspect_returns_empty_for_none() -> None:
    assert introspect_agent_tree(None) == []


def test_pydantic_output_schema_is_json_serialised() -> None:
    class FakeSchema:
        @staticmethod
        def model_json_schema() -> dict[str, Any]:
            return {"type": "object", "properties": {"answer": {"type": "string"}}}

    agent = LlmAgent(name="schema_agent", output_schema=FakeSchema)

    [definition] = introspect_agent_tree(agent)

    assert definition.output_structure == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }


# ─── Static-config fallback tests ───────────────────────────────────────────


def test_registration_to_definition_strips_appended_output_structure() -> None:
    config = {
        "model": "gemini-2.5-flash",
        "prompt": "You are helpful.\n\nOutput structure: {result: 'string'}",
        "tools": ["search", "calculator"],
        "meta": {"role": "AssistantAgent", "outputStructure": "{result: 'string'}"},
    }
    reg = AgentRegistration(
        category="requirements",
        name="apis",
        description="Summarise APIs",
        config_path="agentconfig/agents/requirements/apis.json",
        config_hash="abc",
        config=config,
    )

    definition = _registration_to_definition(reg)

    assert definition.kind == "single"
    assert definition.system_message == "You are helpful."
    assert definition.tools == ["search", "calculator"]
    assert definition.output_structure == "{result: 'string'}"


# ─── FastAPI router tests ───────────────────────────────────────────────────


def _seed_static_agentconfig(root: Path) -> tuple[Path, Path]:
    agents_dir = root / "agentconfig" / "agents"
    (agents_dir / "requirements").mkdir(parents=True)
    (agents_dir / "requirements" / "apis.json").write_text(
        json.dumps(
            {
                "name": "apis",
                "role": "AssistantAgent",
                "system_message": "Summarise APIs",
                "outputstructure": "{apis: []}",
                "tools": ["http_get"],
            }
        ),
        encoding="utf-8",
    )
    return agents_dir, root / "agentconfig" / "modules.json"


def _build_app(
    *,
    service_name: str,
    agentconfig_dir: Path,
    modules_file: Path,
    agent_tree_provider: Any | None = None,
) -> FastAPI:
    app = FastAPI()
    router = build_liveness_router(
        service_name=service_name,
        sdk_version="0.1.0-test",
        agent_tree_provider=agent_tree_provider,
        agentconfig_dir=agentconfig_dir,
        modules_file=modules_file,
    )
    app.include_router(router)
    return app


def test_health_endpoint_returns_ok_and_service(tmp_path: Path) -> None:
    agents_dir, modules_file = _seed_static_agentconfig(tmp_path)
    app = _build_app(
        service_name="rapid-adk-test",
        agentconfig_dir=agents_dir,
        modules_file=modules_file,
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "rapid-adk-test"
    assert body["version"] == "0.1.0-test"
    assert body["uptime_s"] >= 0


def test_agents_list_with_runtime_provider(tmp_path: Path) -> None:
    agents_dir, modules_file = _seed_static_agentconfig(tmp_path)
    root_agent = SequentialAgent(
        name="pipeline",
        sub_agents=[LlmAgent(name="extract"), LlmAgent(name="enrich")],
    )
    app = _build_app(
        service_name="rapid-adk-test",
        agentconfig_dir=agents_dir,
        modules_file=modules_file,
        agent_tree_provider=lambda: root_agent,
    )
    client = TestClient(app)

    response = client.get("/agents/list")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "rapid-adk-test"
    assert len(body["agents"]) == 1
    assert body["agents"][0]["kind"] == "multi"
    assert body["agents"][0]["orchestrator"] == "SequentialAgent"
    assert {s["name"] for s in body["agents"][0]["sub_agents"]} == {"extract", "enrich"}


def test_agents_list_falls_back_to_static_when_no_provider(tmp_path: Path) -> None:
    agents_dir, modules_file = _seed_static_agentconfig(tmp_path)
    app = _build_app(
        service_name="rapid-adk-test",
        agentconfig_dir=agents_dir,
        modules_file=modules_file,
    )
    client = TestClient(app)

    response = client.get("/agents/list")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "rapid-adk-test"
    [agent] = body["agents"]
    assert agent["name"] == "apis"
    assert agent["kind"] == "single"
    assert agent["system_message"] == "Summarise APIs"
    assert agent["tools"] == ["http_get"]
    assert agent["output_structure"] == "{apis: []}"


def test_agents_list_falls_back_when_provider_raises(tmp_path: Path) -> None:
    agents_dir, modules_file = _seed_static_agentconfig(tmp_path)

    def boom() -> Any:
        raise RuntimeError("agent not constructed yet")

    app = _build_app(
        service_name="rapid-adk-test",
        agentconfig_dir=agents_dir,
        modules_file=modules_file,
        agent_tree_provider=boom,
    )
    client = TestClient(app)

    response = client.get("/agents/list")

    assert response.status_code == 200
    body = response.json()
    # static fallback kicked in
    assert [a["name"] for a in body["agents"]] == ["apis"]


# ─── DebugEngineClient.register_liveness_routes integration ─────────────────


@pytest.mark.asyncio
async def test_client_register_liveness_routes_mounts_endpoints(tmp_path: Path) -> None:
    agents_dir, modules_file = _seed_static_agentconfig(tmp_path)
    debug_engine = DebugEngineClient(
        base_url="http://debug-engine",
        api_key="key",
        deployment_id="acme",
        service_name="rapid-adk-test",
    )
    app = FastAPI()
    debug_engine.register_liveness_routes(
        app,
        agentconfig_dir=agents_dir,
        modules_file=modules_file,
    )

    client = TestClient(app)
    health = client.get("/health").json()
    agents = client.get("/agents/list").json()

    assert health["service"] == "rapid-adk-test"
    assert health["status"] == "ok"
    assert agents["service"] == "rapid-adk-test"
    assert [a["name"] for a in agents["agents"]] == ["apis"]

    await debug_engine.stop()


def test_agent_definition_to_dict_is_json_serialisable() -> None:
    definition = AgentDefinition(
        category="requirements",
        name="apis",
        kind="single",
        model="gemini-2.5-flash",
        system_message="hello",
        tools=["search"],
    )
    payload = definition.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["kind"] == "single"
