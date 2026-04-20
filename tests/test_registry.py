"""Tests for the agent discovery + registration payload builder."""

from __future__ import annotations

import json
from pathlib import Path

from rapid_debug_engine.registry import ModulesFile, discover_agents


def _write_agent(path: Path, name: str, system_message: str = "default") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": name,
                "role": "AssistantAgent",
                "system_message": system_message,
                "outputstructure": "{result: 'string'}",
            }
        ),
        encoding="utf-8",
    )


def test_discover_agents_finds_all_json_files(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agentconfig" / "agents"
    _write_agent(agents_dir / "requirements" / "apis.json", "apis")
    _write_agent(agents_dir / "requirements" / "entities.json", "entities")
    _write_agent(agents_dir / "architecture" / "parse.json", "parse")

    registrations = discover_agents(agents_dir)

    assert len(registrations) == 3
    by_name = {(r.category, r.name) for r in registrations}
    assert ("requirements", "apis") in by_name
    assert ("requirements", "entities") in by_name
    assert ("architecture", "parse") in by_name


def test_config_hash_is_stable_and_changes_with_content(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agentconfig" / "agents"
    _write_agent(agents_dir / "requirements" / "apis.json", "apis", "v1")
    first = discover_agents(agents_dir)
    first_hash = first[0].config_hash

    second = discover_agents(agents_dir)
    assert second[0].config_hash == first_hash  # stable

    _write_agent(agents_dir / "requirements" / "apis.json", "apis", "v2")
    third = discover_agents(agents_dir)
    assert third[0].config_hash != first_hash  # sensitive


def test_modules_file_maps_agents_to_screens(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agentconfig" / "agents"
    _write_agent(agents_dir / "requirements" / "apis.json", "apis")
    _write_agent(agents_dir / "requirements" / "entities.json", "entities")

    modules_file = tmp_path / "agentconfig" / "modules.json"
    modules_file.write_text(
        json.dumps(
            {
                "version": 1,
                "modules": [
                    {
                        "name": "Filing Management",
                        "screens": [
                            {
                                "name": "Filing Intake",
                                "agents": [{"category": "requirements", "name": "apis"}],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    registrations = discover_agents(agents_dir, modules_file)
    by_name = {(r.category, r.name): r for r in registrations}

    assert by_name[("requirements", "apis")].module_name == "Filing Management"
    assert by_name[("requirements", "apis")].screen_name == "Filing Intake"
    # entities is unmapped — should stay None so the server renders it under "Unmapped"
    assert by_name[("requirements", "entities")].module_name is None
    assert by_name[("requirements", "entities")].screen_name is None


def test_discover_agents_returns_empty_when_directory_missing(tmp_path: Path) -> None:
    result = discover_agents(tmp_path / "missing")
    assert result == []


def test_modules_file_returns_empty_when_file_missing(tmp_path: Path) -> None:
    result = ModulesFile.load(tmp_path / "absent.json")
    assert result.agent_to_screen == {}


def test_registration_payload_shape(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agentconfig" / "agents"
    _write_agent(agents_dir / "requirements" / "apis.json", "apis")
    registrations = discover_agents(agents_dir)

    payload = registrations[0].to_payload()
    assert payload["category"] == "requirements"
    assert payload["name"] == "apis"
    assert "configHash" in payload
    assert "config" in payload
    assert "model" in payload["config"]
    assert "prompt" in payload["config"]
