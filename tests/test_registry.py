"""Tests for the agent discovery + registration payload builder."""

from __future__ import annotations

import json
from pathlib import Path

from rapid_debug_engine.registry import (
    ModulesFile,
    discover_agents,
    resolve_default_module_screen,
)


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


def test_path_rule_classifies_agents_with_no_metadata(tmp_path: Path) -> None:
    """A new agent JSON without `module`/`screen` and without modules.json
    should still pick up the path-based default when service_name is supplied."""
    agents_dir = tmp_path / "agentconfig" / "agents"
    _write_agent(agents_dir / "ideation" / "newbot.json", "newbot")

    [reg] = discover_agents(agents_dir, service_name="rapid-adk-ideation")
    assert reg.module_name == "Ideation"
    assert reg.screen_name == "Brainstorm Board"


def test_path_rule_picks_most_specific_subfolder(tmp_path: Path) -> None:
    """When sub-paths nest (e.g. requirements/goldenthread/...), the deeper
    rule wins over the parent category rule."""
    agents_dir = tmp_path / "agentconfig" / "agents"
    _write_agent(agents_dir / "requirements" / "top.json", "top")
    _write_agent(agents_dir / "requirements" / "goldenthread" / "deep.json", "deep")

    by_name = {
        r.name: r
        for r in discover_agents(agents_dir, service_name="rapid-adk-requirements")
    }
    assert by_name["top"].screen_name == "Requirements"
    assert by_name["deep"].screen_name == "Golden Thread"
    assert by_name["top"].module_name == by_name["deep"].module_name == "Filing Management"


def test_path_rule_only_applies_when_service_name_provided(tmp_path: Path) -> None:
    """Without service_name, path-rule fallback is inactive (preserves the
    old behaviour for callers that don't know which service they're in)."""
    agents_dir = tmp_path / "agentconfig" / "agents"
    _write_agent(agents_dir / "ideation" / "newbot.json", "newbot")

    [reg] = discover_agents(agents_dir)  # no service_name
    assert reg.module_name is None
    assert reg.screen_name is None


def test_per_agent_metadata_overrides_path_rule(tmp_path: Path) -> None:
    """Per-agent JSON fields beat the path-rule default."""
    agents_dir = tmp_path / "agentconfig" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "ideation").mkdir()
    (agents_dir / "ideation" / "custom.json").write_text(
        json.dumps(
            {
                "name": "custom",
                "module": "Custom Module",
                "screen": "Custom Screen",
            }
        ),
        encoding="utf-8",
    )

    [reg] = discover_agents(agents_dir, service_name="rapid-adk-ideation")
    assert reg.module_name == "Custom Module"
    assert reg.screen_name == "Custom Screen"


def test_resolve_default_module_screen_returns_none_for_unknown_service() -> None:
    assert resolve_default_module_screen("never-heard-of-it", "ideation", ()) is None
    assert resolve_default_module_screen(None, "ideation", ()) is None


def test_per_agent_module_screen_metadata_is_authoritative(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agentconfig" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "architecture").mkdir()
    (agents_dir / "architecture" / "blueprint.json").write_text(
        json.dumps(
            {
                "name": "blueprint",
                "role": "AssistantAgent",
                "system_message": "draft architecture blueprint",
                "outputstructure": "{result: 'string'}",
                "module": "Architecture Map",
                "screen": "Map Editor",
            }
        ),
        encoding="utf-8",
    )

    registrations = discover_agents(agents_dir)
    assert len(registrations) == 1
    reg = registrations[0]
    assert reg.module_name == "Architecture Map"
    assert reg.screen_name == "Map Editor"


def test_per_agent_metadata_overrides_modules_json(tmp_path: Path) -> None:
    """When both per-agent fields and modules.json exist, the agent fields win."""
    agents_dir = tmp_path / "agentconfig" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "architecture").mkdir()
    (agents_dir / "architecture" / "blueprint.json").write_text(
        json.dumps(
            {
                "name": "blueprint",
                "role": "AssistantAgent",
                "system_message": "draft",
                "module": "Architecture Map",
                "screen": "Map Editor",
            }
        ),
        encoding="utf-8",
    )
    modules_file = tmp_path / "agentconfig" / "modules.json"
    modules_file.write_text(
        json.dumps(
            {
                "version": 1,
                "modules": [
                    {
                        "name": "Legacy Module",
                        "screens": [
                            {
                                "name": "Legacy Screen",
                                "agents": [{"category": "architecture", "name": "blueprint"}],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    [reg] = discover_agents(agents_dir, modules_file)
    assert reg.module_name == "Architecture Map"
    assert reg.screen_name == "Map Editor"


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
