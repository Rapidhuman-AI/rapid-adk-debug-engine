"""Walk agent configuration files on disk and build the registration payload.

Mirrors the structure the Python agent services use in `agentconfig/agents/<category>/<name>.json`:

    {
        "name": "apis",
        "role": "AssistantAgent",
        "system_message": "...",
        "outputstructure": "..."
    }

And the optional `agentconfig/modules.json` file which declares the
Module → Screen → Agent hierarchy. When present, its mapping is authoritative
(unmapped agents fall back to UI authoring in the Debug Engine).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048


@dataclass
class AgentRegistration:
    """One agent discovered on disk, ready to POST to the Debug Engine."""

    category: str
    name: str
    description: str | None
    config_path: str
    config_hash: str
    config: dict[str, Any]
    module_name: str | None = None
    screen_name: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category,
            "name": self.name,
            "configPath": self.config_path,
            "configHash": self.config_hash,
            "config": self.config,
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.module_name is not None:
            payload["moduleName"] = self.module_name
        if self.screen_name is not None:
            payload["screenName"] = self.screen_name
        return payload


@dataclass
class ModulesFile:
    """Parsed `agentconfig/modules.json`. Authoritative when present."""

    version: int = 1
    # (category, name) → (module_name, screen_name)
    agent_to_screen: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ModulesFile":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = int(raw.get("version", 1))
        mapping: dict[tuple[str, str], tuple[str, str]] = {}
        for module in raw.get("modules", []):
            mod_name = module["name"]
            for screen in module.get("screens", []):
                screen_name = screen["name"]
                for agent in screen.get("agents", []):
                    key = (agent["category"], agent["name"])
                    mapping[key] = (mod_name, screen_name)
        return cls(version=version, agent_to_screen=mapping)


def _hash_config(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _extract_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the agent JSON into the Debug Engine AgentConfig shape.

    The Python services' JSON shape predates the Debug Engine, so we map
    loosely and fill in the rest with defaults. Whatever is stored becomes
    the v1 config for this agent in the Debug Engine.
    """
    prompt = raw.get("system_message", "")
    output_structure = raw.get("outputstructure")
    if output_structure:
        prompt = f"{prompt}\n\nOutput structure: {output_structure}"
    return {
        "model": raw.get("model", DEFAULT_MODEL),
        "temperature": float(raw.get("temperature", DEFAULT_TEMPERATURE)),
        "maxTokens": int(raw.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "prompt": prompt,
        "tools": list(raw.get("tools", [])),
        "meta": {
            "role": raw.get("role"),
            "outputStructure": output_structure,
        },
    }


def discover_agents(
    agentconfig_dir: Path,
    modules_file: Path | None = None,
) -> list[AgentRegistration]:
    """Walk agentconfig/agents/<category>/<name>.json and build the registration list."""
    if not agentconfig_dir.exists():
        logger.warning("debug_engine: agentconfig directory not found: %s", agentconfig_dir)
        return []

    modules = ModulesFile.load(modules_file) if modules_file else ModulesFile()
    registrations: list[AgentRegistration] = []

    for category_dir in sorted(p for p in agentconfig_dir.iterdir() if p.is_dir()):
        category = category_dir.name
        for json_file in sorted(category_dir.glob("*.json")):
            try:
                raw = json.loads(json_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as err:
                logger.warning("debug_engine: skipping %s: %s", json_file, err)
                continue

            name = raw.get("name", json_file.stem)
            config = _extract_config(raw)
            config_hash = _hash_config(config)
            module_screen = modules.agent_to_screen.get((category, name))

            registrations.append(
                AgentRegistration(
                    category=category,
                    name=name,
                    description=raw.get("description"),
                    config_path=str(json_file.relative_to(agentconfig_dir.parent)),
                    config_hash=config_hash,
                    config=config,
                    module_name=module_screen[0] if module_screen else None,
                    screen_name=module_screen[1] if module_screen else None,
                )
            )

    logger.info("debug_engine: discovered %d agents in %s", len(registrations), agentconfig_dir)
    return registrations
