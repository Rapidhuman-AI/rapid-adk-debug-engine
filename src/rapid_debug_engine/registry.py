"""Walk agent configuration files on disk and build the registration payload.

Mirrors the structure the Python agent services use in `agentconfig/agents/<category>/<name>.json`:

    {
        "name": "apis",
        "role": "AssistantAgent",
        "system_message": "...",
        "outputstructure": "...",
        "module": "Filing Management",
        "screen": "Document Upload"
    }

Resolution order for an agent's (module, screen) placement, highest priority first:

    1. Per-agent JSON fields (`module`, `screen`) — explicit authorial intent.
    2. `agentconfig/modules.json` — central mapping file, kept for backward compat.
    3. ``DEFAULT_MODULE_SCREEN_RULES`` — path-based defaults keyed on
       (service_name, category, sub_path). Lets a new agent dropped into a
       known folder auto-classify with zero metadata.
    4. None — agent registers without a module link and surfaces under
       "Unmapped" in the Debug Engine UI.

To support a brand-new ADK or category, add a one-line entry to
``DEFAULT_MODULE_SCREEN_RULES`` below.
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


# Path-based defaults for agents that don't ship explicit `module`/`screen`
# metadata. Keyed on (service_name, category, sub_path) — the script picks the
# longest matching sub_path so more specific subfolders override the parent
# (e.g. `requirements/goldenthread` overrides `requirements`).
#
# When a new ADK or category appears, add a row here so its agents auto-classify
# without per-file edits. Per-agent JSON fields and `modules.json` still take
# precedence — these are pure fallback defaults.
DEFAULT_MODULE_SCREEN_RULES: dict[tuple[str, str, str], tuple[str, str]] = {
    # rapid-adk-architecture
    ("rapid-adk-architecture", "architecture", ""): ("Architecture Map", "Map Editor"),
    ("rapid-adk-architecture", "architecture", "rag"): ("Architecture Map", "RAG"),
    # rapid-adk-ideation
    ("rapid-adk-ideation", "ideation", ""): ("Ideation", "Brainstorm Board"),
    ("rapid-adk-ideation", "interview-analysis", ""): ("Ideation", "Interview Analysis"),
    ("rapid-adk-ideation", "wireframes", ""): ("Ideation", "Wireframes"),
    ("rapid-adk-ideation", "utility", ""): ("Ideation", "Utility"),
    # rapid-adk-requirements
    ("rapid-adk-requirements", "cipc", ""): ("Filing Management", "CIPC Filings"),
    ("rapid-adk-requirements", "process", ""): ("Filing Management", "Process"),
    ("rapid-adk-requirements", "requirements", ""): ("Filing Management", "Requirements"),
    ("rapid-adk-requirements", "requirements", "advanced_requirements"): (
        "Filing Management",
        "Advanced Requirements",
    ),
    ("rapid-adk-requirements", "requirements", "goldenthread"): (
        "Filing Management",
        "Golden Thread",
    ),
    ("rapid-adk-requirements", "requirements", "multi"): (
        "Filing Management",
        "Use Cases",
    ),
    ("rapid-adk-requirements", "utility", ""): ("Filing Management", "Utility"),
    # rapid-adk-spec
    ("rapid-adk-spec", "spec", ""): ("Specification Authoring", "Composer"),
    ("rapid-adk-spec", "requirements", ""): ("Specification Authoring", "Requirements"),
    ("rapid-adk-spec", "requirements", "advanced_requirements"): (
        "Specification Authoring",
        "Advanced Requirements",
    ),
    ("rapid-adk-spec", "requirements", "goldenthread"): (
        "Specification Authoring",
        "Golden Thread",
    ),
    ("rapid-adk-spec", "requirements", "multi"): (
        "Specification Authoring",
        "Use Cases",
    ),
    ("rapid-adk-spec", "utility", ""): ("Specification Authoring", "Utility"),
    # rapid-adk-transformation
    ("rapid-adk-transformation", "chat", ""): ("Chat Copilot", "Workspace Chat"),
    ("rapid-adk-transformation", "systems_overview", ""): (
        "Process Discovery",
        "Systems Overview",
    ),
    ("rapid-adk-transformation", "process_diagram", ""): (
        "Process Discovery",
        "Process Diagram",
    ),
    ("rapid-adk-transformation", "subprocess_breakdown", ""): (
        "Process Discovery",
        "Subprocess Breakdown",
    ),
    ("rapid-adk-transformation", "reimagined_process", ""): (
        "Process Discovery",
        "Reimagined Process",
    ),
    ("rapid-adk-transformation", "implementation_plan", ""): (
        "Process Discovery",
        "Implementation Plan",
    ),
    ("rapid-adk-transformation", "architecture", ""): ("Process Discovery", "Architecture"),
    ("rapid-adk-transformation", "rag", ""): ("Process Discovery", "RAG"),
}


def resolve_default_module_screen(
    service_name: str | None,
    category: str,
    sub_segments: tuple[str, ...],
) -> tuple[str, str] | None:
    """Pick the most specific path-rule match for this agent.

    Returns ``None`` if no rule matches or ``service_name`` is not provided.
    Per-agent JSON fields and modules.json still take precedence over this
    fallback in :func:`discover_agents`.
    """
    if not service_name:
        return None
    for depth in range(len(sub_segments), -1, -1):
        prefix = "/".join(sub_segments[:depth])
        rule = DEFAULT_MODULE_SCREEN_RULES.get((service_name, category, prefix))
        if rule:
            return rule
    return None


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
    service_name: str | None = None,
) -> list[AgentRegistration]:
    """Walk agentconfig/agents/<category>/.../<name>.json and build the registration list.

    ``service_name`` enables the path-based default rules in
    :data:`DEFAULT_MODULE_SCREEN_RULES`. When omitted, only per-agent JSON
    fields and ``modules.json`` provide module/screen placement.
    """
    if not agentconfig_dir.exists():
        logger.warning("debug_engine: agentconfig directory not found: %s", agentconfig_dir)
        return []

    modules = ModulesFile.load(modules_file) if modules_file else ModulesFile()
    registrations: list[AgentRegistration] = []

    # Recursively walk so nested folders (e.g. requirements/goldenthread/foo.json)
    # are discovered. The first path segment after agentconfig_dir is the category.
    for json_file in sorted(agentconfig_dir.rglob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            logger.warning("debug_engine: skipping %s: %s", json_file, err)
            continue
        if not isinstance(raw, dict):
            logger.warning("debug_engine: skipping %s: top-level is not an object", json_file)
            continue

        rel_parts = json_file.relative_to(agentconfig_dir).parts
        if len(rel_parts) < 2:
            # Stray JSON directly under agentconfig/agents/ — has no category.
            continue
        category = rel_parts[0]
        sub_segments = rel_parts[1:-1]

        name = raw.get("name", json_file.stem)
        config = _extract_config(raw)
        config_hash = _hash_config(config)

        # Resolution order: per-agent JSON → modules.json → path-rule defaults.
        module_name = raw.get("module") if isinstance(raw.get("module"), str) else None
        screen_name = raw.get("screen") if isinstance(raw.get("screen"), str) else None

        if module_name is None and screen_name is None:
            module_screen = modules.agent_to_screen.get((category, name))
            if module_screen:
                module_name, screen_name = module_screen

        if module_name is None and screen_name is None:
            default = resolve_default_module_screen(service_name, category, sub_segments)
            if default:
                module_name, screen_name = default

        registrations.append(
            AgentRegistration(
                category=category,
                name=name,
                description=raw.get("description"),
                config_path=str(json_file.relative_to(agentconfig_dir.parent)),
                config_hash=config_hash,
                config=config,
                module_name=module_name,
                screen_name=screen_name,
            )
        )

    logger.info("debug_engine: discovered %d agents in %s", len(registrations), agentconfig_dir)
    return registrations
