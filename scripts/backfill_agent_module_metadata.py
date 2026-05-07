"""Backfill the `module` and `screen` fields on every agent JSON across the 5 ADKs.

Why this exists:
    Since the SDK now applies the path-based defaults at registration time
    (see ``DEFAULT_MODULE_SCREEN_RULES`` in ``rapid_debug_engine.registry``),
    runtime classification is automatic for any agent dropped into a known
    folder. This script is for the offline case where you want the metadata
    persisted into the JSON files themselves — useful for auditing, code
    review, or making the placement explicit in version control.

What this does:
    - Walks each ADK's ``agentconfig/agents/`` tree (recursively).
    - Resolves (service, category, sub-path) via the SDK's own rule table.
    - If an agent JSON is missing ``module`` or ``screen``, fills in the
      inferred values. Existing values are left alone (manual overrides win —
      idempotent).
    - Logs files outside the rule table so you can decide whether to add a rule.

Usage:
    python scripts/backfill_agent_module_metadata.py            # dry-run
    python scripts/backfill_agent_module_metadata.py --apply    # write changes

To support a new ADK or category, edit ``DEFAULT_MODULE_SCREEN_RULES`` in
``src/rapid_debug_engine/registry.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

# Make the SDK importable so we can share the single source of truth for rules.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from rapid_debug_engine.registry import resolve_default_module_screen  # noqa: E402

# Each ADK's agentconfig root, keyed by service name (matches what the SDK posts
# as `serviceName`). Update this if a new ADK joins the constellation.
ADK_ROOTS: dict[str, Path] = {
    "rapid-adk-architecture": Path(
        r"C:\Users\akani\Documents\GitHub\Rapid Architecture\rapid-adk-architecture\agentconfig\agents"
    ),
    "rapid-adk-ideation": Path(
        r"C:\Users\akani\Documents\GitHub\Rapid Ideation\rapid-adk-ideation\agentconfig\agents"
    ),
    "rapid-adk-requirements": Path(
        r"C:\Users\akani\Documents\GitHub\Rapid Requirements\rapid-adk-requirements\agentconfig\agents"
    ),
    "rapid-adk-spec": Path(
        r"C:\Users\akani\Documents\GitHub\Rapid Spec\rapid-adk-spec\agentconfig\agents"
    ),
    "rapid-adk-transformation": Path(
        r"C:\Users\akani\Documents\GitHub\Rapid Transformation\rapid-adk-transformation\agentconfig\agents"
    ),
}


def reorder_agent_keys(data: dict) -> OrderedDict:
    """Place `module`/`screen` right after the agent's identity fields so the
    JSON stays readable. Existing keys keep their order otherwise."""
    if "module" not in data and "screen" not in data:
        return OrderedDict(data)

    preferred_first = ["name", "role", "description", "module", "screen"]
    out: OrderedDict = OrderedDict()
    for k in preferred_first:
        if k in data:
            out[k] = data[k]
    for k, v in data.items():
        if k not in out:
            out[k] = v
    return out


def process_agent_file(
    path: Path,
    service: str,
    apply: bool,
) -> str:
    """Returns one of: 'updated', 'skipped-already-set', 'skipped-no-rule', 'error'."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print(f"  ERROR  {path}: {err}", file=sys.stderr)
        return "error"

    if not isinstance(raw, dict):
        print(f"  ERROR  {path}: top-level is not an object", file=sys.stderr)
        return "error"

    # Path inside agentconfig/agents/ → category/sub.../file.json
    rel = path.relative_to(ADK_ROOTS[service])
    parts = rel.parts
    if len(parts) < 2:
        return "skipped-no-rule"
    category = parts[0]
    sub_segments = parts[1:-1]  # subfolders only, drop the filename

    rule = resolve_default_module_screen(service, category, sub_segments)
    if rule is None:
        print(f"  SKIP   {path} (no rule for {service}/{category}/{'/'.join(sub_segments)})")
        return "skipped-no-rule"

    module, screen = rule
    has_module = isinstance(raw.get("module"), str) and raw["module"].strip()
    has_screen = isinstance(raw.get("screen"), str) and raw["screen"].strip()

    if has_module and has_screen:
        return "skipped-already-set"

    if not has_module:
        raw["module"] = module
    if not has_screen:
        raw["screen"] = screen

    if apply:
        ordered = reorder_agent_keys(raw)
        path.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return "updated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to disk. Without this flag, only reports what would change.",
    )
    parser.add_argument(
        "--service",
        choices=sorted(ADK_ROOTS),
        help="Only process one ADK (default: all).",
    )
    args = parser.parse_args()

    services = [args.service] if args.service else list(ADK_ROOTS)
    grand_total = {"updated": 0, "skipped-already-set": 0, "skipped-no-rule": 0, "error": 0}

    for service in services:
        root = ADK_ROOTS[service]
        if not root.exists():
            print(f"\n!! {service}: agentconfig path not found ({root}) — skipping")
            continue
        print(f"\n== {service} ({root})")
        files = sorted(root.rglob("*.json"))
        per_service = {"updated": 0, "skipped-already-set": 0, "skipped-no-rule": 0, "error": 0}
        for path in files:
            result = process_agent_file(path, service, apply=args.apply)
            per_service[result] += 1
            grand_total[result] += 1
        print(
            f"   updated={per_service['updated']} "
            f"already-set={per_service['skipped-already-set']} "
            f"no-rule={per_service['skipped-no-rule']} "
            f"errors={per_service['error']}"
        )

    print("\n== TOTAL")
    print(
        f"   updated={grand_total['updated']} "
        f"already-set={grand_total['skipped-already-set']} "
        f"no-rule={grand_total['skipped-no-rule']} "
        f"errors={grand_total['error']}"
    )
    if not args.apply:
        print("\n(dry-run — re-run with --apply to write changes)")
    return 0 if grand_total["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
