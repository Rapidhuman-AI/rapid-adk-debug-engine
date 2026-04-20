# rapid-debug-engine-sdk

Python SDK installed into `rapid-adk-requirements` and `rapid-adk-transformation` so
the Rapid Debug Engine can monitor Google ADK agents running in those services.

## What it does

1. **Agent registration at startup** — Walks `agentconfig/agents/**/*.json` and optional
   `agentconfig/modules.json`, POSTs to `/api/v1/agents/register`. Idempotent.
2. **Config hot-reload** *(M4)* — Pub/Sub subscriber swaps configs in memory in <5s.
3. **`@shadow_aware`** *(M7)* — Dual-invokes candidate configs per shadow experiment.
4. **`enrich_span`** *(M3)* — Adds `rapid.module_id`, `rapid.screen_id`, `rapid.agent_id`
   attributes to the current OpenTelemetry span so the Debug Engine mapper can stitch
   traces to the agent registry.
5. **Guardrail event publisher** *(M8)* — Ships Pydantic validation outcomes to the
   Debug Engine so they surface in the Guardrails screens.

## Minimal integration (M1 scope only)

```python
# main.py lifespan
from contextlib import asynccontextmanager
from fastapi import FastAPI
from rapid_debug_engine import DebugEngineClient

debug_engine = DebugEngineClient(
    base_url="http://localhost:8080",
    api_key="dev-debug-engine-key",
    deployment_id="acme",
    service_name="rapid-adk-requirements",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await debug_engine.register_on_startup()
    yield
    await debug_engine.stop()

app = FastAPI(lifespan=lifespan)
```

## Local install

```bash
pip install -e ../rapid-debug-engine/observability-sdk-python
```
