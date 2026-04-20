# rapid-observatory-sdk

Python SDK installed into `rapid-adk-requirements` and `rapid-adk-transformation` so
the AI Agent Observatory can monitor Google ADK agents running in those services.

## What it does

1. **Agent registration at startup** — Walks `agentconfig/agents/**/*.json` and optional
   `agentconfig/modules.json`, POSTs to `/api/v1/agents/register`. Idempotent.
2. **Config hot-reload** *(M4)* — Pub/Sub subscriber swaps configs in memory in <5s.
3. **`@shadow_aware`** *(M7)* — Dual-invokes candidate configs per shadow experiment.
4. **`enrich_span`** *(M3)* — Adds `rapid.module_id`, `rapid.screen_id`, `rapid.agent_id`
   attributes to the current OpenTelemetry span so the Observatory mapper can stitch
   traces to the agent registry.
5. **Guardrail event publisher** *(M8)* — Ships Pydantic validation outcomes to the
   Observatory so they surface in the Guardrails screens.

## Minimal integration (M1 scope only)

```python
# main.py lifespan
from contextlib import asynccontextmanager
from fastapi import FastAPI
from rapid_observatory import ObservatoryClient

observatory = ObservatoryClient(
    base_url="http://localhost:8080",
    api_key="dev-observatory-key",
    deployment_id="acme",
    service_name="rapid-adk-requirements",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await observatory.register_on_startup()
    yield
    await observatory.stop()

app = FastAPI(lifespan=lifespan)
```

## Local install

```bash
pip install -e ../rapid-agent-observatory/observability-sdk-python
```
