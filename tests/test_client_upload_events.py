"""Tests for DebugEngineClient.record_upload_event() — Phase 1/2 metadata sink."""

from __future__ import annotations

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from rapid_debug_engine import DebugEngineClient


def _make_client() -> DebugEngineClient:
    return DebugEngineClient(
        base_url="http://debug-engine",
        api_key="key",
        deployment_id="acme",
        service_name="rapid-adk-transformation",
    )


@pytest.mark.asyncio
async def test_record_upload_event_posts_full_payload(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="http://debug-engine/api/v1/ingest/upload-events",
        status_code=201,
        json={"id": "upl_1", "duplicate": False},
    )

    client = _make_client()
    try:
        await client.record_upload_event(
            tenant_id="adams_adams",
            project_id="p-1",
            solution="transformation",
            phase="phase_2",
            document_id="doc-1",
            sha256="a" * 64,
            mime_type="application/pdf",
            size_bytes=12345,
            extracted_by="process_diagram.orchestrator",
            extraction_version="1.0",
            lifecycle_state="active",
            idempotency_key="ik-1",
            confidence=0.91,
            trace_id="t" * 32,
        )
    finally:
        await client.stop()

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body == {
        "tenantId": "adams_adams",
        "projectId": "p-1",
        "solution": "transformation",
        "phase": "phase_2",
        "documentId": "doc-1",
        "sha256": "a" * 64,
        "mimeType": "application/pdf",
        "sizeBytes": 12345,
        "extractedBy": "process_diagram.orchestrator",
        "extractionVersion": "1.0",
        "lifecycleState": "active",
        "idempotencyKey": "ik-1",
        "confidence": 0.91,
        "traceId": "t" * 32,
    }


@pytest.mark.asyncio
async def test_record_upload_event_omits_optional_fields_when_none(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url="http://debug-engine/api/v1/ingest/upload-events",
        status_code=201,
        json={"id": "upl_2"},
    )

    client = _make_client()
    try:
        await client.record_upload_event(
            tenant_id="acme",
            project_id="p-2",
            solution="transformation",
            phase="phase_1",
            document_id="d",
            sha256="b" * 64,
            mime_type="text/markdown",
            size_bytes=100,
            extracted_by="source_analyzer",
            extraction_version="0.9.0",
            lifecycle_state="draft",
            idempotency_key="ik-2",
        )
    finally:
        await client.stop()

    body = json.loads(httpx_mock.get_requests()[0].content)
    assert "confidence" not in body
    assert "traceId" not in body


@pytest.mark.asyncio
async def test_record_upload_event_swallows_http_errors(httpx_mock: HTTPXMock) -> None:
    """Fire-and-forget: a server error must NOT propagate to the caller."""
    httpx_mock.add_response(
        method="POST",
        url="http://debug-engine/api/v1/ingest/upload-events",
        status_code=500,
        text="Internal Server Error",
    )

    client = _make_client()
    try:
        # Must not raise.
        result = await client.record_upload_event(
            tenant_id="acme",
            project_id="p",
            solution="transformation",
            phase="phase_2",
            document_id="d",
            sha256="c" * 64,
            mime_type="x",
            size_bytes=1,
            extracted_by="a",
            extraction_version="1",
            lifecycle_state="active",
            idempotency_key="ik",
        )
        assert result is None
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_record_upload_event_swallows_network_errors(httpx_mock: HTTPXMock) -> None:
    """Connection failure also must not propagate."""
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    client = _make_client()
    try:
        # Must not raise.
        await client.record_upload_event(
            tenant_id="acme",
            project_id="p",
            solution="transformation",
            phase="phase_2",
            document_id="d",
            sha256="d" * 64,
            mime_type="x",
            size_bytes=1,
            extracted_by="a",
            extraction_version="1",
            lifecycle_state="active",
            idempotency_key="ik-net",
        )
    finally:
        await client.stop()
