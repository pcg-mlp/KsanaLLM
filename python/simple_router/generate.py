# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

import config
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from name_service.name_service import NameServiceProvider, NameServiceRegistry

from comm_id_allocator import allocate_comm_id

logger = logging.getLogger(__name__)

_NAME_SERVICE_PROVIDER: NameServiceProvider | None = None


def init_name_service() -> None:
    """Initialize the global name service provider. Called from lifespan."""
    global _NAME_SERVICE_PROVIDER
    _NAME_SERVICE_PROVIDER = NameServiceRegistry.get_provider_from_config(
        config.get_settings().name_service_provider
    )


def _get_provider() -> NameServiceProvider:
    if _NAME_SERVICE_PROVIDER is None:
        raise RuntimeError(
            "Name service provider not initialized. "
            "Ensure init_name_service() is called during startup."
        )
    return _NAME_SERVICE_PROVIDER


raw_router = APIRouter()

# Global HTTP client for connection pooling
_HTTP_CLIENT: httpx.AsyncClient | None = None

# Settings
_SETTINGS = config.get_settings()


def get_http_client() -> httpx.AsyncClient:
    """Get or create global HTTP client with connection pooling."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=None, limits=httpx.Limits(max_connections=65535, max_keepalive_connections=8192)
        )
    return _HTTP_CLIENT


async def close_http_client():
    """Close the global HTTP client (call on shutdown)."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        await _HTTP_CLIENT.aclose()
        _HTTP_CLIENT = None


# Constants for stream processing
DELIM = b"\0"  # Delimiter for separating chunks in the stream
EOS = b"[DONE]"  # End of stream marker


async def forward_request(req: Request, endpoint_path: str):
    """Route a client request to prefill/decode nodes and merge their streams."""

    # Get upstream nodes from name service provider
    try:
        prefill, decode, prefill_instance, decode_instance = _get_provider().get_available_nodes()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to select nodes: %s", exc)
        raise HTTPException(status_code=503, detail="No available nodes for processing") from exc
    prefill_name, prefill_address = prefill
    decode_name, decode_address = decode
    prefill_url = f"http://{prefill_address}{endpoint_path}"
    decode_url = f"http://{decode_address}{endpoint_path}"

    # Start timing
    router_start_ns = time.time_ns()
    client = get_http_client()

    # Allocate unique comm_id across uvicorn workers
    try:
        comm_id_start_time = time.time_ns()
        comm_id = allocate_comm_id()
        comm_id_ms = (time.time_ns() - comm_id_start_time) // 1_000_000
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to allocate comm_id: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to allocate comm_id") from exc

    # Use absolute timestamp for timing
    headers = {
        "kv-comm-group-key": f"{prefill_name}__{decode_name}",
        "kv-comm-request-id": str(comm_id),
        "kv-comm-start-ns": str(router_start_ns),
    }
    
    # Forward raw request body without parsing
    request_body = await req.body()
    
    # Prepare request kwargs using content parameter to forward raw bytes
    request_kwargs: Dict[str, Any] = {
        "headers": headers,
        "content": request_body,
    }
    
    # Copy original content type if present
    if "content-type" in req.headers:
        request_kwargs["headers"]["content-type"] = req.headers["content-type"]

    logger.info(
        "[ROUTER-1] comm_id=%s alloc_cost=%dms | Start routing | prefill=%s decode=%s",
        comm_id,
        comm_id_ms,
        prefill_url,
        decode_url,
    )
    try:
        prefill_ctx = client.stream(req.method, prefill_url, **request_kwargs)
        decode_ctx = client.stream(req.method, decode_url, **request_kwargs)
    except httpx.RequestError as exc:
        logger.error("HTTP stream setup failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to connect to nodes: {exc}") from exc

    # Get both responses concurrently and ensure connections are established.
    # CRITICAL: Both prefill and decode must receive the request at the same time
    # because prefill needs to transfer KV cache to decode via NCCL. If decode
    # hasn't received the request yet, the NCCL transfer deadlocks.
    try:
        prefill_resp, decode_resp = await asyncio.gather(
            prefill_ctx.__aenter__(),
            decode_ctx.__aenter__(),
        )
    except Exception:
        for ctx in [prefill_ctx, decode_ctx]:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:  # pylint: disable=broad-except
                pass
        raise
    elapsed_ms = (time.time_ns() - router_start_ns) // 1_000_000
    logger.info(
        "[ROUTER-2] comm_id=%s | Got response headers | elapsed=%dms | prefill_status=%d decode_status=%d",
        comm_id,
        elapsed_ms,
        prefill_resp.status_code,
        decode_resp.status_code,
    )
    # update node call results
    _get_provider().update_nodes_call_result(
        prefill_instance=prefill_instance,
        decode_instance=decode_instance,
        prefill_success=prefill_resp.status_code < 400,
        decode_success=decode_resp.status_code < 400,
    )

    # Check upstream response types
    is_streaming = prefill_resp.headers.get("transfer-encoding") == "chunked"

    # Helper to drain prefill stream in background (let it complete normally)
    async def _drain_prefill():
        try:
            async for _ in prefill_resp.aiter_bytes():
                pass
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Prefill stream error (ignored): %s", e)
        finally:
            await prefill_ctx.__aexit__(None, None, None)

    if not is_streaming:
        # Drain prefill in background, return decode response
        asyncio.create_task(_drain_prefill())
        content = await decode_resp.aread()
        await decode_ctx.__aexit__(None, None, None)
        elapsed_ms = (time.time_ns() - router_start_ns) // 1_000_000
        logger.info(
            "[ROUTER-3] comm_id=%s | Response complete (non-streaming) | elapsed=%dms | body_size=%d",
            comm_id,
            elapsed_ms,
            len(content),
        )
        return Response(
            content=content,
            status_code=decode_resp.status_code,
            headers=dict(decode_resp.headers),
        )

    # Streaming response: check if we need to return prefill first (for /generate endpoint)
    is_generate_endpoint = endpoint_path == "/generate"

    if is_generate_endpoint:
        # For /generate: return prefill stream first, then decode stream
        async def stream_generator():
            prefill_closed = False
            try:
                async for chunk in prefill_resp.aiter_bytes():
                    yield chunk
                await prefill_ctx.__aexit__(None, None, None)
                prefill_closed = True

                async for chunk in decode_resp.aiter_bytes():
                    yield chunk
                logger.info("Communication %s completed, sending EOS", comm_id)
                yield EOS + DELIM
            finally:
                if not prefill_closed:
                    await prefill_ctx.__aexit__(None, None, None)
                await decode_ctx.__aexit__(None, None, None)
    else:
        # For other endpoints: drain prefill in background, forward decode stream
        asyncio.create_task(_drain_prefill())

        async def stream_generator():
            try:
                async for chunk in decode_resp.aiter_bytes():
                    yield chunk
                logger.info("Communication %s completed, sending EOS", comm_id)
                yield EOS + DELIM
            finally:
                await decode_ctx.__aexit__(None, None, None)

    # Merge headers from decode response (prioritize decode headers over prefill)
    response_headers = dict(decode_resp.headers)
    elapsed_ms = (time.time_ns() - router_start_ns) // 1_000_000
    logger.info(
        "[ROUTER-3] comm_id=%s | Starting stream response | elapsed=%dms",
        comm_id,
        elapsed_ms,
    )
    return StreamingResponse(stream_generator(), media_type="application/octet-stream", headers=response_headers)
