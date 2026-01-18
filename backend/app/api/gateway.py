"""API Gateway routes - OpenAI-compatible proxy endpoints"""

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import async_session_maker, get_db
from app.models.deployment import Deployment
from app.services.gateway import gateway_service
from app.services.semantic_router import semantic_router_service

# Special model names that trigger semantic routing
SEMANTIC_ROUTER_MODEL_NAMES = {"mom", "mixture-of-models", "auto", "semantic-router"}

logger = logging.getLogger(__name__)

router = APIRouter()

# HTTP client configuration
HTTP_TIMEOUT = 300.0  # 5 minutes for long-running model requests


async def get_api_key_from_header(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """Dependency to validate API key from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing Authorization header",
                    "type": "invalid_request_error",
                }
            },
        )

    access_key, secret_key = gateway_service.parse_api_key(authorization)

    if not access_key or not secret_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key format",
                    "type": "invalid_request_error",
                }
            },
        )

    api_key = await gateway_service.validate_api_key(db, access_key, secret_key)

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid or expired API key",
                    "type": "invalid_request_error",
                }
            },
        )

    return api_key


@router.get("/models")
async def list_models(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """List available models (OpenAI-compatible)."""
    api_key = None

    # If authorization provided, validate and filter by allowed models
    if authorization:
        access_key, secret_key = gateway_service.parse_api_key(authorization)
        if access_key and secret_key:
            api_key = await gateway_service.validate_api_key(db, access_key, secret_key)

    models = await gateway_service.get_available_models(db, api_key)

    return {
        "object": "list",
        "data": models,
    }


@router.get("/models/{model_id:path}")
async def get_model(
    model_id: str,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """Get specific model information (OpenAI-compatible)."""
    # Find the model
    result = await gateway_service.find_deployment_for_model(db, model_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_id}' not found",
                    "type": "model_not_found",
                }
            },
        )

    deployment, model = result

    return {
        "id": model.name,
        "object": "model",
        "created": int(model.created_at.timestamp()) if model.created_at else 0,
        "owned_by": "lmstack",
        "root": model.model_id,
        "parent": None,
        "permission": [
            {
                "id": f"modelperm-{model.id}",
                "object": "model_permission",
                "created": int(model.created_at.timestamp()) if model.created_at else 0,
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False,
            }
        ],
    }


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """Proxy chat completions request (OpenAI-compatible)."""
    # Validate API key
    api_key = await get_api_key_from_header(authorization, db)

    # Get request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                }
            },
        )

    model_name = body.get("model")
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "model is required",
                    "type": "invalid_request_error",
                }
            },
        )

    # Check if using semantic routing (model="MoM", "auto", etc.)
    if model_name.lower() in SEMANTIC_ROUTER_MODEL_NAMES:
        return await _proxy_to_semantic_router(
            db=db,
            body=body,
            api_key=api_key,
            endpoint="/v1/chat/completions",
        )

    # Find deployment for model
    result = await gateway_service.find_deployment_for_model(db, model_name)
    if not result:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_name}' not found or not available",
                    "type": "model_not_found",
                }
            },
        )

    deployment, model = result

    # Check model access
    if not await gateway_service.check_model_access(api_key, model.id):
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": f"API key does not have access to model '{model_name}'",
                    "type": "permission_error",
                }
            },
        )

    # Get worker address
    result = await db.execute(
        select(Deployment)
        .options(selectinload(Deployment.worker))
        .where(Deployment.id == deployment.id)
    )
    deployment = result.scalar_one()

    upstream_url = gateway_service.build_upstream_url(
        deployment.worker.address,
        deployment.port,
        deployment.container_name,
    )

    # Replace model name with the actual model_id for vLLM
    body["model"] = model.model_id

    # Check if streaming
    is_streaming = body.get("stream", False)

    if is_streaming:
        return await proxy_streaming_request(
            upstream_url=f"{upstream_url}/v1/chat/completions",
            body=body,
            api_key_id=api_key.id,
            model_id=model.id,
            deployment_id=deployment.id,
            db=db,
        )
    else:
        return await proxy_request(
            upstream_url=f"{upstream_url}/v1/chat/completions",
            body=body,
            api_key_id=api_key.id,
            model_id=model.id,
            deployment_id=deployment.id,
            db=db,
        )


@router.post("/completions")
async def completions(
    request: Request,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """Proxy completions request (OpenAI-compatible)."""
    # Validate API key
    api_key = await get_api_key_from_header(authorization, db)

    # Get request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                }
            },
        )

    model_name = body.get("model")
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "model is required",
                    "type": "invalid_request_error",
                }
            },
        )

    # Find deployment for model
    result = await gateway_service.find_deployment_for_model(db, model_name)
    if not result:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_name}' not found or not available",
                    "type": "model_not_found",
                }
            },
        )

    deployment, model = result

    # Check model access
    if not await gateway_service.check_model_access(api_key, model.id):
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": f"API key does not have access to model '{model_name}'",
                    "type": "permission_error",
                }
            },
        )

    # Get worker address
    result = await db.execute(
        select(Deployment)
        .options(selectinload(Deployment.worker))
        .where(Deployment.id == deployment.id)
    )
    deployment = result.scalar_one()

    upstream_url = gateway_service.build_upstream_url(
        deployment.worker.address,
        deployment.port,
        deployment.container_name,
    )

    # Replace model name with the actual model_id for vLLM
    body["model"] = model.model_id

    # Check if streaming
    is_streaming = body.get("stream", False)

    if is_streaming:
        return await proxy_streaming_request(
            upstream_url=f"{upstream_url}/v1/completions",
            body=body,
            api_key_id=api_key.id,
            model_id=model.id,
            deployment_id=deployment.id,
            db=db,
        )
    else:
        return await proxy_request(
            upstream_url=f"{upstream_url}/v1/completions",
            body=body,
            api_key_id=api_key.id,
            model_id=model.id,
            deployment_id=deployment.id,
            db=db,
        )


async def proxy_request(
    upstream_url: str,
    body: dict,
    api_key_id: int,
    model_id: int,
    deployment_id: int,
    db: AsyncSession,
) -> JSONResponse:
    """Proxy a non-streaming request to upstream."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                upstream_url,
                json=body,
                headers={"Content-Type": "application/json"},
            )

            response_data = response.json()

            # Record usage
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            await gateway_service.record_usage(
                db=db,
                api_key_id=api_key_id,
                model_id=model_id,
                deployment_id=deployment_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            return JSONResponse(content=response_data, status_code=response.status_code)

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "message": "Request to model timed out",
                    "type": "timeout_error",
                }
            },
        )
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": "Failed to connect to model",
                    "type": "connection_error",
                }
            },
        )


async def record_usage_background(
    api_key_id: int,
    model_id: int,
    deployment_id: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Record usage in a background task with its own database session."""
    async with async_session_maker() as db:
        await gateway_service.record_usage(
            db=db,
            api_key_id=api_key_id,
            model_id=model_id,
            deployment_id=deployment_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


async def proxy_streaming_request(
    upstream_url: str,
    body: dict,
    api_key_id: int,
    model_id: int,
    deployment_id: int,
    db: AsyncSession,
) -> StreamingResponse:
    """Proxy a streaming request to upstream."""
    # Store usage info to record after streaming completes
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0}

    async def stream_generator() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    upstream_url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

                        # Try to extract usage from final chunk
                        try:
                            chunk_str = chunk.decode("utf-8")
                            for line in chunk_str.split("\n"):
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    data = json.loads(line[6:])
                                    if "usage" in data:
                                        usage_info["prompt_tokens"] = data["usage"].get(
                                            "prompt_tokens", 0
                                        )
                                        usage_info["completion_tokens"] = data["usage"].get(
                                            "completion_tokens", 0
                                        )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass  # Expected for binary chunks or incomplete JSON

        except httpx.TimeoutException:
            logger.error(f"Streaming timeout for {upstream_url}")
            error_data = {
                "error": {
                    "message": "Request to model timed out",
                    "type": "timeout_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n".encode()
        except httpx.RequestError as e:
            logger.error(f"Streaming connection error: {e}")
            error_data = {
                "error": {
                    "message": f"Connection error: {e}",
                    "type": "connection_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n".encode()
        except Exception as e:
            logger.exception(f"Unexpected streaming error: {e}")
            error_data = {"error": {"message": str(e), "type": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n".encode()

        # Estimate tokens if not provided by stream
        if usage_info["prompt_tokens"] == 0:
            messages = body.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    usage_info["prompt_tokens"] += len(content) // 4

        # Record usage with a new database session (background task)
        await record_usage_background(
            api_key_id=api_key_id,
            model_id=model_id,
            deployment_id=deployment_id,
            prompt_tokens=usage_info["prompt_tokens"],
            completion_tokens=usage_info["completion_tokens"],
        )

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/embeddings")
async def embeddings(
    request: Request,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """Proxy embeddings request (OpenAI-compatible)."""
    # Validate API key
    api_key = await get_api_key_from_header(authorization, db)

    # Get request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                }
            },
        )

    model_name = body.get("model")
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "model is required",
                    "type": "invalid_request_error",
                }
            },
        )

    # Find deployment for model
    result = await gateway_service.find_deployment_for_model(db, model_name)
    if not result:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_name}' not found or not available",
                    "type": "model_not_found",
                }
            },
        )

    deployment, model = result

    # Check model access
    if not await gateway_service.check_model_access(api_key, model.id):
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": f"API key does not have access to model '{model_name}'",
                    "type": "permission_error",
                }
            },
        )

    # Get worker address
    result = await db.execute(
        select(Deployment)
        .options(selectinload(Deployment.worker))
        .where(Deployment.id == deployment.id)
    )
    deployment = result.scalar_one()

    upstream_url = gateway_service.build_upstream_url(
        deployment.worker.address,
        deployment.port,
        deployment.container_name,
    )

    # Replace model name with the actual model_id for vLLM
    body["model"] = model.model_id

    return await proxy_request(
        upstream_url=f"{upstream_url}/v1/embeddings",
        body=body,
        api_key_id=api_key.id,
        model_id=model.id,
        deployment_id=deployment.id,
        db=db,
    )


def convert_responses_input_to_messages(
    input_items: list[dict], instructions: str = None
) -> list[dict]:
    """Convert OpenAI Responses API input format to Chat Completions messages format.

    Responses API: input = [{role: "user", content: [{type: "input_text", text: "..."}]}]
    Chat Completions: messages = [{role: "user", content: "..."}]
    """
    messages = []

    # Add system instructions if provided
    if instructions:
        messages.append({"role": "system", "content": instructions})

    for item in input_items:
        role = item.get("role", "user")
        content_items = item.get("content", [])

        # Handle string content directly
        if isinstance(content_items, str):
            messages.append({"role": role, "content": content_items})
            continue

        # Extract text content from content items
        text_parts = []
        for content_item in content_items:
            if isinstance(content_item, str):
                text_parts.append(content_item)
            elif isinstance(content_item, dict):
                content_type = content_item.get("type", "")
                if content_type == "input_text":
                    text_parts.append(content_item.get("text", ""))
                elif content_type == "text":
                    text_parts.append(content_item.get("text", ""))

        if text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    return messages


def strip_thinking_tags(content: str) -> str:
    """Strip <think>...</think> tags from Qwen3 model responses."""
    if not content:
        return ""
    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    return cleaned.strip()


def convert_chat_response_to_responses_format(
    chat_response: dict,
    model: str,
) -> dict:
    """Convert Chat Completions response to Responses API format.

    Chat Completions: {choices: [{message: {role, content}}], usage: {...}}
    Responses API: {id, object, model, output: [{type: "message", ...}], usage: {...}}
    """
    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    output = []
    choices = chat_response.get("choices", [])

    for choice in choices:
        message = choice.get("message", {})
        role = message.get("role", "assistant")
        content = message.get("content") or ""

        # Handle content that might be None (thinking models)
        if content is None:
            content = ""

        # Strip <think>...</think> tags from Qwen3 and similar models
        content = strip_thinking_tags(content)

        output.append(
            {
                "type": "message",
                "role": role,
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                    }
                ],
            }
        )

    usage = chat_response.get("usage", {})

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


async def proxy_responses_streaming(
    upstream_url: str,
    chat_body: dict,
    model_name: str,
) -> StreamingResponse:
    """Proxy streaming request and convert to Responses API streaming format."""
    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    async def stream_generator() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    upstream_url,
                    json=chat_body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    # Send initial response.created event
                    created_event = {
                        "type": "response.created",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": int(time.time()),
                            "model": model_name,
                            "status": "in_progress",
                            "output": [],
                        },
                    }
                    yield f"event: response.created\ndata: {json.dumps(created_event)}\n\n".encode()

                    # Send output item added
                    item_added_event = {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                        },
                    }
                    yield f"event: response.output_item.added\ndata: {json.dumps(item_added_event)}\n\n".encode()

                    # Send content part added
                    content_added_event = {
                        "type": "response.content_part.added",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                        },
                    }
                    yield f"event: response.content_part.added\ndata: {json.dumps(content_added_event)}\n\n".encode()

                    # Process streaming chunks from chat/completions
                    full_content = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                continue
                            try:
                                chunk_data = json.loads(data_str)
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_content += content
                                        # Send text delta
                                        delta_event = {
                                            "type": "response.output_text.delta",
                                            "output_index": 0,
                                            "content_index": 0,
                                            "delta": content,
                                        }
                                        yield f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n".encode()
                            except json.JSONDecodeError:
                                pass

                    # Send content part done
                    content_done_event = {
                        "type": "response.content_part.done",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": full_content,
                        },
                    }
                    yield f"event: response.content_part.done\ndata: {json.dumps(content_done_event)}\n\n".encode()

                    # Send output item done
                    item_done_event = {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": full_content}],
                        },
                    }
                    yield f"event: response.output_item.done\ndata: {json.dumps(item_done_event)}\n\n".encode()

                    # Send response completed
                    completed_event = {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": int(time.time()),
                            "model": model_name,
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": full_content}],
                                }
                            ],
                        },
                    }
                    yield f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n".encode()

        except Exception as e:
            logger.error(f"Responses streaming error: {e}")
            error_event = {
                "type": "error",
                "error": {"message": str(e), "type": "stream_error"},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/responses")
async def responses(
    request: Request,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """OpenAI Responses API endpoint (converts to Chat Completions internally).

    This endpoint provides compatibility with OpenAI's newer Responses API
    by translating requests to the Chat Completions format.
    """
    # Validate API key
    api_key = await get_api_key_from_header(authorization, db)

    # Get request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                }
            },
        )

    model_name = body.get("model")
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "model is required",
                    "type": "invalid_request_error",
                }
            },
        )

    # Find deployment for model
    result = await gateway_service.find_deployment_for_model(db, model_name)
    if not result:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_name}' not found or not available",
                    "type": "model_not_found",
                }
            },
        )

    deployment, model = result

    # Check model access
    if not await gateway_service.check_model_access(api_key, model.id):
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": f"API key does not have access to model '{model_name}'",
                    "type": "permission_error",
                }
            },
        )

    # Get worker address
    result = await db.execute(
        select(Deployment)
        .options(selectinload(Deployment.worker))
        .where(Deployment.id == deployment.id)
    )
    deployment = result.scalar_one()

    upstream_url = gateway_service.build_upstream_url(
        deployment.worker.address,
        deployment.port,
        deployment.container_name,
    )

    # Convert Responses API format to Chat Completions format
    input_items = body.get("input", [])
    instructions = body.get("instructions", "")
    messages = convert_responses_input_to_messages(input_items, instructions)

    # Check if streaming is requested
    is_streaming = body.get("stream", False)

    # Build Chat Completions request
    chat_body = {
        "model": model.model_id,
        "messages": messages,
        "stream": is_streaming,
    }

    # Copy over compatible options
    if "temperature" in body:
        chat_body["temperature"] = body["temperature"]
    if "top_p" in body:
        chat_body["top_p"] = body["top_p"]
    if "max_output_tokens" in body:
        chat_body["max_tokens"] = body["max_output_tokens"]

    if is_streaming:
        # Streaming response
        return await proxy_responses_streaming(
            upstream_url=f"{upstream_url}/v1/chat/completions",
            chat_body=chat_body,
            model_name=model_name,
        )

    # Non-streaming response
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{upstream_url}/v1/chat/completions",
                json=chat_body,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.error(f"Backend error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail={
                        "error": {
                            "message": f"Backend error: {response.text}",
                            "type": "backend_error",
                        }
                    },
                )

            chat_response = response.json()

    except httpx.RequestError as e:
        logger.error(f"Request error to backend: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": f"Backend connection error: {str(e)}",
                    "type": "backend_error",
                }
            },
        )

    # Convert response to Responses API format
    responses_response = convert_chat_response_to_responses_format(
        chat_response,
        model_name,
    )

    logger.info(f"Responses API response: {json.dumps(responses_response)[:500]}")

    return JSONResponse(content=responses_response)


# =============================================================================
# Semantic Router Proxy
# =============================================================================


async def _proxy_to_semantic_router(
    db: AsyncSession,
    body: dict,
    api_key,
    endpoint: str,
) -> JSONResponse | StreamingResponse:
    """Proxy request to Semantic Router for intelligent model selection.

    Args:
        db: Database session
        body: Request body
        api_key: Validated API key
        endpoint: API endpoint (e.g., /v1/chat/completions)

    Returns:
        Response from Semantic Router
    """
    # Check if Semantic Router is deployed
    router_url = await semantic_router_service.get_semantic_router_url(db)
    if not router_url:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Semantic Router is not deployed. Deploy it from the Apps page to use automatic model routing.",
                    "type": "service_unavailable",
                    "hint": "Use a specific model name instead, or deploy Semantic Router first.",
                }
            },
        )

    # Remove the special model name and let Semantic Router decide
    # The router will use its config to select the best model
    body_copy = body.copy()
    body_copy.pop("model", None)  # Let Semantic Router handle model selection

    upstream_url = f"{router_url}{endpoint}"
    is_streaming = body.get("stream", False)

    if is_streaming:
        return await _proxy_semantic_router_streaming(upstream_url, body_copy, api_key.id)
    else:
        return await _proxy_semantic_router_request(upstream_url, body_copy, api_key.id)


async def _proxy_semantic_router_request(
    upstream_url: str,
    body: dict,
    api_key_id: int,
) -> JSONResponse:
    """Proxy non-streaming request to Semantic Router."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                upstream_url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            return JSONResponse(content=response.json(), status_code=response.status_code)

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "message": "Request to Semantic Router timed out",
                    "type": "timeout_error",
                }
            },
        )
    except httpx.RequestError as e:
        logger.error(f"Semantic Router request error: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": "Failed to connect to Semantic Router",
                    "type": "connection_error",
                }
            },
        )


async def _proxy_semantic_router_streaming(
    upstream_url: str,
    body: dict,
    api_key_id: int,
) -> StreamingResponse:
    """Proxy streaming request to Semantic Router."""

    async def stream_generator() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    upstream_url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        except httpx.TimeoutException:
            error_data = {
                "error": {
                    "message": "Request to Semantic Router timed out",
                    "type": "timeout_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n".encode()
        except httpx.RequestError as e:
            logger.error(f"Semantic Router streaming error: {e}")
            error_data = {
                "error": {
                    "message": f"Connection error: {e}",
                    "type": "connection_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n".encode()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
