"""Chat Proxy API - Proxy requests to external OpenAI-compatible endpoints."""

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.deps import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()

HTTP_TIMEOUT = 300.0  # 5 minutes


class ChatProxyRequest(BaseModel):
    """Request body for chat proxy endpoint."""

    endpoint: str
    api_key: str | None = None
    payload: dict


class FetchModelsRequest(BaseModel):
    """Request body for fetching models from external endpoint."""

    endpoint: str
    api_key: str | None = None


@router.post("/chat-proxy")
async def proxy_chat_request(
    request: ChatProxyRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Proxy chat requests to external OpenAI-compatible endpoints.

    This endpoint allows the frontend to make requests to external LLM APIs
    without running into CORS issues.
    """
    # Normalize endpoint URL
    endpoint = request.endpoint.strip()
    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]

    # Append /chat/completions if not present
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    # Build headers
    headers = {"Content-Type": "application/json"}
    if request.api_key:
        headers["Authorization"] = f"Bearer {request.api_key}"

    # Log request details for debugging
    logger.info(f"Proxying request to: {endpoint}")
    logger.info(f"Payload model: {request.payload.get('model', 'not specified')}")

    # Check if streaming is requested
    is_streaming = request.payload.get("stream", False)

    if is_streaming:
        # Streaming response - client lifecycle managed inside generator
        async def stream_response():
            client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
            try:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=request.payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"data: {error_text.decode()}\n\n"
                        return

                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.ConnectError as e:
                logger.error(f"Connection error to {endpoint}: {e}")
                yield 'data: {"error": "Failed to connect to endpoint"}\n\n'
            except httpx.TimeoutException as e:
                logger.error(f"Timeout connecting to {endpoint}: {e}")
                yield 'data: {"error": "Request timed out"}\n\n'
            except Exception as e:
                logger.error(f"Error proxying request to {endpoint}: {e}")
                yield f'data: {{"error": "{str(e)}"}}\n\n'
            finally:
                await client.aclose()

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming response
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.post(
                    endpoint,
                    json=request.payload,
                    headers=headers,
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=response.text,
                    )

                return response.json()

        except httpx.ConnectError as e:
            logger.error(f"Connection error to {endpoint}: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to connect to endpoint: {endpoint}",
            )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to {endpoint}: {e}")
            raise HTTPException(
                status_code=504,
                detail="Request to endpoint timed out",
            )
        except Exception as e:
            logger.error(f"Error proxying request to {endpoint}: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e),
            )


@router.post("/fetch-models")
async def fetch_remote_models(
    request: FetchModelsRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Fetch available models from an external OpenAI-compatible endpoint.

    Returns a list of model IDs available at the endpoint.
    """
    # Normalize endpoint URL
    endpoint = request.endpoint.strip()
    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]

    # Build models endpoint
    models_endpoint = f"{endpoint}/models"

    # Build headers
    headers = {"Content-Type": "application/json"}
    if request.api_key:
        headers["Authorization"] = f"Bearer {request.api_key}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(models_endpoint, headers=headers)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch models: {response.text}",
                )

            data = response.json()

            # Extract model IDs from OpenAI-compatible response
            models = []
            if "data" in data:
                for model in data["data"]:
                    model_id = model.get("id")
                    if model_id:
                        models.append(
                            {
                                "id": model_id,
                                "owned_by": model.get("owned_by", "unknown"),
                            }
                        )

            return {"models": models}

    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to endpoint: {models_endpoint}",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Request to endpoint timed out",
        )
    except Exception as e:
        logger.error(f"Error fetching models from {models_endpoint}: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
