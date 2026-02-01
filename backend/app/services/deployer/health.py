"""Health check and API readiness operations.

This module handles health checking for deployed models,
including waiting for APIs to become ready.
"""

import asyncio
import json
import logging

import httpx
from sqlalchemy import select

from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import BackendType

logger = logging.getLogger(__name__)

# Health check configuration constants
HEALTH_CHECK_INTERVAL = 5  # seconds between checks
HEALTH_CHECK_SLOW_THRESHOLD = 600  # seconds before showing "slow loading" message (10 min)
HEALTH_CHECK_REQUEST_TIMEOUT = 10  # timeout for each health check request


async def wait_for_ollama_ready(
    worker_address: str,
    port: int,
    timeout: int = 60,
    container_name: str | None = None,
) -> bool:
    """Wait for Ollama API to be available.

    Args:
        worker_address: Worker address (host:port)
        port: Ollama container port
        timeout: Maximum wait time in seconds
        container_name: Container name for Docker network (Windows compatibility)

    Returns:
        True if Ollama is ready, False on timeout
    """
    # Ollama is configured to use port 8000 (OLLAMA_HOST=0.0.0.0:8000)
    if container_name:
        api_url = f"http://{container_name}:8000/api/tags"
    else:
        worker_ip = worker_address.split(":")[0]
        api_url = f"http://{worker_ip}:{port}/api/tags"

    logger.info(f"Waiting for Ollama API at {api_url}")

    elapsed = 0
    check_interval = 2

    async with httpx.AsyncClient(timeout=10.0) as client:
        while elapsed < timeout:
            try:
                response = await client.get(api_url)
                if response.status_code == 200:
                    logger.info(f"Ollama API ready after {elapsed}s")
                    return True
            except httpx.ConnectError:
                logger.debug(f"Ollama not ready yet ({elapsed}s)")
            except Exception as e:
                logger.debug(f"Ollama check error: {e}")

            await asyncio.sleep(check_interval)
            elapsed += check_interval

    logger.error(f"Ollama API not ready after {timeout}s")
    return False


async def ollama_pull_model(
    worker_address: str,
    port: int,
    model_id: str,
    container_name: str | None = None,
) -> bool:
    """Pull a model using Ollama API.

    Ollama requires models to be pulled before they can be used.
    This method calls the /api/pull endpoint and waits for completion.
    """
    # Ollama is configured to use port 8000 (OLLAMA_HOST=0.0.0.0:8000)
    if container_name:
        api_url = f"http://{container_name}:8000/api/pull"
    else:
        worker_ip = worker_address.split(":")[0]
        api_url = f"http://{worker_ip}:{port}/api/pull"

    logger.info(f"Pulling Ollama model: {model_id}")

    try:
        async with httpx.AsyncClient(timeout=1800.0) as client:  # 30 min timeout
            # Ollama pull is a streaming endpoint
            async with client.stream(
                "POST",
                api_url,
                json={"name": model_id, "stream": True},
            ) as response:
                if response.status_code != 200:
                    logger.error(f"Ollama pull failed: {response.status_code}")
                    return False

                # Process the streaming response
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")

                            # Log progress
                            if "completed" in data and "total" in data:
                                pct = int(data["completed"] / data["total"] * 100)
                                logger.debug(f"Ollama pull: {status} ({pct}%)")
                            elif status:
                                logger.debug(f"Ollama pull: {status}")

                            # Check for completion
                            if status == "success":
                                logger.info(f"Ollama model {model_id} pulled successfully")
                                return True

                        except Exception as e:
                            logger.debug(f"Error parsing Ollama response: {e}")

                logger.info(f"Ollama model {model_id} pull completed")
                return True

    except httpx.ConnectError:
        logger.error(f"Cannot connect to Ollama at {api_url}")
        return False
    except Exception as e:
        logger.error(f"Ollama pull error: {e}")
        return False


async def wait_for_api_ready(
    worker_address: str,
    port: int,
    deployment_id: int,
    db,
    backend: str = BackendType.VLLM.value,
    container_name: str | None = None,
) -> bool | None:
    """
    Poll the OpenAI API endpoint until it's ready or cancelled.

    Args:
        worker_address: Worker address (host:port)
        port: Host port for the model API
        deployment_id: Deployment ID for status updates
        db: Database session
        backend: Backend type (vllm, ollama, etc.)
        container_name: Container name for local Docker network communication.
                       If set, uses container_name:8000 instead of worker_ip:port.
                       This is needed for Windows Docker Desktop compatibility.

    Returns:
        True: API is ready
        None: Cancelled (user stopped deployment)
    """
    # For local deployments with container_name, use Docker internal networking
    # All backends (vLLM, SGLang, Ollama) are configured to use port 8000
    if container_name:
        api_base_url = f"http://{container_name}:8000"
        logger.info(f"Using Docker network for API: {api_base_url}")
    else:
        worker_ip = worker_address.split(":")[0]
        api_base_url = f"http://{worker_ip}:{port}"

    # Both vLLM and Ollama support OpenAI-compatible /v1/models endpoint
    health_endpoint = f"{api_base_url}/v1/models"

    # For Ollama, we can also check /api/tags as a fallback
    is_ollama = backend == BackendType.OLLAMA.value

    elapsed = 0
    check_count = 0
    shown_slow_message = False

    logger.info(f"Waiting for API to be ready at {health_endpoint} (backend={backend})")

    async with httpx.AsyncClient(timeout=HEALTH_CHECK_REQUEST_TIMEOUT) as client:
        while True:  # Wait indefinitely until ready or cancelled
            check_count += 1

            # Check if deployment was cancelled
            try:
                result = await db.execute(select(Deployment).where(Deployment.id == deployment_id))
                deployment = result.scalar_one_or_none()
                if deployment and deployment.status in [
                    DeploymentStatus.STOPPED.value,
                    DeploymentStatus.STOPPING.value,
                ]:
                    logger.info(f"Deployment {deployment_id} was cancelled")
                    return None
            except Exception as e:
                logger.debug(f"Error checking deployment status: {e}")

            try:
                response = await client.get(health_endpoint)

                if response.status_code == 200:
                    data = response.json()
                    # vLLM returns {"object": "list", "data": [...]}
                    # Ollama returns {"object": "list", "data": [...]} (OpenAI compat)
                    if data.get("data") and len(data["data"]) > 0:
                        logger.info(
                            f"API ready at {health_endpoint} after {elapsed}s "
                            f"({check_count} checks)"
                        )
                        return True

                # For Ollama, also try the native /api/tags endpoint
                if is_ollama and response.status_code != 200:
                    ollama_endpoint = f"{api_base_url}/api/tags"
                    ollama_response = await client.get(ollama_endpoint)
                    if ollama_response.status_code == 200:
                        ollama_data = ollama_response.json()
                        if ollama_data.get("models") and len(ollama_data["models"]) > 0:
                            logger.info(f"Ollama API ready at {ollama_endpoint} after {elapsed}s")
                            return True

                logger.debug(f"Health check {check_count}: status={response.status_code}")

            except httpx.ConnectError:
                # Container not ready yet, this is expected during startup
                logger.debug(f"Health check {check_count}: connection refused")
            except httpx.ReadTimeout:
                logger.debug(f"Health check {check_count}: read timeout")
            except Exception as e:
                logger.debug(f"Health check {check_count}: {type(e).__name__}: {e}")

            # Update status message periodically
            if check_count % 6 == 0:  # Every 30 seconds
                try:
                    result = await db.execute(
                        select(Deployment).where(Deployment.id == deployment_id)
                    )
                    deployment = result.scalar_one_or_none()
                    if deployment and deployment.status == DeploymentStatus.STARTING.value:
                        mins = elapsed // 60
                        secs = elapsed % 60
                        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

                        # Show patience message after threshold
                        if elapsed >= HEALTH_CHECK_SLOW_THRESHOLD and not shown_slow_message:
                            deployment.status_message = (
                                f"Loading model... ({time_str}) - "
                                "Large model or slow network detected. Please be patient."
                            )
                            shown_slow_message = True
                        elif shown_slow_message:
                            deployment.status_message = (
                                f"Loading model... ({time_str}) - Please be patient."
                            )
                        else:
                            deployment.status_message = (
                                f"Loading model into GPU memory... ({time_str})"
                            )
                        await db.commit()
                except Exception as e:
                    logger.debug(f"Error updating deployment status message: {e}")

            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            elapsed += HEALTH_CHECK_INTERVAL


async def wait_for_native_api_ready(
    worker_address: str,
    port: int,
    deployment_id: int,
    db,
    backend: str = BackendType.OLLAMA.value,
    timeout: int = 300,
) -> bool | None:
    """Wait for native API to be ready.

    For native deployments, we check via worker agent because
    Ollama only listens on localhost.

    Returns:
        True: API is ready
        False: Timeout
        None: Cancelled
    """
    # For native deployments, check via worker agent's health endpoint
    worker_health_url = f"http://{worker_address}/native/health"

    elapsed = 0
    check_interval = 5

    async with httpx.AsyncClient(timeout=10.0) as client:
        while elapsed < timeout:
            # Check if cancelled
            try:
                result = await db.execute(select(Deployment).where(Deployment.id == deployment_id))
                deployment = result.scalar_one_or_none()
                if deployment and deployment.status in [
                    DeploymentStatus.STOPPED.value,
                    DeploymentStatus.STOPPING.value,
                ]:
                    return None
            except Exception:
                pass

            try:
                response = await client.get(
                    worker_health_url, params={"backend": backend, "port": port}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ready"):
                        logger.info("Native API ready (checked via worker)")
                        return True
            except Exception:
                pass

            await asyncio.sleep(check_interval)
            elapsed += check_interval

            # Update status periodically
            if elapsed % 30 == 0:
                try:
                    result = await db.execute(
                        select(Deployment).where(Deployment.id == deployment_id)
                    )
                    deployment = result.scalar_one_or_none()
                    if deployment:
                        deployment.status_message = f"Loading model... ({elapsed}s)"
                        await db.commit()
                except Exception:
                    pass

    return False
