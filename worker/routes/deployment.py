"""Deployment routes for LMStack Worker.

Contains endpoints for deploying, stopping, and managing model containers.
"""

import logging
import re

from fastapi import APIRouter, HTTPException

try:
    from models import DeployRequest, StopRequest
except ImportError:
    from worker.models import DeployRequest, StopRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["deployment"])

# Agent reference - set by main app
_agent = None


def set_agent(agent):
    """Set the agent reference for route handlers."""
    global _agent
    _agent = agent


def get_agent():
    """Get the agent reference or raise error."""
    if _agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    return _agent


def sanitize_container_name(name: str) -> str:
    """Sanitize container name to comply with Docker naming rules.

    Docker container names must match [a-zA-Z0-9][a-zA-Z0-9_.-]
    Common issues: Ollama model IDs contain ':' (e.g., 'qwen2.5:7b')
    """
    # Replace colons with dashes (common in Ollama model IDs)
    sanitized = name.replace(":", "-")
    # Replace any other invalid characters with dashes
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "-", sanitized)
    # Ensure it starts with alphanumeric
    if sanitized and not sanitized[0].isalnum():
        sanitized = "c" + sanitized
    # Remove consecutive dashes
    sanitized = re.sub(r"-+", "-", sanitized)
    # Strip trailing dashes
    sanitized = sanitized.strip("-")
    return sanitized or "container"


@router.post("/deploy")
async def deploy(request: DeployRequest):
    """Deploy a model container."""
    agent = get_agent()

    port_info = f" on port {request.port}" if request.port else ""
    logger.info(f"Deploying {request.deployment_name} with image {request.image}{port_info}")

    # Sanitize container name to handle special characters
    container_name = sanitize_container_name(f"lmstack-{request.deployment_name}")

    try:
        container_id, port = await agent.docker.run(
            name=container_name,
            image=request.image,
            command=request.command,
            gpu_indexes=request.gpu_indexes,
            environment=request.environment,
            deployment_id=request.deployment_id,
            port=request.port,
        )

        logger.info(f"Container {container_id[:12]} started on port {port}")

        return {
            "container_id": container_id,
            "port": port,
        }

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop(request: StopRequest):
    """Stop a container."""
    agent = get_agent()
    logger.info(f"Stopping container {request.container_id[:12]}")

    try:
        await agent.docker.stop(request.container_id)
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Failed to stop container: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def logs(container_id: str, tail: int = 100):
    """Get container logs."""
    agent = get_agent()

    try:
        logs_output = await agent.docker.logs(container_id, tail=tail)
        return {"logs": logs_output}
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        return {"logs": f"Error: {str(e)}"}


@router.get("/gpus")
async def gpus():
    """Get GPU information."""
    agent = get_agent()
    return {"gpus": agent.gpu_detector.detect()}


@router.get("/pull-progress/{deployment_id}")
async def pull_progress(deployment_id: int):
    """Get image pull progress for a deployment."""
    try:
        from docker_ops.runner import get_pull_progress
    except ImportError:
        from worker.docker_ops.runner import get_pull_progress
    return get_pull_progress(deployment_id)
