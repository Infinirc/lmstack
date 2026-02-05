"""Deployment routes for LMStack Worker.

Contains endpoints for deploying, stopping, and managing model containers.
"""

import logging
import re

from fastapi import APIRouter, HTTPException

try:
    from models import AdoptRequest, DeployRequest, StopRequest
except ImportError:
    from worker.models import AdoptRequest, DeployRequest, StopRequest

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

    # Check if Docker is available
    if agent.docker is None:
        raise HTTPException(
            status_code=400,
            detail="Docker not available on this worker. Use native deployment for macOS.",
        )

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

    # Check if Docker is available
    if agent.docker is None:
        raise HTTPException(
            status_code=400,
            detail="Docker not available on this worker.",
        )

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


@router.get("/discover")
async def discover():
    """Discover unmanaged inference containers on this worker.

    Scans running Docker containers for vllm/sglang/ollama images
    that are not already managed by LMStack. For each discovered
    container, probes its API to determine the loaded model.
    """
    agent = get_agent()

    if agent.container_manager is None:
        return {"items": []}

    try:
        discovered = agent.container_manager.discover_unmanaged_containers()
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Probe each container's API for model info
    import httpx

    for container in discovered:
        port = container.get("port")
        if not port:
            continue

        backend = container.get("backend")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                if backend in ("vllm", "sglang"):
                    resp = await client.get(f"http://localhost:{port}/v1/models")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("data", [])
                        if models:
                            container["model_id"] = models[0].get("id", container.get("model_id"))
                elif backend == "ollama":
                    resp = await client.get(f"http://localhost:{port}/api/tags")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("models", [])
                        if models:
                            container["model_id"] = models[0].get("name", container.get("model_id"))
        except Exception:
            # Probing failed, keep whatever model_id we parsed from command
            pass

    return {"items": discovered}


@router.post("/adopt")
async def adopt(request: AdoptRequest):
    """Mark a container as adopted by LMStack."""
    agent = get_agent()

    if agent.container_manager is None:
        raise HTTPException(status_code=400, detail="Docker not available on this worker")

    try:
        result = agent.container_manager.adopt_container(request.container_id)
        return result
    except Exception as e:
        logger.error(f"Adoption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
