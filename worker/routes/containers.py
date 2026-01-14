"""Container management routes for LMStack Worker.

Contains endpoints for listing, creating, starting, stopping,
and managing Docker containers.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

try:
    from models import ContainerCreateRequest, ContainerExecRequest
except ImportError:
    from worker.models import ContainerCreateRequest, ContainerExecRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/containers", tags=["containers"])

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


@router.get("")
async def list_containers(all: bool = True, managed_only: bool = False):
    """List all containers on this worker."""
    agent = get_agent()

    try:
        containers = agent.container_manager.list_containers(
            all=all,
            managed_only=managed_only,
        )
        return {"items": containers, "total": len(containers)}
    except Exception as e:
        logger.error(f"Failed to list containers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{container_id}")
async def get_container(container_id: str):
    """Get container details."""
    agent = get_agent()

    try:
        return agent.container_manager.get_container_detail(container_id)
    except Exception as e:
        logger.error(f"Failed to get container {container_id}: {e}")
        raise HTTPException(
            status_code=404, detail=f"Container not found: {container_id}"
        )


@router.get("/{container_id}/stats")
async def get_container_stats(container_id: str):
    """Get container resource stats."""
    agent = get_agent()

    try:
        return agent.container_manager.get_container_stats(container_id)
    except Exception as e:
        logger.error(f"Failed to get stats for container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_container(request: ContainerCreateRequest):
    """Create and start a new container."""
    agent = get_agent()

    try:
        return agent.container_manager.create_container(
            name=request.name,
            image=request.image,
            command=request.command,
            entrypoint=request.entrypoint,
            env=request.env,
            ports=request.ports,
            volumes=request.volumes,
            gpu_ids=request.gpu_ids,
            restart_policy=request.restart_policy,
            labels=request.labels,
            cpu_limit=request.cpu_limit,
            memory_limit=request.memory_limit,
            cap_add=request.cap_add,
        )
    except Exception as e:
        logger.error(f"Failed to create container {request.name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{container_id}/start")
async def start_container(container_id: str):
    """Start a stopped container."""
    agent = get_agent()

    try:
        return agent.container_manager.start_container(container_id)
    except Exception as e:
        logger.error(f"Failed to start container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{container_id}/stop")
async def stop_container(container_id: str, timeout: int = 10):
    """Stop a running container."""
    agent = get_agent()

    try:
        return agent.container_manager.stop_container(container_id, timeout=timeout)
    except Exception as e:
        logger.error(f"Failed to stop container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{container_id}/restart")
async def restart_container(container_id: str, timeout: int = 10):
    """Restart a container."""
    agent = get_agent()

    try:
        return agent.container_manager.restart_container(container_id, timeout=timeout)
    except Exception as e:
        logger.error(f"Failed to restart container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{container_id}/pause")
async def pause_container(container_id: str):
    """Pause a running container."""
    agent = get_agent()

    try:
        return agent.container_manager.pause_container(container_id)
    except Exception as e:
        logger.error(f"Failed to pause container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{container_id}/unpause")
async def unpause_container(container_id: str):
    """Unpause a paused container."""
    agent = get_agent()

    try:
        return agent.container_manager.unpause_container(container_id)
    except Exception as e:
        logger.error(f"Failed to unpause container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{container_id}")
async def delete_container(
    container_id: str, force: bool = False, volumes: bool = False
):
    """Delete a container."""
    agent = get_agent()

    try:
        agent.container_manager.remove_container(
            container_id,
            force=force,
            remove_volumes=volumes,
        )
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Failed to delete container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{container_id}/logs")
async def get_container_logs(
    container_id: str,
    tail: int = 100,
    since: Optional[int] = None,
    until: Optional[int] = None,
    timestamps: bool = True,
):
    """Get container logs."""
    agent = get_agent()

    try:
        return agent.container_manager.get_logs(
            container_id=container_id,
            tail=tail,
            since=since,
            until=until,
            timestamps=timestamps,
        )
    except Exception as e:
        logger.error(f"Failed to get logs for container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{container_id}/exec")
async def exec_container_command(container_id: str, request: ContainerExecRequest):
    """Execute a command in a running container."""
    agent = get_agent()

    try:
        return agent.container_manager.exec_command(
            container_id=container_id,
            command=request.command,
            tty=request.tty,
            privileged=request.privileged,
            user=request.user,
            workdir=request.workdir,
            env=request.env,
        )
    except Exception as e:
        logger.error(f"Failed to exec command in container {container_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
