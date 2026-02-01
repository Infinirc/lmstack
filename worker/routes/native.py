"""Native deployment routes for Mac workers.

Handles deployments using native backends (Ollama, MLX, llama.cpp)
without Docker containers.
"""

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from worker.agent import WorkerAgent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["native"])

# Global agent reference (set by agent.py)
_agent: "WorkerAgent | None" = None


def set_agent(agent: "WorkerAgent"):
    """Set the global agent reference."""
    global _agent
    _agent = agent


class NativeDeployRequest(BaseModel):
    """Request to deploy a model using native backend."""

    deployment_id: int
    deployment_name: str
    model_id: str
    backend: str  # ollama, mlx, llama_cpp
    port: int = 0  # 0 = auto-assign
    extra_params: dict | None = None


class NativeDeployResponse(BaseModel):
    """Response from native deployment."""

    process_id: str
    port: int
    backend: str
    status: str


class NativeStopRequest(BaseModel):
    """Request to stop a native deployment."""

    process_id: str


@router.post("/native/deploy", response_model=NativeDeployResponse)
async def native_deploy(request: NativeDeployRequest):
    """Deploy a model using native backend (Ollama, MLX, llama.cpp).

    This endpoint is used for Mac workers without Docker support.
    """
    if not _agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    # Check if native process manager exists
    if not hasattr(_agent, "native_manager"):
        raise HTTPException(
            status_code=400,
            detail="Native deployments not supported on this worker",
        )

    try:
        process_id = f"native-{request.deployment_id}"

        # Determine port
        if request.port == 0:
            # Auto-assign port
            if request.backend == "ollama":
                port = 11434  # Ollama's default port
            else:
                # Find available port
                port = _find_available_port()
        else:
            port = request.port

        # Start the native process
        process = await _agent.native_manager.start_process(
            process_id=process_id,
            backend=request.backend,
            model_id=request.model_id,
            port=port,
            **(request.extra_params or {}),
        )

        return NativeDeployResponse(
            process_id=process.process_id,
            port=process.port,
            backend=process.backend,
            status="running",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Native deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/native/stop")
async def native_stop(request: NativeStopRequest):
    """Stop a native deployment."""
    if not _agent or not hasattr(_agent, "native_manager"):
        raise HTTPException(status_code=500, detail="Native manager not available")

    success = await _agent.native_manager.stop_process(request.process_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Process {request.process_id} not found or already stopped",
        )

    return {"status": "stopped", "process_id": request.process_id}


@router.get("/native/processes")
async def list_native_processes():
    """List all native processes."""
    if not _agent or not hasattr(_agent, "native_manager"):
        return {"processes": []}

    processes = _agent.native_manager.list_processes()
    return {
        "processes": [
            {
                "process_id": p.process_id,
                "backend": p.backend,
                "model_id": p.model_id,
                "port": p.port,
                "running": _agent.native_manager.is_running(p.process_id),
            }
            for p in processes
        ]
    }


@router.get("/native/logs/{process_id}")
async def get_native_logs(process_id: str, tail: int = 100):
    """Get logs from a native process."""
    if not _agent or not hasattr(_agent, "native_manager"):
        raise HTTPException(status_code=500, detail="Native manager not available")

    logs = _agent.native_manager.get_logs(process_id, tail)
    return {"logs": logs}


@router.get("/native/health")
async def native_health(backend: str = "ollama", port: int = 11434):
    """Check if native backend API is ready.

    This endpoint checks locally (on the worker) so it works even when
    Ollama only listens on localhost.
    """
    import httpx

    try:
        if backend == "ollama":
            # Check Ollama's OpenAI-compatible endpoint
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    # Check if there are models available
                    if data.get("data") and len(data["data"]) > 0:
                        return {"ready": True, "models": len(data["data"])}
                return {"ready": False, "reason": "No models loaded"}
        else:
            # For MLX and llama.cpp, check the /v1/models endpoint
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/v1/models")
                if response.status_code == 200:
                    return {"ready": True}
                return {"ready": False, "reason": f"Status {response.status_code}"}
    except httpx.ConnectError:
        return {"ready": False, "reason": "Connection refused"}
    except Exception as e:
        return {"ready": False, "reason": str(e)}


def _find_available_port(start: int = 8001, end: int = 9000) -> int:
    """Find an available port."""
    import socket

    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports in range {start}-{end}")
