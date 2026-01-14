"""Headscale API routes

Provides endpoints for managing the Headscale VPN server.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.auth import require_admin
from app.services.headscale_manager import LMSTACK_USER, get_headscale_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================


class HeadscaleStartRequest(BaseModel):
    """Request to start Headscale."""

    server_url: str | None = Field(
        None, description="Server URL (auto-detected if not provided)"
    )
    http_port: int = Field(8080, description="HTTP port for Headscale")
    grpc_port: int = Field(50443, description="gRPC port for Headscale")


class HeadscaleStatusResponse(BaseModel):
    """Headscale status response."""

    enabled: bool
    running: bool
    container_status: str | None = None
    server_url: str | None = None
    nodes_count: int | None = None
    online_nodes: int | None = None


class PreauthKeyRequest(BaseModel):
    """Request to create a preauth key."""

    reusable: bool = Field(True, description="Allow multiple uses")
    ephemeral: bool = Field(False, description="Nodes using this key will be ephemeral")
    expiration: str = Field("720h", description="Key expiration (e.g., 24h, 720h)")
    tags: list[str] | None = Field(None, description="Tags for nodes using this key")


class PreauthKeyResponse(BaseModel):
    """Preauth key response."""

    key: str
    join_command: str


class HeadscaleNodeResponse(BaseModel):
    """Node in Headscale."""

    id: int
    name: str
    given_name: str | None = None
    ip_addresses: list[str]
    ipv4: str | None = None
    online: bool
    last_seen: str | None = None
    created_at: str | None = None


class HeadscaleNodesResponse(BaseModel):
    """List of Headscale nodes."""

    items: list[HeadscaleNodeResponse]
    total: int


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/status", response_model=HeadscaleStatusResponse)
async def get_headscale_status(
    _: dict = Depends(require_admin),
):
    """Get Headscale status."""
    manager = get_headscale_manager()
    try:
        status = await manager.get_status()
        return HeadscaleStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get Headscale status: {e}")
        return HeadscaleStatusResponse(enabled=False, running=False)


@router.post("/start", response_model=HeadscaleStatusResponse)
async def start_headscale(
    request: Request,
    data: HeadscaleStartRequest,
    _: dict = Depends(require_admin),
):
    """Start Headscale server."""
    manager = get_headscale_manager()

    # Auto-detect server URL if not provided
    server_url = data.server_url
    if not server_url:
        host = request.headers.get("host", "localhost:8000")
        host_ip = host.split(":")[0]
        if host_ip in ("localhost", "127.0.0.1"):
            # Try to get actual IP
            import socket

            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                host_ip = s.getsockname()[0]
                s.close()
            except OSError:
                pass  # Keep localhost if we can't determine actual IP
        server_url = f"http://{host_ip}"

    try:
        success = await manager.start(
            server_url=server_url,
            http_port=data.http_port,
            grpc_port=data.grpc_port,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start Headscale")

        status = await manager.get_status()
        return HeadscaleStatusResponse(**status)

    except Exception as e:
        logger.exception(f"Failed to start Headscale: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_headscale(
    _: dict = Depends(require_admin),
):
    """Stop Headscale server."""
    manager = get_headscale_manager()
    try:
        await manager.stop()
        return {"message": "Headscale stopped"}
    except Exception as e:
        logger.exception(f"Failed to stop Headscale: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preauth-key", response_model=PreauthKeyResponse)
async def create_preauth_key(
    data: PreauthKeyRequest,
    _: dict = Depends(require_admin),
):
    """Create a pre-authentication key for workers to join."""
    manager = get_headscale_manager()

    if not await manager.is_running():
        raise HTTPException(status_code=400, detail="Headscale is not running")

    try:
        key = await manager.create_preauth_key(
            user=LMSTACK_USER,
            reusable=data.reusable,
            ephemeral=data.ephemeral,
            expiration=data.expiration,
            tags=data.tags,
        )

        if not key:
            raise HTTPException(status_code=500, detail="Failed to create preauth key")

        join_command = manager.get_join_command(key)

        return PreauthKeyResponse(key=key, join_command=join_command)

    except Exception as e:
        logger.exception(f"Failed to create preauth key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes", response_model=HeadscaleNodesResponse)
async def list_headscale_nodes(
    _: dict = Depends(require_admin),
):
    """List all nodes in Headscale."""
    manager = get_headscale_manager()

    if not await manager.is_running():
        return HeadscaleNodesResponse(items=[], total=0)

    try:
        nodes = await manager.list_nodes()
        items = [
            HeadscaleNodeResponse(
                id=node.id,
                name=node.name,
                given_name=node.given_name,
                ip_addresses=node.ip_addresses,
                ipv4=node.ipv4,
                online=node.online,
                last_seen=node.last_seen,
                created_at=node.created_at,
            )
            for node in nodes
        ]
        return HeadscaleNodesResponse(items=items, total=len(items))

    except Exception as e:
        logger.exception(f"Failed to list nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/nodes/{node_id}")
async def delete_headscale_node(
    node_id: int,
    _: dict = Depends(require_admin),
):
    """Delete a node from Headscale."""
    manager = get_headscale_manager()

    if not await manager.is_running():
        raise HTTPException(status_code=400, detail="Headscale is not running")

    try:
        success = await manager.delete_node(node_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete node")
        return {"message": f"Node {node_id} deleted"}

    except Exception as e:
        logger.exception(f"Failed to delete node: {e}")
        raise HTTPException(status_code=500, detail=str(e))
