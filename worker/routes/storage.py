"""Storage management routes for LMStack Worker.

Contains endpoints for managing Docker disk usage, volumes, and pruning.
"""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/storage", tags=["storage"])

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


@router.get("/disk-usage")
async def get_disk_usage():
    """Get Docker disk usage (docker system df)."""
    agent = get_agent()

    try:
        df = agent.docker.client.df()

        # Calculate totals
        images_size = sum(img.get("Size", 0) for img in df.get("Images", []))
        images_count = len(df.get("Images", []))

        containers_size = sum(c.get("SizeRw", 0) or 0 for c in df.get("Containers", []))
        containers_count = len(df.get("Containers", []))

        volumes_size = sum(
            v.get("UsageData", {}).get("Size", 0) or 0 for v in df.get("Volumes", [])
        )
        volumes_count = len(df.get("Volumes", []))

        build_cache_size = sum(bc.get("Size", 0) for bc in df.get("BuildCache", []))
        build_cache_count = len(df.get("BuildCache", []))

        # Calculate reclaimable
        images_reclaimable = sum(
            img.get("Size", 0)
            for img in df.get("Images", [])
            if img.get("Containers", 0) == 0
        )
        containers_reclaimable = sum(
            c.get("SizeRw", 0) or 0
            for c in df.get("Containers", [])
            if c.get("State") != "running"
        )
        volumes_reclaimable = sum(
            v.get("UsageData", {}).get("Size", 0) or 0
            for v in df.get("Volumes", [])
            if v.get("UsageData", {}).get("RefCount", 0) == 0
        )
        build_cache_reclaimable = sum(
            bc.get("Size", 0)
            for bc in df.get("BuildCache", [])
            if not bc.get("InUse", False)
        )

        return {
            "images": {
                "count": images_count,
                "size": images_size,
                "reclaimable": images_reclaimable,
            },
            "containers": {
                "count": containers_count,
                "size": containers_size,
                "reclaimable": containers_reclaimable,
            },
            "volumes": {
                "count": volumes_count,
                "size": volumes_size,
                "reclaimable": volumes_reclaimable,
            },
            "build_cache": {
                "count": build_cache_count,
                "size": build_cache_size,
                "reclaimable": build_cache_reclaimable,
            },
            "total_size": images_size
            + containers_size
            + volumes_size
            + build_cache_size,
            "total_reclaimable": (
                images_reclaimable
                + containers_reclaimable
                + volumes_reclaimable
                + build_cache_reclaimable
            ),
        }
    except Exception as e:
        logger.error(f"Failed to get disk usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volumes")
async def list_volumes():
    """List all Docker volumes."""
    agent = get_agent()

    try:
        volumes = agent.docker.client.volumes.list()
        result = []
        for vol in volumes:
            attrs = vol.attrs
            result.append(
                {
                    "name": vol.name,
                    "driver": attrs.get("Driver") or "local",
                    "mountpoint": attrs.get("Mountpoint") or "",
                    "created_at": attrs.get("CreatedAt") or "",
                    "labels": attrs.get("Labels") or {},
                    "scope": attrs.get("Scope") or "local",
                    "options": attrs.get("Options") or {},
                }
            )
        logger.info(f"Found {len(result)} volumes")
        return {"items": result, "total": len(result)}
    except Exception as e:
        logger.error(f"Failed to list volumes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/volumes/{volume_name}")
async def delete_volume(volume_name: str, force: bool = False):
    """Delete a Docker volume."""
    agent = get_agent()

    try:
        volume = agent.docker.client.volumes.get(volume_name)
        volume.remove(force=force)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Failed to delete volume {volume_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prune")
async def prune_storage(
    images: bool = True,
    containers: bool = True,
    volumes: bool = False,
    build_cache: bool = True,
):
    """Prune unused Docker resources."""
    agent = get_agent()

    try:
        result = {
            "images_deleted": 0,
            "containers_deleted": 0,
            "volumes_deleted": 0,
            "build_cache_deleted": 0,
            "space_reclaimed": 0,
        }

        if containers:
            prune_result = agent.docker.client.containers.prune()
            deleted = prune_result.get("ContainersDeleted") or []
            result["containers_deleted"] = len(deleted)
            result["space_reclaimed"] += prune_result.get("SpaceReclaimed", 0)

        if images:
            prune_result = agent.docker.client.images.prune(filters={"dangling": False})
            deleted = prune_result.get("ImagesDeleted") or []
            result["images_deleted"] = len(deleted)
            result["space_reclaimed"] += prune_result.get("SpaceReclaimed", 0)

        if volumes:
            prune_result = agent.docker.client.volumes.prune()
            deleted = prune_result.get("VolumesDeleted") or []
            result["volumes_deleted"] = len(deleted)
            result["space_reclaimed"] += prune_result.get("SpaceReclaimed", 0)

        if build_cache:
            # Build cache prune via API
            try:
                prune_result = agent.docker.client.api.prune_builds()
                result["build_cache_deleted"] = (
                    prune_result.get("CachesDeleted", 0) or 0
                )
                result["space_reclaimed"] += prune_result.get("SpaceReclaimed", 0)
            except AttributeError:
                # Build cache prune may not be available in older Docker versions
                logger.debug("Build cache prune not available")

        return result
    except Exception as e:
        logger.error(f"Failed to prune storage: {e}")
        raise HTTPException(status_code=500, detail=str(e))
