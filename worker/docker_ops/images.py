"""Docker image management for LMStack Worker.

Provides methods for listing, inspecting, pulling, building,
and removing Docker images on the worker node.
"""

import io
import logging
import tarfile
from typing import Any, Callable, Optional

import docker
from docker.errors import APIError, ImageNotFound

logger = logging.getLogger(__name__)


class ImageManager:
    """Docker image management operations."""

    def __init__(self, client: docker.DockerClient):
        self.client = client

    def list_images(
        self,
        repository: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List all Docker images on this node.

        Args:
            repository: Optional filter by repository name (supports partial match)

        Returns:
            List of image dictionaries with standardized format
        """
        images = []

        for image in self.client.images.list(all=False):
            # Skip images without tags (intermediate layers)
            if not image.tags:
                continue

            for tag in image.tags:
                # Parse repository and tag
                if ":" in tag:
                    repo, img_tag = tag.rsplit(":", 1)
                else:
                    repo, img_tag = tag, "latest"

                # Apply repository filter
                if repository and repository.lower() not in repo.lower():
                    continue

                # Get image creation time
                created_at = image.attrs.get("Created", "")
                if created_at and "." in created_at:
                    # Normalize timestamp format (remove nanoseconds)
                    created_at = created_at.split(".")[0] + "Z"

                images.append(
                    {
                        "id": image.short_id.replace("sha256:", ""),
                        "repository": repo,
                        "tag": img_tag,
                        "full_name": tag,
                        "size": image.attrs.get("Size", 0),
                        "created_at": created_at,
                        "digest": image.attrs.get("RepoDigests", [None])[0],
                        "labels": image.labels or {},
                    }
                )

        return images

    def get_image_detail(self, image_id: str) -> dict[str, Any]:
        """Get detailed information about an image.

        Args:
            image_id: Image ID or name:tag

        Returns:
            Detailed image information including layers and config

        Raises:
            ImageNotFound: If image doesn't exist
        """
        image = self.client.images.get(image_id)

        # Get image history for layer info
        history = image.history()
        layers = []
        for layer in history:
            if layer.get("Size", 0) > 0:  # Skip empty layers
                layers.append(
                    {
                        "digest": layer.get("Id", "")[:20] if layer.get("Id") else "",
                        "size": layer.get("Size", 0),
                        "instruction": layer.get("CreatedBy", ""),
                    }
                )

        # Parse config
        config = image.attrs.get("Config", {})

        # Get first tag info
        tag = image.tags[0] if image.tags else image_id
        if ":" in tag:
            repo, img_tag = tag.rsplit(":", 1)
        else:
            repo, img_tag = tag, "latest"

        return {
            "id": image.short_id.replace("sha256:", ""),
            "repository": repo,
            "tag": img_tag,
            "full_name": tag,
            "size": image.attrs.get("Size", 0),
            "created_at": image.attrs.get("Created", ""),
            "digest": image.attrs.get("RepoDigests", [None])[0],
            "labels": image.labels or {},
            "layers": layers,
            "config": {
                "env": config.get("Env"),
                "cmd": config.get("Cmd"),
                "entrypoint": config.get("Entrypoint"),
                "working_dir": config.get("WorkingDir"),
                "exposed_ports": list(config.get("ExposedPorts", {}).keys()),
                "volumes": (
                    list(config.get("Volumes", {}).keys())
                    if config.get("Volumes")
                    else None
                ),
            },
        }

    def pull_image(
        self,
        image: str,
        auth_config: Optional[dict[str, str]] = None,
        progress_callback: Optional[Callable[[int, dict], None]] = None,
    ) -> dict[str, Any]:
        """Pull an image from a registry.

        Args:
            image: Image reference (e.g., "nginx:latest")
            auth_config: Optional registry authentication
            progress_callback: Optional callback for progress updates

        Returns:
            Pulled image information
        """
        logger.info(f"Pulling image: {image}")

        # Prepare auth if provided
        auth = None
        if auth_config:
            auth = {
                "username": auth_config.get("username"),
                "password": auth_config.get("password"),
                "serveraddress": auth_config.get("server_address"),
            }

        # Pull with progress tracking
        layers_progress: dict[str, dict] = {}

        for line in self.client.api.pull(
            image, stream=True, decode=True, auth_config=auth
        ):
            if progress_callback and "id" in line:
                layer_id = line["id"]
                detail = line.get("progressDetail", {})
                layers_progress[layer_id] = {
                    "status": line.get("status", ""),
                    "current": detail.get("current", 0),
                    "total": detail.get("total", 0),
                }

                # Calculate overall progress
                total_size = sum(lp.get("total", 0) for lp in layers_progress.values())
                downloaded = sum(
                    lp.get("current", 0) for lp in layers_progress.values()
                )
                progress = int((downloaded / total_size) * 100) if total_size > 0 else 0

                progress_callback(progress, layers_progress)

        # Get the pulled image info
        pulled_image = self.client.images.get(image)
        return {
            "id": pulled_image.short_id.replace("sha256:", ""),
            "tags": pulled_image.tags,
            "size": pulled_image.attrs.get("Size", 0),
        }

    def build_image(
        self,
        dockerfile: str,
        tag: str,
        build_args: Optional[dict[str, str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        """Build an image from a Dockerfile.

        Args:
            dockerfile: Dockerfile content as string
            tag: Tag for the built image
            build_args: Optional build arguments
            progress_callback: Optional callback for progress updates

        Returns:
            Built image information
        """
        logger.info(f"Building image with tag: {tag}")

        # Create a tar archive with the Dockerfile
        dockerfile_bytes = dockerfile.encode("utf-8")

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            dockerfile_info = tarfile.TarInfo(name="Dockerfile")
            dockerfile_info.size = len(dockerfile_bytes)
            tar.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))

        tar_buffer.seek(0)

        # Build the image
        image, build_logs = self.client.images.build(
            fileobj=tar_buffer,
            custom_context=True,
            tag=tag,
            buildargs=build_args or {},
            rm=True,  # Remove intermediate containers
            forcerm=True,  # Always remove intermediate containers
        )

        # Log build output
        for log in build_logs:
            if "stream" in log:
                logger.debug(log["stream"].strip())
            if progress_callback and "stream" in log:
                progress_callback(log["stream"])

        return {
            "id": image.short_id.replace("sha256:", ""),
            "tags": image.tags,
            "size": image.attrs.get("Size", 0),
        }

    def remove_image(self, image_id: str, force: bool = False) -> None:
        """Remove an image.

        Args:
            image_id: Image ID or name:tag
            force: Force removal even if in use

        Raises:
            ImageNotFound: If image doesn't exist
            APIError: If image cannot be removed
        """
        logger.info(f"Removing image: {image_id}")
        try:
            self.client.images.remove(image_id, force=force)
        except ImageNotFound:
            logger.warning(f"Image {image_id} not found")
            raise
        except APIError as e:
            logger.error(f"Failed to remove image {image_id}: {e}")
            raise
