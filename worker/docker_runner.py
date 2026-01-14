"""Docker container management for LMStack Worker - DEPRECATED.

This module is kept for backward compatibility.
New code should import from worker.docker package directly.

Example:
    from worker.docker import DockerRunner, GPUDetector, SystemDetector
    from worker.docker import ImageManager, ContainerManager
    from worker.docker.runner import get_pull_progress
"""

import warnings

warnings.warn(
    "docker_runner module is deprecated. "
    "Import from 'docker_ops' package instead: from docker_ops import DockerRunner",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all classes for backward compatibility
try:
    from docker_ops import (
        ContainerManager,
        DockerRunner,
        GPUDetector,
        ImageManager,
        SystemDetector,
        get_pull_progress,
    )
except ImportError:
    from worker.docker_ops import (
        ContainerManager,
        DockerRunner,
        GPUDetector,
        ImageManager,
        SystemDetector,
        get_pull_progress,
    )

# Legacy global - kept for backward compatibility
pull_progress: dict[str, dict] = {}

__all__ = [
    "GPUDetector",
    "SystemDetector",
    "DockerRunner",
    "ImageManager",
    "ContainerManager",
    "get_pull_progress",
    "pull_progress",
]
