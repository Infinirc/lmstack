"""Worker database model"""

from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class WorkerStatus(str, Enum):
    """Worker status enum"""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"


class ConnectionType(str, Enum):
    """Worker connection type"""

    DIRECT = "direct"  # Direct IP connection (internal network)
    TAILSCALE = "tailscale"  # Via Tailscale/Headscale VPN


class OSType(str, Enum):
    """Worker operating system type"""

    LINUX = "linux"
    DARWIN = "darwin"  # macOS
    WINDOWS = "windows"


class GPUType(str, Enum):
    """GPU type for inference"""

    NVIDIA = "nvidia"
    APPLE_SILICON = "apple_silicon"  # M1/M2/M3/M4
    AMD = "amd"
    NONE = "none"


class Worker(Base):
    """Worker node model - represents a GPU node that can run LLM inference"""

    __tablename__ = "workers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    address: Mapped[str] = mapped_column(String(255), nullable=False)  # IP:Port (direct connection)
    status: Mapped[str] = mapped_column(String(50), default=WorkerStatus.OFFLINE.value)

    # Connection type and Tailscale support
    connection_type: Mapped[str] = mapped_column(String(50), default=ConnectionType.DIRECT.value)
    tailscale_ip: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # IP in Tailscale network
    headscale_node_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # Node ID in Headscale

    # System detection
    os_type: Mapped[str] = mapped_column(String(50), default=OSType.LINUX.value)
    gpu_type: Mapped[str] = mapped_column(String(50), default=GPUType.NVIDIA.value)
    capabilities: Mapped[dict | None] = mapped_column(
        JSON, nullable=True
    )  # Available backends/tools

    gpu_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    system_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    labels: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    deployments: Mapped[list["Deployment"]] = relationship(
        "Deployment", back_populates="worker", cascade="all, delete-orphan"
    )

    @property
    def effective_address(self) -> str:
        """Get the effective address to use for connections.

        Returns tailscale_ip:port if connection_type is tailscale,
        otherwise returns the direct address.
        """
        if self.connection_type == ConnectionType.TAILSCALE.value and self.tailscale_ip:
            # Extract port from direct address and use with tailscale IP
            port = self.address.split(":")[-1] if ":" in self.address else "8000"
            return f"{self.tailscale_ip}:{port}"
        return self.address

    @property
    def supports_docker(self) -> bool:
        """Check if this worker supports Docker deployments."""
        # macOS without Docker support uses native process management
        caps = self.capabilities or {}
        return caps.get("docker", self.os_type == OSType.LINUX.value)

    @property
    def is_mac(self) -> bool:
        """Check if this is a macOS worker."""
        return self.os_type == OSType.DARWIN.value

    @property
    def available_backends(self) -> list[str]:
        """Get list of available backends for this worker."""
        caps = self.capabilities or {}
        backends = []

        if self.supports_docker:
            # Docker-based backends
            if self.gpu_type == GPUType.NVIDIA.value:
                backends.extend(["vllm", "sglang", "ollama"])
            else:
                backends.append("ollama")
        else:
            # Native backends (Mac)
            if caps.get("ollama"):
                backends.append("ollama")
            if caps.get("mlx"):
                backends.append("mlx")
            if caps.get("llama_cpp"):
                backends.append("llama_cpp")

        return backends

    def __repr__(self) -> str:
        return f"<Worker(id={self.id}, name='{self.name}', status='{self.status}', os='{self.os_type}')>"
