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


class Worker(Base):
    """Worker node model - represents a GPU node that can run LLM inference"""

    __tablename__ = "workers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    address: Mapped[str] = mapped_column(
        String(255), nullable=False
    )  # IP:Port (direct connection)
    status: Mapped[str] = mapped_column(String(50), default=WorkerStatus.OFFLINE.value)

    # Connection type and Tailscale support
    connection_type: Mapped[str] = mapped_column(
        String(50), default=ConnectionType.DIRECT.value
    )
    tailscale_ip: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # IP in Tailscale network
    headscale_node_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # Node ID in Headscale

    gpu_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    system_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    labels: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_heartbeat: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

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

    def __repr__(self) -> str:
        return f"<Worker(id={self.id}, name='{self.name}', status='{self.status}', connection='{self.connection_type}')>"
