"""Deployment database model"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.llm_model import BackendType

if TYPE_CHECKING:
    from app.models.llm_model import LLMModel
    from app.models.worker import Worker


class DeploymentStatus(str, Enum):
    """Deployment status enum"""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class Deployment(Base):
    """Deployment instance - represents a running model on a worker"""

    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Foreign keys
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("llm_models.id"), nullable=False
    )
    worker_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workers.id"), nullable=False
    )

    # Inference backend (vllm, sglang, ollama)
    backend: Mapped[str] = mapped_column(
        String(50), default=BackendType.VLLM.value, nullable=False
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(50), default=DeploymentStatus.PENDING.value
    )
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Container info
    container_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    port: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Configuration
    gpu_indexes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    extra_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    model: Mapped["LLMModel"] = relationship("LLMModel", back_populates="deployments")
    worker: Mapped["Worker"] = relationship("Worker", back_populates="deployments")
    usages: Mapped[list["Usage"]] = relationship("Usage", back_populates="deployment")

    def __repr__(self) -> str:
        return f"<Deployment(id={self.id}, name='{self.name}', status='{self.status}')>"
