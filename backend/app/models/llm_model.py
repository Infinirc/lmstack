"""LLM Model database model"""

from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class BackendType(str, Enum):
    """Inference backend type"""

    VLLM = "vllm"
    SGLANG = "sglang"
    OLLAMA = "ollama"


class ModelSource(str, Enum):
    """Model source type"""

    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class LLMModel(Base):
    """LLM Model definition - represents a model that can be deployed"""

    __tablename__ = "llm_models"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    model_id: Mapped[str] = mapped_column(
        String(512), nullable=False
    )  # HuggingFace model ID or Ollama model name
    source: Mapped[str] = mapped_column(
        String(50), default=ModelSource.HUGGINGFACE.value
    )  # huggingface or ollama
    # Keep backend for backwards compatibility with existing databases
    backend: Mapped[str | None] = mapped_column(
        String(50), default=BackendType.VLLM.value, nullable=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Default inference parameters
    default_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Docker image for this model (optional override)
    docker_image: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    deployments: Mapped[list["Deployment"]] = relationship(
        "Deployment", back_populates="model", cascade="all, delete-orphan"
    )
    usages: Mapped[list["Usage"]] = relationship("Usage", back_populates="model")

    def __repr__(self) -> str:
        return f"<LLMModel(id={self.id}, name='{self.name}', source='{self.source}')>"
