"""API Key model for authentication and access control"""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


class ApiKey(Base):
    """API Key for external access with optional model restrictions"""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)

    # Key components (prefix_accesskey_secretkey format)
    access_key = Column(String(32), unique=True, nullable=False, index=True)
    hashed_secret = Column(String(128), nullable=False)

    # Optional restrictions
    allowed_model_ids = Column(JSON, nullable=True)  # List of model IDs, null = all
    monthly_token_limit = Column(
        Integer, nullable=True
    )  # Monthly token limit, null = unlimited
    expires_at = Column(DateTime, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    usages = relationship("Usage", back_populates="api_key")

    __table_args__ = (Index("ix_api_keys_access_key", "access_key"),)


class Usage(Base):
    """Usage tracking per API key, per model, per day"""

    __tablename__ = "usages"

    id = Column(Integer, primary_key=True, index=True)

    # Foreign keys
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=True)
    model_id = Column(Integer, ForeignKey("llm_models.id"), nullable=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"), nullable=True)

    # Aggregation date
    date = Column(DateTime, nullable=False, index=True)

    # Counters
    request_count = Column(Integer, default=0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    api_key = relationship("ApiKey", back_populates="usages")
    model = relationship("LLMModel", back_populates="usages")
    deployment = relationship("Deployment", back_populates="usages")

    __table_args__ = (
        Index("ix_usages_date_api_key", "date", "api_key_id"),
        Index("ix_usages_date_model", "date", "model_id"),
    )
