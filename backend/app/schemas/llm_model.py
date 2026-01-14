"""LLM Model Pydantic schemas"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.llm_model import ModelSource


class LLMModelBase(BaseModel):
    """Base LLM model schema"""
    name: str = Field(..., min_length=1, max_length=255)
    model_id: str = Field(..., min_length=1, max_length=512)
    source: ModelSource = ModelSource.HUGGINGFACE
    description: Optional[str] = None
    default_params: Optional[dict] = None
    docker_image: Optional[str] = None


class LLMModelCreate(LLMModelBase):
    """Schema for creating an LLM model"""
    pass


class LLMModelUpdate(BaseModel):
    """Schema for updating an LLM model"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    model_id: Optional[str] = Field(None, min_length=1, max_length=512)
    source: Optional[ModelSource] = None
    description: Optional[str] = None
    default_params: Optional[dict] = None
    docker_image: Optional[str] = None


class LLMModelResponse(LLMModelBase):
    """Schema for LLM model response"""
    id: int
    created_at: datetime
    updated_at: datetime
    deployment_count: int = 0

    class Config:
        from_attributes = True


class LLMModelListResponse(BaseModel):
    """Schema for LLM model list response"""
    items: list[LLMModelResponse]
    total: int
