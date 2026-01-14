"""API Key Pydantic schemas"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    """Request to create an API key"""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    allowed_model_ids: Optional[List[int]] = None  # None = all models allowed
    monthly_token_limit: Optional[int] = None  # None = unlimited
    expires_in_days: Optional[int] = None  # None = never expires


class ApiKeyUpdate(BaseModel):
    """Request to update an API key"""

    name: Optional[str] = None
    description: Optional[str] = None
    allowed_model_ids: Optional[List[int]] = None
    monthly_token_limit: Optional[int] = None


class ApiKeyResponse(BaseModel):
    """API key response (without secret)"""

    id: int
    name: str
    description: Optional[str]
    access_key: str
    allowed_model_ids: Optional[List[int]]
    monthly_token_limit: Optional[int]
    expires_at: Optional[datetime]
    created_at: datetime
    last_used_at: Optional[datetime]

    class Config:
        from_attributes = True


class ApiKeyCreateResponse(ApiKeyResponse):
    """Response when creating an API key (includes full key value)"""

    api_key: str  # Full key: lmsk_{access_key}_{secret_key}


class ApiKeyListResponse(BaseModel):
    """Paginated API key list"""

    items: List[ApiKeyResponse]
    total: int
