"""API Key Pydantic schemas"""

from datetime import datetime

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    """Request to create an API key"""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    allowed_model_ids: list[int] | None = None  # None = all models allowed
    monthly_token_limit: int | None = None  # None = unlimited
    expires_in_days: int | None = None  # None = never expires


class ApiKeyUpdate(BaseModel):
    """Request to update an API key"""

    name: str | None = None
    description: str | None = None
    allowed_model_ids: list[int] | None = None
    monthly_token_limit: int | None = None


class ApiKeyResponse(BaseModel):
    """API key response (without secret)"""

    id: int
    name: str
    description: str | None
    access_key: str
    allowed_model_ids: list[int] | None
    monthly_token_limit: int | None
    expires_at: datetime | None
    created_at: datetime
    last_used_at: datetime | None

    class Config:
        from_attributes = True


class ApiKeyCreateResponse(ApiKeyResponse):
    """Response when creating an API key (includes full key value)"""

    api_key: str  # Full key: lmsk_{access_key}_{secret_key}


class ApiKeyListResponse(BaseModel):
    """Paginated API key list"""

    items: list[ApiKeyResponse]
    total: int
