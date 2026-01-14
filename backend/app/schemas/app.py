"""App schemas for API requests/responses"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class AppDefinition(BaseModel):
    """App definition schema"""

    type: str
    name: str
    description: str
    image: str


class AppDeploy(BaseModel):
    """Schema for deploying an app"""

    app_type: str = Field(..., description="App type (e.g., open-webui)")
    worker_id: int = Field(..., description="Worker ID to deploy on")
    name: Optional[str] = Field(None, description="Custom name for the app")
    use_proxy: bool = Field(
        True, description="Use LMStack nginx proxy (recommended) or direct worker connection"
    )


class AppResponse(BaseModel):
    """App response schema"""

    id: int
    app_type: str
    name: str
    worker_id: int
    worker_name: Optional[str] = None
    worker_address: Optional[str] = None
    status: str
    status_message: Optional[str] = None
    container_id: Optional[str] = None
    port: Optional[int] = None
    proxy_path: str
    proxy_url: Optional[str] = None
    use_proxy: bool = True
    access_url: Optional[str] = None  # The URL to access the app
    api_key_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AppListResponse(BaseModel):
    """App list response"""

    items: List[AppResponse]
    total: int


class AvailableAppsResponse(BaseModel):
    """List of available apps that can be deployed"""

    items: List[AppDefinition]


class AppLogsResponse(BaseModel):
    """App logs response"""

    app_id: int
    logs: str
