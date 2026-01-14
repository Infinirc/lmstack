"""App schemas for API requests/responses"""

from datetime import datetime

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
    name: str | None = Field(None, description="Custom name for the app")
    use_proxy: bool = Field(
        True,
        description="Use LMStack nginx proxy (recommended) or direct worker connection",
    )


class AppResponse(BaseModel):
    """App response schema"""

    id: int
    app_type: str
    name: str
    worker_id: int
    worker_name: str | None = None
    worker_address: str | None = None
    status: str
    status_message: str | None = None
    container_id: str | None = None
    port: int | None = None
    proxy_path: str
    proxy_url: str | None = None
    use_proxy: bool = True
    access_url: str | None = None  # The URL to access the app
    api_key_id: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AppListResponse(BaseModel):
    """App list response"""

    items: list[AppResponse]
    total: int


class AvailableAppsResponse(BaseModel):
    """List of available apps that can be deployed"""

    items: list[AppDefinition]


class AppLogsResponse(BaseModel):
    """App logs response"""

    app_id: int
    logs: str
