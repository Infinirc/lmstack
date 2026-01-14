"""Worker Pydantic schemas"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.worker import WorkerStatus, ConnectionType


class GPUInfo(BaseModel):
    """GPU information schema"""

    index: int
    name: str
    memory_total: int  # bytes
    memory_used: int = 0
    memory_free: int = 0
    utilization: int = 0  # percentage
    temperature: int = 0  # Celsius


class CPUInfo(BaseModel):
    """CPU information schema"""

    percent: float = 0  # CPU usage percentage
    count: int = 0  # Number of CPU cores
    freq_mhz: float = 0  # CPU frequency in MHz


class MemoryInfo(BaseModel):
    """Memory information schema"""

    total: int = 0  # bytes
    used: int = 0  # bytes
    free: int = 0  # bytes
    percent: float = 0  # percentage


class DiskInfo(BaseModel):
    """Disk information schema"""

    total: int = 0  # bytes
    used: int = 0  # bytes
    free: int = 0  # bytes
    percent: float = 0  # percentage


class SystemInfo(BaseModel):
    """System information schema (CPU, Memory, Disk)"""

    cpu: Optional[CPUInfo] = None
    memory: Optional[MemoryInfo] = None
    disk: Optional[DiskInfo] = None


class WorkerBase(BaseModel):
    """Base worker schema"""

    name: str = Field(..., min_length=1, max_length=255)
    address: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    labels: Optional[dict] = None
    connection_type: str = Field(
        default=ConnectionType.DIRECT.value, description="Connection type: direct or tailscale"
    )


class WorkerCreate(WorkerBase):
    """Schema for creating a worker"""

    gpu_info: Optional[list[GPUInfo]] = None
    system_info: Optional[SystemInfo] = None
    tailscale_ip: Optional[str] = Field(None, description="IP address in Tailscale network")
    headscale_node_id: Optional[int] = Field(None, description="Node ID in Headscale")


class WorkerUpdate(BaseModel):
    """Schema for updating a worker"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    address: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    labels: Optional[dict] = None
    status: Optional[WorkerStatus] = None
    gpu_info: Optional[list[GPUInfo]] = None
    system_info: Optional[SystemInfo] = None
    connection_type: Optional[str] = Field(None, description="Connection type: direct or tailscale")
    tailscale_ip: Optional[str] = Field(None, description="IP address in Tailscale network")
    headscale_node_id: Optional[int] = Field(None, description="Node ID in Headscale")


class WorkerResponse(WorkerBase):
    """Schema for worker response"""

    id: int
    status: str
    gpu_info: Optional[list[dict]] = None
    system_info: Optional[dict] = None
    tailscale_ip: Optional[str] = None
    headscale_node_id: Optional[int] = None
    effective_address: Optional[str] = None  # The actual address to use for connections
    created_at: datetime
    updated_at: datetime
    last_heartbeat: Optional[datetime] = None
    deployment_count: int = 0

    class Config:
        from_attributes = True


class WorkerListResponse(BaseModel):
    """Schema for worker list response"""

    items: list[WorkerResponse]
    total: int


class WorkerHeartbeat(BaseModel):
    """Schema for worker heartbeat"""

    worker_id: int
    gpu_info: Optional[list[GPUInfo]] = None
    system_info: Optional[SystemInfo] = None
    status: WorkerStatus = WorkerStatus.ONLINE
