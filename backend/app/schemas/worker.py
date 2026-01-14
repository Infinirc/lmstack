"""Worker Pydantic schemas"""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.worker import ConnectionType, WorkerStatus


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

    cpu: CPUInfo | None = None
    memory: MemoryInfo | None = None
    disk: DiskInfo | None = None


class WorkerBase(BaseModel):
    """Base worker schema"""

    name: str = Field(..., min_length=1, max_length=255)
    address: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    labels: dict | None = None
    connection_type: str = Field(
        default=ConnectionType.DIRECT.value,
        description="Connection type: direct or tailscale",
    )


class WorkerCreate(WorkerBase):
    """Schema for creating a worker"""

    gpu_info: list[GPUInfo] | None = None
    system_info: SystemInfo | None = None
    tailscale_ip: str | None = Field(None, description="IP address in Tailscale network")
    headscale_node_id: int | None = Field(None, description="Node ID in Headscale")


class WorkerUpdate(BaseModel):
    """Schema for updating a worker"""

    name: str | None = Field(None, min_length=1, max_length=255)
    address: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    labels: dict | None = None
    status: WorkerStatus | None = None
    gpu_info: list[GPUInfo] | None = None
    system_info: SystemInfo | None = None
    connection_type: str | None = Field(None, description="Connection type: direct or tailscale")
    tailscale_ip: str | None = Field(None, description="IP address in Tailscale network")
    headscale_node_id: int | None = Field(None, description="Node ID in Headscale")


class WorkerResponse(WorkerBase):
    """Schema for worker response"""

    id: int
    status: str
    gpu_info: list[dict] | None = None
    system_info: dict | None = None
    tailscale_ip: str | None = None
    headscale_node_id: int | None = None
    effective_address: str | None = None  # The actual address to use for connections
    created_at: datetime
    updated_at: datetime
    last_heartbeat: datetime | None = None
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
    gpu_info: list[GPUInfo] | None = None
    system_info: SystemInfo | None = None
    status: WorkerStatus = WorkerStatus.ONLINE
