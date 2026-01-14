"""Dashboard Pydantic schemas"""

from datetime import date

from pydantic import BaseModel


class ResourceCounts(BaseModel):
    """Resource count summary"""

    worker_count: int
    worker_online_count: int
    gpu_count: int
    model_count: int
    deployment_count: int
    deployment_running_count: int


class GPUSummary(BaseModel):
    """GPU summary across all workers"""

    total_memory_gb: float
    used_memory_gb: float
    utilization_avg: float
    temperature_avg: float = 0  # Average temperature in Celsius
    temperature_max: float = 0  # Maximum temperature in Celsius


class UsagePoint(BaseModel):
    """Single point in usage time series"""

    date: date
    value: int


class UsageSummary(BaseModel):
    """Usage statistics summary"""

    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    request_history: list[UsagePoint]  # Last 30 days
    token_history: list[UsagePoint]  # Last 30 days


class TopModel(BaseModel):
    """Top model by usage"""

    model_id: int
    model_name: str
    request_count: int
    token_count: int


class TopApiKey(BaseModel):
    """Top API key by usage"""

    api_key_id: int
    api_key_name: str
    request_count: int
    token_count: int


class DashboardResponse(BaseModel):
    """Complete dashboard response"""

    resources: ResourceCounts
    gpu_summary: GPUSummary
    usage: UsageSummary
    top_models: list[TopModel]
    top_api_keys: list[TopApiKey]
