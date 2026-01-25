"""Auto-Tuning and Benchmark Pydantic schemas"""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.tuning import OptimizationTarget

# ============================================================================
# LLM Configuration for Agent
# ============================================================================


class LLMConfig(BaseModel):
    """Configuration for the LLM used by the auto-tuning agent"""

    deployment_id: int | None = Field(None, description="Use a local deployment as the agent LLM")
    base_url: str | None = Field(None, description="Custom OpenAI-compatible endpoint URL")
    api_key: str | None = Field(None, description="API key for the endpoint")
    model: str | None = Field(None, description="Model name to use")


# ============================================================================
# Tuning Job Schemas
# ============================================================================


class TuningJobCreate(BaseModel):
    """Schema for creating a tuning job"""

    model_id: int = Field(..., description="ID of the model to tune")
    worker_id: int = Field(..., description="ID of the worker to use")
    optimization_target: OptimizationTarget = Field(
        default=OptimizationTarget.BALANCED, description="What to optimize for"
    )
    llm_config: LLMConfig | None = Field(
        None, description="LLM configuration for the agent (uses chat panel's selected model)"
    )


class TuningJobProgress(BaseModel):
    """Progress information for a tuning job"""

    step: int
    total_steps: int
    step_name: str
    step_description: str
    configs_tested: int = 0
    configs_total: int = 0
    current_config: dict | None = None
    best_config_so_far: dict | None = None
    best_score_so_far: float | None = None


class ConversationMessage(BaseModel):
    """A message in the agent conversation log"""

    role: str  # "user", "assistant", or "tool"
    content: str
    timestamp: str | None = None
    tool_calls: list[dict] | None = None  # For assistant messages with tool calls
    tool_call_id: str | None = None  # For tool responses
    name: str | None = None  # Tool name for tool responses


class TuningJobResponse(BaseModel):
    """Schema for tuning job response"""

    id: int
    model_id: int
    worker_id: int
    optimization_target: str
    status: str
    status_message: str | None = None
    current_step: int
    total_steps: int
    progress: TuningJobProgress | None = None
    best_config: dict | None = None
    all_results: list | None = None
    conversation_log: list[ConversationMessage] | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    # Related info
    model_name: str | None = None
    worker_name: str | None = None

    class Config:
        from_attributes = True


class TuningJobListResponse(BaseModel):
    """Schema for listing tuning jobs"""

    items: list[TuningJobResponse]
    total: int


# ============================================================================
# Benchmark Schemas
# ============================================================================


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark test"""

    engine: str = Field(..., description="Inference engine: vllm, sglang, ollama")
    quantization: str | None = Field(default=None, description="Quantization: fp16, fp8, awq, gptq")
    tensor_parallel: int = Field(default=1, description="Tensor parallelism degree")
    extra_args: dict | None = Field(default=None, description="Additional engine arguments")


class BenchmarkRequest(BaseModel):
    """Schema for running a benchmark"""

    deployment_id: int = Field(..., description="ID of the deployment to benchmark")
    test_type: str = Field(default="throughput", description="Test type: throughput, latency")
    duration_seconds: int = Field(default=60, ge=10, le=600, description="Test duration")
    input_length: int = Field(default=512, ge=1, le=32768, description="Input token length")
    output_length: int = Field(default=128, ge=1, le=8192, description="Output token length")
    concurrency: int = Field(default=1, ge=1, le=64, description="Number of concurrent requests")


class BenchmarkMetrics(BaseModel):
    """Benchmark performance metrics"""

    throughput_tps: float | None = Field(None, description="Tokens per second")
    ttft_ms: float | None = Field(None, description="Time to first token (ms)")
    tpot_ms: float | None = Field(None, description="Time per output token (ms)")
    total_latency_ms: float | None = Field(None, description="Total request latency (ms)")
    gpu_utilization: float | None = Field(None, description="GPU utilization (0-100%)")
    vram_usage_gb: float | None = Field(None, description="VRAM usage in GB")


class BenchmarkResultResponse(BaseModel):
    """Schema for benchmark result response"""

    id: int
    tuning_job_id: int | None = None
    deployment_id: int
    config: dict
    test_type: str
    test_duration_seconds: int
    input_length: int
    output_length: int
    concurrency: int
    metrics: BenchmarkMetrics
    error_message: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class BenchmarkResultListResponse(BaseModel):
    """Schema for listing benchmark results"""

    items: list[BenchmarkResultResponse]
    total: int


# ============================================================================
# Knowledge Base Schemas
# ============================================================================


class KnowledgeQuery(BaseModel):
    """Query for the knowledge base"""

    model_name: str | None = Field(default=None, description="Model name pattern to match")
    model_family: str | None = Field(default=None, description="Model family: Qwen, Llama, etc.")
    gpu_model: str | None = Field(default=None, description="GPU model pattern")
    min_vram_gb: float | None = Field(default=None, description="Minimum VRAM")
    optimization_target: OptimizationTarget = Field(
        default=OptimizationTarget.BALANCED, description="Optimization target for scoring"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return")


class KnowledgeRecord(BaseModel):
    """A knowledge base record"""

    id: int
    gpu_model: str
    gpu_count: int
    total_vram_gb: float
    model_name: str
    model_family: str
    model_params_b: float | None = None
    engine: str
    quantization: str | None = None
    tensor_parallel: int
    extra_args: dict | None = None
    throughput_tps: float
    ttft_ms: float
    tpot_ms: float
    gpu_utilization: float | None = None
    vram_usage_gb: float | None = None
    score: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class KnowledgeQueryResponse(BaseModel):
    """Response for knowledge base query"""

    items: list[KnowledgeRecord]
    total: int
    query: KnowledgeQuery


class KnowledgeSaveRequest(BaseModel):
    """Request to save a record to knowledge base"""

    benchmark_result_id: int = Field(..., description="ID of the benchmark result to save")
    model_family: str = Field(..., description="Model family for categorization")
    model_params_b: float | None = Field(default=None, description="Model parameters in billions")
