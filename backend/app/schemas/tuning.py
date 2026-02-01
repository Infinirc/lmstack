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


class TuningParameters(BaseModel):
    """Parameters to test during tuning"""

    tensor_parallel_size: list[int] = Field(
        default=[1], description="Tensor parallel sizes to test"
    )
    gpu_memory_utilization: list[float] = Field(
        default=[0.85, 0.90], description="GPU memory utilization values to test (0.0-1.0)"
    )
    max_model_len: list[int] = Field(default=[4096], description="Max model lengths to test")
    max_num_seqs: list[int] | None = Field(
        default=None, description="Max concurrent sequences to test"
    )


class BenchmarkSettings(BaseModel):
    """Benchmark test settings"""

    duration_seconds: int = Field(default=60, ge=10, le=300, description="Test duration per config")
    input_length: int = Field(default=512, ge=64, le=8192, description="Input token length")
    output_length: int = Field(default=128, ge=16, le=2048, description="Output token length")
    concurrency: list[int] = Field(default=[1, 4], description="Concurrency levels to test")


class TuningConfig(BaseModel):
    """Full tuning configuration"""

    engines: list[str] = Field(
        default=["vllm"], description="Inference engines to test: vllm, sglang, ollama"
    )
    parameters: TuningParameters = Field(default_factory=TuningParameters)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)


class TuningJobCreate(BaseModel):
    """Schema for creating a tuning job"""

    model_id: int = Field(..., description="ID of the model to tune")
    worker_id: int = Field(..., description="ID of the worker to use")
    optimization_target: OptimizationTarget = Field(
        default=OptimizationTarget.THROUGHPUT, description="What to optimize for"
    )
    tuning_config: TuningConfig = Field(
        default_factory=TuningConfig, description="Tuning configuration"
    )
    llm_config: LLMConfig | None = Field(None, description="LLM configuration for the agent")


class TuningJobProgress(BaseModel):
    """Progress information for a tuning job"""

    step: int
    total_steps: int
    step_name: str
    step_description: str | None = None
    configs_tested: int = 0
    configs_total: int = 0
    current_config: dict | None = None
    best_config_so_far: dict | None = None
    best_score_so_far: float | None = None
    # Bayesian optimization specific fields
    completed_trials: int | None = None
    successful_trials: int | None = None
    deployment_status: str | None = None
    deployment_message: str | None = None
    elapsed_seconds: int | None = None


class ConversationMessage(BaseModel):
    """A message in the agent conversation log"""

    role: str  # "user", "assistant", or "tool"
    content: str
    timestamp: str | None = None
    tool_calls: list[dict] | None = None  # For assistant messages with tool calls
    tool_call_id: str | None = None  # For tool responses
    name: str | None = None  # Tool name for tool responses


class TuningLogEntry(BaseModel):
    """A single log entry"""

    timestamp: str
    level: str
    message: str


class TuningJobResponse(BaseModel):
    """Schema for tuning job response"""

    id: int
    model_id: int
    worker_id: int
    optimization_target: str
    tuning_config: TuningConfig | None = None
    status: str
    status_message: str | None = None
    current_step: int
    total_steps: int
    progress: TuningJobProgress | None = None
    best_config: dict | None = None
    all_results: list | None = None
    logs: list[TuningLogEntry] | None = None
    conversation_id: int | None = None
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


# ============================================================================
# Comprehensive Benchmark Schemas
# ============================================================================


class LatencyPercentiles(BaseModel):
    """Latency metrics with percentiles"""

    mean: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class ComprehensiveBenchmarkMetrics(BaseModel):
    """Comprehensive benchmark metrics with percentiles"""

    # Latency metrics with percentiles
    ttft: LatencyPercentiles = Field(default_factory=LatencyPercentiles)
    itl: LatencyPercentiles = Field(default_factory=LatencyPercentiles)
    tpot: LatencyPercentiles = Field(default_factory=LatencyPercentiles)
    e2e_latency: LatencyPercentiles = Field(default_factory=LatencyPercentiles)

    # Throughput metrics
    throughput_tps: float = Field(0.0, description="Total tokens per second")
    throughput_rps: float = Field(0.0, description="Requests per second")
    output_tps: float = Field(0.0, description="Output tokens per second")

    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0

    # Token statistics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0

    # Timing
    total_duration_seconds: float = 0.0
    concurrency: int = 0


class ComprehensiveBenchmarkRequest(BaseModel):
    """Request for running a comprehensive benchmark"""

    deployment_id: int = Field(..., description="ID of the deployment to benchmark")
    concurrency: int = Field(default=10, ge=1, le=128, description="Concurrent requests")
    num_requests: int = Field(default=50, ge=10, le=1000, description="Total requests")
    warmup_requests: int = Field(default=5, ge=0, le=20, description="Warmup requests")
    prompt_tokens: int = Field(default=256, ge=32, le=8192, description="Approximate input tokens")
    output_tokens: int = Field(default=128, ge=16, le=2048, description="Max output tokens")
    custom_prompt: str | None = Field(default=None, description="Custom prompt to use")


class ComprehensiveBenchmarkResponse(BaseModel):
    """Response for comprehensive benchmark"""

    metrics: ComprehensiveBenchmarkMetrics
    config: dict
    error: str | None = None
    started_at: float
    completed_at: float
    duration_seconds: float


class SaturationDetectionRequest(BaseModel):
    """Request for saturation detection"""

    deployment_id: int = Field(..., description="ID of the deployment to test")
    start_concurrency: int = Field(default=1, ge=1, description="Starting concurrency")
    max_concurrency: int = Field(
        default=64, ge=1, le=256, description="Maximum concurrency to test"
    )
    requests_per_level: int = Field(default=20, ge=10, le=100, description="Requests per level")
    use_exponential: bool = Field(default=True, description="Use exponential stepping")
    step_size: int = Field(default=2, ge=1, description="Linear step size")
    step_multiplier: float = Field(
        default=1.5, ge=1.1, le=3.0, description="Exponential multiplier"
    )


class ConcurrencyLevelResult(BaseModel):
    """Result for a single concurrency level"""

    concurrency: int
    throughput_tps: float
    avg_latency_ms: float
    p95_latency_ms: float
    success_rate: float


class SaturationDetectionResponse(BaseModel):
    """Response for saturation detection"""

    optimal_concurrency: int = Field(description="Recommended concurrency level")
    max_throughput_tps: float = Field(description="Maximum throughput achieved")
    latency_at_optimal_ms: float = Field(description="Latency at optimal concurrency")
    saturation_concurrency: int = Field(description="Concurrency where saturation started")
    saturation_detected: bool = Field(description="Whether saturation was detected")
    stop_reason: str = Field(description="Reason for stopping")
    concurrency_results: list[ConcurrencyLevelResult] = Field(
        default_factory=list, description="Results for each concurrency level"
    )
