"""Auto-Tuning and Benchmark models"""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TuningJobStatus(str, Enum):
    """Tuning job status"""

    PENDING = "pending"
    ANALYZING = "analyzing"
    QUERYING_KB = "querying_kb"
    EXPLORING = "exploring"
    BENCHMARKING = "benchmarking"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationTarget(str, Enum):
    """Optimization target for tuning"""

    THROUGHPUT = "throughput"  # Maximize TPS
    LATENCY = "latency"  # Minimize TTFT/TPOT
    COST = "cost"  # Minimize resource usage
    BALANCED = "balanced"  # Balance all factors


class TuningJob(Base):
    """Auto-tuning job record"""

    __tablename__ = "tuning_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Target configuration
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("llm_models.id"), nullable=False)
    worker_id: Mapped[int] = mapped_column(Integer, ForeignKey("workers.id"), nullable=False)
    optimization_target: Mapped[str] = mapped_column(
        String(50), default=OptimizationTarget.THROUGHPUT.value
    )

    # Tuning configuration - which frameworks and parameters to test
    # Format: {
    #   "engines": ["vllm", "sglang"],
    #   "parameters": {
    #     "tensor_parallel_size": [1, 2],
    #     "gpu_memory_utilization": [0.85, 0.90],
    #     "max_model_len": [4096, 8192]
    #   },
    #   "benchmark": {
    #     "duration_seconds": 60,
    #     "input_length": 512,
    #     "output_length": 128,
    #     "concurrency": [1, 4, 8]
    #   }
    # }
    tuning_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Job status
    status: Mapped[str] = mapped_column(String(50), default=TuningJobStatus.PENDING.value)
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=5)

    # Progress details (JSON for flexibility)
    # Format: {
    #   "step": 3,
    #   "total_steps": 5,
    #   "step_name": "benchmarking",
    #   "configs_tested": 5,
    #   "configs_total": 12,
    #   "current_config": {"engine": "vllm", "tensor_parallel_size": 2, ...},
    #   "results": [{"config": {...}, "throughput_tps": 1234.5, ...}, ...]
    # }
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Results - final sorted results
    # Format: [{"rank": 1, "engine": "vllm", "config": {...}, "throughput_tps": 1500, ...}, ...]
    best_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    all_results: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Agent conversation ID (links to conversations table for Agent Chat display)
    conversation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("conversations.id"), nullable=True
    )

    # Legacy: Agent conversation log (for backward compatibility)
    conversation_log: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Tuning logs for frontend display
    logs: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    model = relationship("LLMModel", backref="tuning_jobs")
    worker = relationship("Worker", backref="tuning_jobs")


class BenchmarkResult(Base):
    """Benchmark result for a specific configuration"""

    __tablename__ = "benchmark_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Associated tuning job (optional - can run standalone benchmarks)
    tuning_job_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("tuning_jobs.id"), nullable=True
    )

    # Deployment being benchmarked
    deployment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("deployments.id"), nullable=False
    )

    # Configuration tested
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    # Example config:
    # {
    #     "engine": "vllm",
    #     "quantization": "fp16",
    #     "tensor_parallel": 1,
    #     "extra_args": {...}
    # }

    # Test parameters
    test_type: Mapped[str] = mapped_column(String(50), default="throughput")
    test_duration_seconds: Mapped[int] = mapped_column(Integer, default=60)
    input_length: Mapped[int] = mapped_column(Integer, default=512)
    output_length: Mapped[int] = mapped_column(Integer, default=128)
    concurrency: Mapped[int] = mapped_column(Integer, default=1)

    # Performance metrics
    throughput_tps: Mapped[float | None] = mapped_column(Float, nullable=True)  # Tokens per second
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)  # Time to first token (ms)
    tpot_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # Time per output token (ms)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Resource usage
    gpu_utilization: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-100%
    vram_usage_gb: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Raw results
    raw_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    # Relationships
    tuning_job = relationship("TuningJob", backref="benchmark_results")
    deployment = relationship("Deployment", backref="benchmark_results")


class PerformanceKnowledge(Base):
    """Performance knowledge base for configuration recommendations

    This table stores historical tuning results to enable:
    1. Fast lookup of known-good configurations
    2. Transfer learning across similar models/hardware
    """

    __tablename__ = "performance_knowledge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Hardware info
    gpu_model: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "NVIDIA H100 80GB"
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_vram_gb: Mapped[float] = mapped_column(Float, nullable=False)

    # Model info
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)  # e.g., "Qwen/Qwen2.5-72B"
    model_family: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "Qwen"
    model_params_b: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # Parameters in billions

    # Configuration
    engine: Mapped[str] = mapped_column(String(50), nullable=False)  # vllm, sglang, ollama
    quantization: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # fp16, fp8, awq, gptq
    tensor_parallel: Mapped[int] = mapped_column(Integer, default=1)
    extra_args: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Performance metrics
    throughput_tps: Mapped[float] = mapped_column(Float, nullable=False)
    ttft_ms: Mapped[float] = mapped_column(Float, nullable=False)
    tpot_ms: Mapped[float] = mapped_column(Float, nullable=False)
    gpu_utilization: Mapped[float | None] = mapped_column(Float, nullable=True)
    vram_usage_gb: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Test conditions
    test_dataset: Mapped[str] = mapped_column(String(100), default="synthetic")
    input_length: Mapped[int] = mapped_column(Integer, default=512)
    output_length: Mapped[int] = mapped_column(Integer, default=128)
    concurrency: Mapped[int] = mapped_column(Integer, default=1)

    # Recommendation score (computed based on optimization target)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Source
    source_tuning_job_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("tuning_jobs.id"), nullable=True
    )

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    # Relationships
    source_tuning_job = relationship("TuningJob", backref="knowledge_records")
