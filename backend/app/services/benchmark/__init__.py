"""
Benchmark Module

Independent benchmarking system for LLM inference performance evaluation.
Supports comprehensive metrics, multiple load patterns, and saturation detection.

Usage:
    from app.services.benchmark import BenchmarkRunner, BenchmarkConfig, LoadPattern

    config = BenchmarkConfig(
        endpoint="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        load_pattern=LoadPattern.FIXED,
        concurrency=10,
        duration_seconds=60,
    )

    runner = BenchmarkRunner(config)
    result = await runner.run()

    print(f"Throughput: {result.metrics.throughput_tps:.2f} TPS")
    print(f"TTFT p95: {result.metrics.ttft.p95:.2f} ms")
"""

from .config import BenchmarkConfig, LoadPattern, SaturationConfig
from .metrics import BenchmarkMetrics, LatencyMetrics, RequestResult, compute_percentiles
from .runner import BenchmarkResult, BenchmarkRunner
from .saturation import SaturationDetector, SaturationResult

__all__ = [
    # Config
    "BenchmarkConfig",
    "LoadPattern",
    "SaturationConfig",
    # Metrics
    "LatencyMetrics",
    "BenchmarkMetrics",
    "RequestResult",
    "compute_percentiles",
    # Runner
    "BenchmarkRunner",
    "BenchmarkResult",
    # Saturation
    "SaturationDetector",
    "SaturationResult",
]
