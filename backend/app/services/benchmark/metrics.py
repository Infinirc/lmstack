"""
Benchmark Metrics

Comprehensive metrics calculation including:
- TTFT (Time to First Token)
- ITL (Inter-Token Latency)
- TPOT (Time Per Output Token)
- Percentiles (p50, p90, p95, p99)
- Throughput (tokens/sec, requests/sec)
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    """Result from a single request"""

    # Timing metrics (in milliseconds)
    ttft_ms: float = 0.0  # Time to first token
    total_latency_ms: float = 0.0  # Total request duration
    token_timestamps_ms: list[float] = field(default_factory=list)  # Timestamp of each token

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Request metadata
    success: bool = True
    error_message: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def itl_values_ms(self) -> list[float]:
        """Calculate inter-token latencies from timestamps"""
        if len(self.token_timestamps_ms) < 2:
            return []

        itl_values = []
        for i in range(1, len(self.token_timestamps_ms)):
            itl = self.token_timestamps_ms[i] - self.token_timestamps_ms[i - 1]
            itl_values.append(itl)
        return itl_values

    @property
    def tpot_ms(self) -> float:
        """Time per output token (excluding TTFT)"""
        if self.completion_tokens <= 1:
            return 0.0

        generation_time = self.total_latency_ms - self.ttft_ms
        return generation_time / (self.completion_tokens - 1)


@dataclass
class LatencyMetrics:
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

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary"""
        return {
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "std": round(self.std, 2),
            "p50": round(self.p50, 2),
            "p90": round(self.p90, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


def compute_percentiles(values: Sequence[float]) -> LatencyMetrics:
    """Compute latency metrics with percentiles from a sequence of values"""
    if not values:
        return LatencyMetrics()

    sorted_values = sorted(values)
    n = len(sorted_values)

    def percentile(p: float) -> float:
        """Calculate percentile value"""
        if n == 1:
            return sorted_values[0]
        k = (n - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    return LatencyMetrics(
        mean=statistics.mean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
        std=statistics.stdev(values) if n > 1 else 0.0,
        p50=percentile(50),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
    )


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics"""

    # Latency metrics
    ttft: LatencyMetrics = field(default_factory=LatencyMetrics)
    itl: LatencyMetrics = field(default_factory=LatencyMetrics)
    tpot: LatencyMetrics = field(default_factory=LatencyMetrics)
    e2e_latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    # Throughput metrics
    throughput_tps: float = 0.0  # Tokens per second
    throughput_rps: float = 0.0  # Requests per second
    output_tps: float = 0.0  # Output tokens per second

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

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "ttft": self.ttft.to_dict(),
            "itl": self.itl.to_dict(),
            "tpot": self.tpot.to_dict(),
            "e2e_latency": self.e2e_latency.to_dict(),
            "throughput_tps": round(self.throughput_tps, 2),
            "throughput_rps": round(self.throughput_rps, 4),
            "output_tps": round(self.output_tps, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 1),
            "avg_completion_tokens": round(self.avg_completion_tokens, 1),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "concurrency": self.concurrency,
        }

    @classmethod
    def from_results(
        cls,
        results: list[RequestResult],
        total_duration: float,
        concurrency: int,
    ) -> "BenchmarkMetrics":
        """Compute metrics from a list of request results"""
        if not results:
            return cls()

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if not successful:
            return cls(
                total_requests=len(results),
                failed_requests=len(failed),
                total_duration_seconds=total_duration,
                concurrency=concurrency,
            )

        # Collect latency values
        ttft_values = [r.ttft_ms for r in successful if r.ttft_ms > 0]
        e2e_values = [r.total_latency_ms for r in successful if r.total_latency_ms > 0]
        tpot_values = [r.tpot_ms for r in successful if r.tpot_ms > 0]

        # Collect all ITL values
        all_itl_values: list[float] = []
        for r in successful:
            all_itl_values.extend(r.itl_values_ms)

        # Token counts
        total_prompt = sum(r.prompt_tokens for r in successful)
        total_completion = sum(r.completion_tokens for r in successful)
        total_tokens = total_prompt + total_completion

        # Throughput
        throughput_tps = total_tokens / total_duration if total_duration > 0 else 0
        throughput_rps = len(successful) / total_duration if total_duration > 0 else 0
        output_tps = total_completion / total_duration if total_duration > 0 else 0

        return cls(
            ttft=compute_percentiles(ttft_values),
            itl=compute_percentiles(all_itl_values) if all_itl_values else LatencyMetrics(),
            tpot=compute_percentiles(tpot_values),
            e2e_latency=compute_percentiles(e2e_values),
            throughput_tps=throughput_tps,
            throughput_rps=throughput_rps,
            output_tps=output_tps,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            success_rate=len(successful) / len(results) if results else 0,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            avg_prompt_tokens=total_prompt / len(successful) if successful else 0,
            avg_completion_tokens=total_completion / len(successful) if successful else 0,
            total_duration_seconds=total_duration,
            concurrency=concurrency,
        )

    def to_simple_dict(self) -> dict[str, float]:
        """Convert to simple dictionary for backward compatibility with BayesianTuner"""
        return {
            "throughput_tps": self.output_tps,
            "avg_ttft_ms": self.ttft.mean,
            "avg_tpot_ms": self.tpot.mean,
            "avg_itl_ms": self.itl.mean,
            "ttft_p50_ms": self.ttft.p50,
            "ttft_p95_ms": self.ttft.p95,
            "ttft_p99_ms": self.ttft.p99,
            "tpot_p50_ms": self.tpot.p50,
            "tpot_p95_ms": self.tpot.p95,
            "tpot_p99_ms": self.tpot.p99,
            "itl_p50_ms": self.itl.p50,
            "itl_p95_ms": self.itl.p95,
            "itl_p99_ms": self.itl.p99,
            "e2e_latency_p50_ms": self.e2e_latency.p50,
            "e2e_latency_p95_ms": self.e2e_latency.p95,
            "e2e_latency_p99_ms": self.e2e_latency.p99,
            "successful_requests": float(self.successful_requests),
            "total_requests": float(self.total_requests),
            "success_rate": self.success_rate,
        }
