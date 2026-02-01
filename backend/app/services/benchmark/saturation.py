"""
Saturation Detection

Automatically finds the optimal concurrency level by detecting when
performance starts to degrade (throughput plateaus or latency spikes).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import BenchmarkConfig, LoadPattern, SaturationConfig
from .metrics import BenchmarkMetrics
from .runner import BenchmarkRunner

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyResult:
    """Result for a single concurrency level"""

    concurrency: int
    metrics: BenchmarkMetrics
    throughput_tps: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    success_rate: float = 0.0


@dataclass
class SaturationResult:
    """Result from saturation detection"""

    # Optimal concurrency found
    optimal_concurrency: int = 0

    # Maximum throughput achieved
    max_throughput_tps: float = 0.0

    # Latency at optimal concurrency
    latency_at_optimal_ms: float = 0.0

    # Saturation concurrency (where degradation starts)
    saturation_concurrency: int = 0

    # All tested concurrency levels
    results_by_concurrency: list[ConcurrencyResult] = field(default_factory=list)

    # Was saturation detected?
    saturation_detected: bool = False

    # Reason for stopping
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "optimal_concurrency": self.optimal_concurrency,
            "max_throughput_tps": round(self.max_throughput_tps, 2),
            "latency_at_optimal_ms": round(self.latency_at_optimal_ms, 2),
            "saturation_concurrency": self.saturation_concurrency,
            "saturation_detected": self.saturation_detected,
            "stop_reason": self.stop_reason,
            "concurrency_results": [
                {
                    "concurrency": r.concurrency,
                    "throughput_tps": round(r.throughput_tps, 2),
                    "avg_latency_ms": round(r.avg_latency_ms, 2),
                    "p95_latency_ms": round(r.p95_latency_ms, 2),
                    "success_rate": round(r.success_rate, 4),
                }
                for r in self.results_by_concurrency
            ],
        }


class SaturationDetector:
    """
    Detects optimal concurrency by running incremental load tests.

    Algorithm:
    1. Start with low concurrency
    2. Increase concurrency and measure throughput/latency
    3. Stop when:
       - Throughput stops increasing (plateau)
       - Latency increases significantly
       - Error rate increases
    4. Return the concurrency level with best throughput/latency balance
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        config: SaturationConfig | None = None,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.config = config or SaturationConfig(enabled=True)
        self._cancelled = False

    async def detect(self) -> SaturationResult:
        """Run saturation detection"""
        result = SaturationResult()
        results_by_level: list[ConcurrencyResult] = []

        current_concurrency = self.config.start_concurrency
        peak_throughput = 0.0
        baseline_latency = 0.0
        consecutive_degradations = 0

        logger.info(
            f"Starting saturation detection: "
            f"start={self.config.start_concurrency}, "
            f"max={self.config.max_concurrency}"
        )

        while current_concurrency <= self.config.max_concurrency and not self._cancelled:
            logger.info(f"Testing concurrency: {current_concurrency}")

            # Run benchmark at current concurrency
            benchmark_config = BenchmarkConfig(
                endpoint=self.endpoint,
                model_name=self.model_name,
                load_pattern=LoadPattern.FIXED,
                concurrency=current_concurrency,
                num_requests=self.config.requests_per_level,
                warmup_requests=2,
                verbose=False,
            )

            runner = BenchmarkRunner(benchmark_config)
            bench_result = await runner.run()

            if bench_result.error:
                logger.warning(
                    f"Benchmark error at concurrency {current_concurrency}: {bench_result.error}"
                )
                consecutive_degradations += 1

                if consecutive_degradations >= self.config.consecutive_degradations:
                    result.stop_reason = f"Too many errors: {bench_result.error}"
                    result.saturation_detected = True
                    result.saturation_concurrency = current_concurrency
                    break

                # Skip to next level
                current_concurrency = self._next_concurrency(current_concurrency)
                continue

            metrics = bench_result.metrics

            # Record result
            level_result = ConcurrencyResult(
                concurrency=current_concurrency,
                metrics=metrics,
                throughput_tps=metrics.output_tps,
                avg_latency_ms=metrics.e2e_latency.mean,
                p95_latency_ms=metrics.e2e_latency.p95,
                success_rate=metrics.success_rate,
            )
            results_by_level.append(level_result)

            logger.info(
                f"Concurrency {current_concurrency}: "
                f"throughput={metrics.output_tps:.1f} TPS, "
                f"latency={metrics.e2e_latency.mean:.1f}ms (p95={metrics.e2e_latency.p95:.1f}ms), "
                f"success={metrics.success_rate:.1%}"
            )

            # Set baseline latency from first successful run
            if baseline_latency == 0.0 and metrics.e2e_latency.mean > 0:
                baseline_latency = metrics.e2e_latency.mean

            # Check for peak throughput
            if metrics.output_tps > peak_throughput:
                peak_throughput = metrics.output_tps
                consecutive_degradations = 0

            # Check for degradation
            degradation_detected = False

            # Throughput degradation
            if peak_throughput > 0:
                throughput_ratio = metrics.output_tps / peak_throughput
                if throughput_ratio < self.config.degradation_threshold:
                    logger.info(
                        f"Throughput degradation detected: "
                        f"{metrics.output_tps:.1f} vs peak {peak_throughput:.1f}"
                    )
                    degradation_detected = True

            # Latency degradation
            if baseline_latency > 0:
                latency_ratio = metrics.e2e_latency.mean / baseline_latency
                if latency_ratio > self.config.latency_threshold:
                    logger.info(
                        f"Latency degradation detected: "
                        f"{metrics.e2e_latency.mean:.1f}ms vs baseline {baseline_latency:.1f}ms"
                    )
                    degradation_detected = True

            # Success rate check
            if metrics.success_rate < 0.95:
                logger.info(f"High error rate detected: {1 - metrics.success_rate:.1%}")
                degradation_detected = True

            if degradation_detected:
                consecutive_degradations += 1
                if consecutive_degradations >= self.config.consecutive_degradations:
                    result.saturation_detected = True
                    result.saturation_concurrency = current_concurrency
                    result.stop_reason = "Performance degradation detected"
                    break
            else:
                consecutive_degradations = 0

            # Move to next concurrency level
            current_concurrency = self._next_concurrency(current_concurrency)

        # Find optimal concurrency (best throughput/latency ratio)
        if results_by_level:
            result.results_by_concurrency = results_by_level

            # Find best result (highest throughput with acceptable latency)
            best = max(
                results_by_level,
                key=lambda r: (
                    r.throughput_tps / max(1, r.avg_latency_ms / 100) if r.success_rate > 0.9 else 0
                ),
            )

            result.optimal_concurrency = best.concurrency
            result.max_throughput_tps = peak_throughput
            result.latency_at_optimal_ms = best.avg_latency_ms

            if not result.saturation_detected:
                result.stop_reason = "Reached max concurrency"

        logger.info(
            f"Saturation detection complete: "
            f"optimal={result.optimal_concurrency}, "
            f"max_throughput={result.max_throughput_tps:.1f} TPS"
        )

        return result

    def _next_concurrency(self, current: int) -> int:
        """Calculate next concurrency level"""
        if self.config.use_exponential:
            next_val = int(current * self.config.step_multiplier)
            # Ensure we make at least step_size progress
            return max(next_val, current + self.config.step_size)
        else:
            return current + self.config.step_size

    def cancel(self):
        """Cancel the saturation detection"""
        self._cancelled = True


async def find_optimal_concurrency(
    endpoint: str,
    model_name: str,
    max_concurrency: int = 64,
    requests_per_level: int = 20,
) -> SaturationResult:
    """
    Convenience function to find optimal concurrency.

    Args:
        endpoint: OpenAI-compatible API endpoint
        model_name: Model name/ID
        max_concurrency: Maximum concurrency to test
        requests_per_level: Requests per concurrency level

    Returns:
        SaturationResult with optimal concurrency and metrics
    """
    config = SaturationConfig(
        enabled=True,
        start_concurrency=1,
        max_concurrency=max_concurrency,
        requests_per_level=requests_per_level,
    )

    detector = SaturationDetector(endpoint, model_name, config)
    return await detector.detect()
