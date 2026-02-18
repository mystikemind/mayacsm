"""
Latency Benchmarking Suite for Project Maya.

This module provides comprehensive latency benchmarks for all pipeline
components, with detailed statistical analysis and reporting.

Usage:
    # Run full benchmark suite
    python -m tests.benchmarks.latency_benchmark

    # Run specific component benchmark
    python -m tests.benchmarks.latency_benchmark --component moshi

    # Generate JSON report
    python -m tests.benchmarks.latency_benchmark --output report.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from maya.config import settings
from maya.constants import (
    DEFAULT_SAMPLE_RATE,
    LATENCY_TARGET_QUICK_PATH_MS,
    LATENCY_TARGET_ENHANCED_PATH_MS,
    MOSHI_LATENCY_TARGET_MS,
    CHATTERBOX_LATENCY_TARGET_MS,
    EMOTION_LATENCY_TARGET_MS,
)
from maya.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from numpy.typing import NDArray

setup_logging()
logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""

    count: int = 0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    target_ms: float = 0.0
    passes_target: bool = False

    @classmethod
    def from_measurements(
        cls,
        measurements: list[float],
        target_ms: float,
    ) -> "LatencyStats":
        """Create stats from raw measurements."""
        if not measurements:
            return cls()

        return cls(
            count=len(measurements),
            mean_ms=statistics.mean(measurements),
            std_ms=statistics.stdev(measurements) if len(measurements) > 1 else 0.0,
            min_ms=min(measurements),
            max_ms=max(measurements),
            p50_ms=float(np.percentile(measurements, 50)),
            p90_ms=float(np.percentile(measurements, 90)),
            p95_ms=float(np.percentile(measurements, 95)),
            p99_ms=float(np.percentile(measurements, 99)),
            target_ms=target_ms,
            passes_target=float(np.percentile(measurements, 95)) < target_ms,
        )


@dataclass
class ComponentBenchmark:
    """Benchmark results for a single component."""

    name: str
    description: str
    latency: LatencyStats
    throughput_rps: float = 0.0  # Requests per second
    vram_usage_gb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    timestamp: str
    environment: dict[str, Any]
    components: list[ComponentBenchmark]
    pipeline: dict[str, LatencyStats]
    summary: dict[str, Any]

    def to_json(self) -> str:
        """Convert report to JSON string."""
        def serialize(obj):
            if isinstance(obj, LatencyStats):
                return asdict(obj)
            if isinstance(obj, ComponentBenchmark):
                return {
                    "name": obj.name,
                    "description": obj.description,
                    "latency": asdict(obj.latency),
                    "throughput_rps": obj.throughput_rps,
                    "vram_usage_gb": obj.vram_usage_gb,
                    "metadata": obj.metadata,
                }
            return obj

        data = {
            "timestamp": self.timestamp,
            "environment": self.environment,
            "components": [serialize(c) for c in self.components],
            "pipeline": {k: asdict(v) for k, v in self.pipeline.items()},
            "summary": self.summary,
        }
        return json.dumps(data, indent=2)


# =============================================================================
# Benchmark Utilities
# =============================================================================


def get_environment_info() -> dict[str, Any]:
    """Gather environment information."""
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })

    return info


def generate_test_audio(duration_seconds: float = 2.0) -> "NDArray[np.float32]":
    """Generate realistic test audio."""
    num_samples = int(DEFAULT_SAMPLE_RATE * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples)

    # Simulate speech with harmonics
    fundamental = 150  # Hz
    audio = (
        0.3 * np.sin(2 * np.pi * fundamental * t) +
        0.15 * np.sin(2 * np.pi * fundamental * 2 * t) +
        0.1 * np.sin(2 * np.pi * fundamental * 3 * t)
    )

    # Add envelope
    envelope = np.ones(num_samples)
    fade = int(0.1 * num_samples)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)

    audio = (audio * envelope).astype(np.float32)
    return audio


def get_vram_usage_gb() -> float:
    """Get current VRAM usage."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


# =============================================================================
# Component Benchmarks
# =============================================================================


class BenchmarkRunner:
    """Runs latency benchmarks for all components."""

    def __init__(
        self,
        iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> None:
        """
        Initialize benchmark runner.

        Args:
            iterations: Number of measurement iterations.
            warmup_iterations: Number of warmup iterations (not measured).
        """
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.test_audio_2s = generate_test_audio(2.0)
        self.test_audio_500ms = generate_test_audio(0.5)

    async def benchmark_moshi(self) -> ComponentBenchmark:
        """Benchmark Moshi engine latency."""
        from maya.core import MoshiEngine, MoshiEngineConfig

        logger.info("Benchmarking Moshi engine...")

        config = MoshiEngineConfig(device="cuda:0", warmup_on_init=True)
        engine = MoshiEngine(config)

        try:
            await engine.initialize()
            vram_before = get_vram_usage_gb()

            # Warmup
            for _ in range(self.warmup_iterations):
                async for _ in engine.process_stream(self.test_audio_500ms):
                    break

            # Measure
            latencies = []
            for i in range(self.iterations):
                start = time.perf_counter()
                async for response in engine.process_stream(self.test_audio_500ms):
                    if response.audio is not None:
                        latency = (time.perf_counter() - start) * 1000
                        latencies.append(latency)
                        break

                if (i + 1) % 20 == 0:
                    logger.info(f"Moshi: {i + 1}/{self.iterations} iterations")

            vram_after = get_vram_usage_gb()

            stats = LatencyStats.from_measurements(latencies, MOSHI_LATENCY_TARGET_MS)
            throughput = len(latencies) / (sum(latencies) / 1000) if latencies else 0

            return ComponentBenchmark(
                name="moshi",
                description="Moshi S2S Engine - Audio to Audio+Text",
                latency=stats,
                throughput_rps=throughput,
                vram_usage_gb=vram_after,
                metadata={
                    "model": config.model_repo,
                    "quantization": "bf16",
                    "audio_duration_ms": 500,
                },
            )

        finally:
            await engine.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    async def benchmark_chatterbox(self) -> ComponentBenchmark:
        """Benchmark Chatterbox TTS latency."""
        from maya.core import ChatterboxTTSEngine, ChatterboxConfig

        logger.info("Benchmarking Chatterbox TTS...")

        config = ChatterboxConfig(device="cuda:0", exaggeration=0.6)
        engine = ChatterboxTTSEngine(config)

        test_texts = [
            "Hello, how are you today?",
            "That's a great question.",
            "Let me think about that for a moment.",
            "I understand what you mean.",
            "Yes, I can help you with that.",
        ]

        try:
            await engine.initialize()
            vram_before = get_vram_usage_gb()

            # Warmup
            for _ in range(self.warmup_iterations):
                async for _ in engine.synthesize_stream(test_texts[0]):
                    break

            # Measure time to first audio chunk
            latencies = []
            for i in range(self.iterations):
                text = test_texts[i % len(test_texts)]
                start = time.perf_counter()
                async for chunk in engine.synthesize_stream(text):
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    break

                if (i + 1) % 20 == 0:
                    logger.info(f"Chatterbox: {i + 1}/{self.iterations} iterations")

            vram_after = get_vram_usage_gb()

            stats = LatencyStats.from_measurements(latencies, CHATTERBOX_LATENCY_TARGET_MS)
            throughput = len(latencies) / (sum(latencies) / 1000) if latencies else 0

            return ComponentBenchmark(
                name="chatterbox",
                description="Chatterbox TTS - Text to Audio (first chunk)",
                latency=stats,
                throughput_rps=throughput,
                vram_usage_gb=vram_after,
                metadata={
                    "exaggeration": config.exaggeration,
                    "streaming": True,
                },
            )

        finally:
            await engine.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    async def benchmark_emotion(self) -> ComponentBenchmark:
        """Benchmark emotion detector latency."""
        from maya.core import EmotionDetector, EmotionDetectorConfig

        logger.info("Benchmarking Emotion Detector...")

        config = EmotionDetectorConfig(device="cuda:0", warmup_on_init=True)
        detector = EmotionDetector(config)

        try:
            await detector.initialize()
            vram_before = get_vram_usage_gb()

            # Warmup
            for _ in range(self.warmup_iterations):
                await detector.detect(self.test_audio_2s)

            # Measure
            latencies = []
            for i in range(self.iterations):
                result = await detector.detect(self.test_audio_2s)
                latencies.append(result.latency_ms)

                if (i + 1) % 20 == 0:
                    logger.info(f"Emotion: {i + 1}/{self.iterations} iterations")

            vram_after = get_vram_usage_gb()

            stats = LatencyStats.from_measurements(latencies, EMOTION_LATENCY_TARGET_MS)
            throughput = len(latencies) / (sum(latencies) / 1000) if latencies else 0

            return ComponentBenchmark(
                name="emotion",
                description="Emotion Detector - Audio to Emotion",
                latency=stats,
                throughput_rps=throughput,
                vram_usage_gb=vram_after,
                metadata={
                    "model": config.model_name,
                    "audio_duration_s": 2.0,
                },
            )

        finally:
            await detector.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    async def benchmark_router(self) -> ComponentBenchmark:
        """Benchmark response router latency."""
        from maya.core import ResponseRouter, RouterConfig

        logger.info("Benchmarking Response Router...")

        router = ResponseRouter(RouterConfig())

        test_texts = [
            "okay",
            "uh-huh",
            "That's a really interesting point you're making.",
            "Let me think about that for a moment.",
            "?",
            "I love that idea!",
        ]

        # Warmup
        for _ in range(100):
            router.route(test_texts[0])

        # Measure
        latencies = []
        for i in range(self.iterations * 10):  # More iterations since it's fast
            text = test_texts[i % len(test_texts)]
            start = time.perf_counter()
            router.route(text)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        stats = LatencyStats.from_measurements(latencies, target_ms=1.0)
        throughput = len(latencies) / (sum(latencies) / 1000) if latencies else 0

        return ComponentBenchmark(
            name="router",
            description="Response Router - Text to Path Decision",
            latency=stats,
            throughput_rps=throughput,
            vram_usage_gb=0.0,
            metadata={
                "cpu_only": True,
            },
        )

    async def benchmark_humanizer(self) -> ComponentBenchmark:
        """Benchmark humanizer latency."""
        from maya.core import Humanizer
        from maya.humanize import FillerConfig, PauseConfig

        logger.info("Benchmarking Humanizer...")

        humanizer = Humanizer(FillerConfig(), PauseConfig())

        test_texts = [
            "Hello, how are you doing today?",
            "That's a great question. Let me explain.",
            "I understand what you mean. Here's my perspective.",
            "Well, there are several factors to consider here.",
        ]

        # Warmup
        for _ in range(100):
            humanizer.humanize(test_texts[0])

        # Measure
        latencies = []
        for i in range(self.iterations * 10):
            text = test_texts[i % len(test_texts)]
            start = time.perf_counter()
            humanizer.humanize(text)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        stats = LatencyStats.from_measurements(latencies, target_ms=5.0)
        throughput = len(latencies) / (sum(latencies) / 1000) if latencies else 0

        return ComponentBenchmark(
            name="humanizer",
            description="Humanizer - Text to Humanized Text",
            latency=stats,
            throughput_rps=throughput,
            vram_usage_gb=0.0,
            metadata={
                "cpu_only": True,
                "includes_fillers": True,
                "includes_pauses": True,
            },
        )

    async def benchmark_quick_path(self) -> LatencyStats:
        """Benchmark full quick path latency."""
        from maya.core import MoshiEngine, MoshiEngineConfig, ResponseRouter, RouterConfig

        logger.info("Benchmarking Quick Path (end-to-end)...")

        moshi_config = MoshiEngineConfig(device="cuda:0", warmup_on_init=True)
        moshi = MoshiEngine(moshi_config)
        router = ResponseRouter(RouterConfig())

        try:
            await moshi.initialize()

            # Warmup
            for _ in range(5):
                async for _ in moshi.process_stream(self.test_audio_500ms):
                    break

            # Measure
            latencies = []
            for i in range(min(self.iterations, 50)):
                start = time.perf_counter()

                async for response in moshi.process_stream(self.test_audio_500ms):
                    if response.audio is not None:
                        # Route decision
                        text = response.text or "okay"
                        decision = router.route(text)

                        # Quick path uses Moshi audio directly
                        if decision.path.value == "quick":
                            latency = (time.perf_counter() - start) * 1000
                            latencies.append(latency)
                        break

                if (i + 1) % 10 == 0:
                    logger.info(f"Quick path: {i + 1}/{min(self.iterations, 50)} iterations")

            return LatencyStats.from_measurements(latencies, LATENCY_TARGET_QUICK_PATH_MS)

        finally:
            await moshi.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    async def benchmark_enhanced_path(self) -> LatencyStats:
        """Benchmark full enhanced path latency."""
        from maya.core import (
            MoshiEngine,
            MoshiEngineConfig,
            ChatterboxTTSEngine,
            ChatterboxConfig,
            ResponseRouter,
            RouterConfig,
            Humanizer,
        )
        from maya.humanize import FillerConfig, PauseConfig

        logger.info("Benchmarking Enhanced Path (end-to-end)...")

        moshi = MoshiEngine(MoshiEngineConfig(device="cuda:0", warmup_on_init=True))
        chatterbox = ChatterboxTTSEngine(ChatterboxConfig(device="cuda:0"))
        router = ResponseRouter(RouterConfig())
        humanizer = Humanizer(FillerConfig(), PauseConfig())

        try:
            await moshi.initialize()
            await chatterbox.initialize()

            # Warmup
            for _ in range(3):
                async for response in moshi.process_stream(self.test_audio_2s):
                    if response.text:
                        humanized = humanizer.humanize(response.text)
                        async for _ in chatterbox.synthesize_stream(humanized):
                            break
                        break

            # Measure
            latencies = []
            for i in range(min(self.iterations, 30)):
                start = time.perf_counter()

                async for response in moshi.process_stream(self.test_audio_2s):
                    if response.text:
                        # Humanize
                        humanized = humanizer.humanize(response.text)

                        # Synthesize
                        async for audio_chunk in chatterbox.synthesize_stream(humanized):
                            latency = (time.perf_counter() - start) * 1000
                            latencies.append(latency)
                            break
                        break

                if (i + 1) % 10 == 0:
                    logger.info(f"Enhanced path: {i + 1}/{min(self.iterations, 30)} iterations")

            return LatencyStats.from_measurements(latencies, LATENCY_TARGET_ENHANCED_PATH_MS)

        finally:
            await moshi.cleanup()
            await chatterbox.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    async def run_full_benchmark(self) -> BenchmarkReport:
        """Run complete benchmark suite."""
        logger.info("Starting full benchmark suite...")

        components = []
        pipeline = {}

        # CPU-only components first
        components.append(await self.benchmark_router())
        components.append(await self.benchmark_humanizer())

        # GPU components
        if torch.cuda.is_available():
            components.append(await self.benchmark_emotion())
            components.append(await self.benchmark_chatterbox())
            components.append(await self.benchmark_moshi())

            # Full pipeline paths
            pipeline["quick_path"] = await self.benchmark_quick_path()
            pipeline["enhanced_path"] = await self.benchmark_enhanced_path()
        else:
            logger.warning("GPU not available, skipping GPU benchmarks")

        # Generate summary
        summary = {
            "total_components": len(components),
            "passing_targets": sum(1 for c in components if c.latency.passes_target),
            "total_vram_gb": sum(c.vram_usage_gb for c in components),
        }

        if pipeline:
            summary["quick_path_p95_ms"] = pipeline.get("quick_path", LatencyStats()).p95_ms
            summary["enhanced_path_p95_ms"] = pipeline.get("enhanced_path", LatencyStats()).p95_ms

        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            environment=get_environment_info(),
            components=components,
            pipeline=pipeline,
            summary=summary,
        )


# =============================================================================
# CLI
# =============================================================================


def print_report(report: BenchmarkReport) -> None:
    """Print benchmark report to console."""
    print("\n" + "=" * 80)
    print("PROJECT MAYA - LATENCY BENCHMARK REPORT")
    print("=" * 80)
    print(f"\nTimestamp: {report.timestamp}")
    print(f"GPU: {report.environment.get('gpu_name', 'N/A')}")
    print(f"CUDA: {report.environment.get('cuda_version', 'N/A')}")

    print("\n" + "-" * 80)
    print("COMPONENT BENCHMARKS")
    print("-" * 80)

    for comp in report.components:
        status = "PASS" if comp.latency.passes_target else "FAIL"
        print(f"\n{comp.name.upper()} - {comp.description}")
        print(f"  Mean:     {comp.latency.mean_ms:7.2f} ms")
        print(f"  P50:      {comp.latency.p50_ms:7.2f} ms")
        print(f"  P95:      {comp.latency.p95_ms:7.2f} ms")
        print(f"  P99:      {comp.latency.p99_ms:7.2f} ms")
        print(f"  Target:   {comp.latency.target_ms:7.2f} ms  [{status}]")
        print(f"  VRAM:     {comp.vram_usage_gb:7.2f} GB")

    if report.pipeline:
        print("\n" + "-" * 80)
        print("END-TO-END PIPELINE LATENCY")
        print("-" * 80)

        for name, stats in report.pipeline.items():
            status = "PASS" if stats.passes_target else "FAIL"
            print(f"\n{name.upper()}")
            print(f"  Mean:     {stats.mean_ms:7.2f} ms")
            print(f"  P95:      {stats.p95_ms:7.2f} ms")
            print(f"  Target:   {stats.target_ms:7.2f} ms  [{status}]")

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"  Components passing target: {report.summary['passing_targets']}/{report.summary['total_components']}")
    print(f"  Total VRAM usage: {report.summary['total_vram_gb']:.2f} GB")

    if "quick_path_p95_ms" in report.summary:
        print(f"  Quick path P95: {report.summary['quick_path_p95_ms']:.2f} ms")
        print(f"  Enhanced path P95: {report.summary['enhanced_path_p95_ms']:.2f} ms")

    print("\n" + "=" * 80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run latency benchmarks for Project Maya",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        choices=["moshi", "chatterbox", "emotion", "router", "humanizer"],
        help="Run benchmark for specific component only",
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(
        iterations=args.iterations,
        warmup_iterations=args.warmup,
    )

    if args.component:
        # Run single component benchmark
        benchmark_funcs = {
            "moshi": runner.benchmark_moshi,
            "chatterbox": runner.benchmark_chatterbox,
            "emotion": runner.benchmark_emotion,
            "router": runner.benchmark_router,
            "humanizer": runner.benchmark_humanizer,
        }
        result = await benchmark_funcs[args.component]()
        print(f"\n{result.name.upper()} Benchmark Results:")
        print(f"  Mean: {result.latency.mean_ms:.2f} ms")
        print(f"  P95:  {result.latency.p95_ms:.2f} ms")
        print(f"  Target: {result.latency.target_ms:.2f} ms")
        print(f"  Status: {'PASS' if result.latency.passes_target else 'FAIL'}")
    else:
        # Run full benchmark suite
        report = await runner.run_full_benchmark()

        # Print report
        print_report(report)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(report.to_json())
            print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
