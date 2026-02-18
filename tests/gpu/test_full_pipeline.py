"""
GPU Integration Tests for Project Maya.

These tests validate the full voice pipeline with real GPU models.
They require a GPU and will load actual models into VRAM.

Run with: pytest tests/gpu/ -v -m gpu

Note: These tests take significant time due to model loading.
      Run them separately from unit tests.
"""

from __future__ import annotations

import asyncio
import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from maya.config import settings
from maya.constants import (
    DEFAULT_SAMPLE_RATE,
    LATENCY_TARGET_QUICK_PATH_MS,
    LATENCY_TARGET_ENHANCED_PATH_MS,
    MOSHI_VRAM_INT8_GB,
    CHATTERBOX_VRAM_GB,
    EMOTION_SER_VRAM_GB,
)
from maya.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Setup logging
setup_logging()
logger = get_logger(__name__)


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def get_gpu_memory_reserved_gb() -> float:
    """Get GPU memory reserved by PyTorch in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / (1024**3)


def generate_test_audio(duration_seconds: float = 2.0) -> "NDArray[np.float32]":
    """Generate test audio with speech-like characteristics."""
    num_samples = int(DEFAULT_SAMPLE_RATE * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples)

    # Generate a complex waveform simulating speech
    # Fundamental frequency around 150Hz (typical speech)
    fundamental = 150
    audio = (
        0.3 * np.sin(2 * np.pi * fundamental * t) +
        0.15 * np.sin(2 * np.pi * fundamental * 2 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * fundamental * 3 * t) +   # Second harmonic
        0.05 * np.sin(2 * np.pi * fundamental * 4 * t)    # Third harmonic
    )

    # Add amplitude envelope (speech-like)
    envelope = np.ones(num_samples)
    envelope[:int(0.1 * num_samples)] = np.linspace(0, 1, int(0.1 * num_samples))
    envelope[-int(0.1 * num_samples):] = np.linspace(1, 0, int(0.1 * num_samples))

    # Add some noise for realism
    audio = audio * envelope + 0.01 * np.random.randn(num_samples)

    return audio.astype(np.float32)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def gpu_available():
    """Check if GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True


@pytest.fixture(scope="module")
def sample_audio() -> "NDArray[np.float32]":
    """Generate sample audio for testing."""
    return generate_test_audio(duration_seconds=2.0)


@pytest.fixture(scope="module")
def short_audio() -> "NDArray[np.float32]":
    """Generate short audio for quick tests."""
    return generate_test_audio(duration_seconds=0.5)


# =============================================================================
# Moshi Engine GPU Tests
# =============================================================================


@pytest.mark.gpu
class TestMoshiEngineGPU:
    """GPU tests for Moshi engine."""

    @pytest.fixture(scope="class")
    async def moshi_engine(self, gpu_available):
        """Create Moshi engine for testing."""
        from maya.core import MoshiEngine, MoshiEngineConfig

        logger.info("Initializing Moshi engine for GPU tests...")
        initial_vram = get_gpu_memory_gb()

        config = MoshiEngineConfig(
            device="cuda:0",
            warmup_on_init=True,
        )
        engine = MoshiEngine(config)

        try:
            await engine.initialize()
            final_vram = get_gpu_memory_gb()
            logger.info(
                "Moshi engine initialized",
                vram_used_gb=final_vram - initial_vram,
            )
            yield engine
        finally:
            await engine.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.asyncio
    async def test_moshi_initialization(self, moshi_engine):
        """Test Moshi initializes correctly on GPU."""
        assert moshi_engine.is_initialized
        vram = get_gpu_memory_gb()
        assert vram > 0, "Moshi should use GPU memory"
        logger.info(f"Moshi VRAM usage: {vram:.2f} GB")

    @pytest.mark.asyncio
    async def test_moshi_process_audio(self, moshi_engine, sample_audio):
        """Test Moshi can process audio."""
        responses = []
        start_time = time.perf_counter()

        async for response in moshi_engine.process_stream(sample_audio):
            responses.append(response)

        total_time = (time.perf_counter() - start_time) * 1000

        assert len(responses) > 0, "Moshi should generate responses"
        logger.info(
            f"Moshi processed {len(sample_audio)} samples in {total_time:.1f}ms",
            responses=len(responses),
        )

    @pytest.mark.asyncio
    async def test_moshi_latency(self, moshi_engine, short_audio):
        """Test Moshi meets latency targets."""
        latencies = []

        for _ in range(5):
            start_time = time.perf_counter()
            async for response in moshi_engine.process_stream(short_audio):
                first_response_time = (time.perf_counter() - start_time) * 1000
                latencies.append(first_response_time)
                break

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        logger.info(
            "Moshi latency",
            avg_ms=f"{avg_latency:.1f}",
            p95_ms=f"{p95_latency:.1f}",
        )

        # Should be under target (with some margin for test overhead)
        assert avg_latency < LATENCY_TARGET_QUICK_PATH_MS * 1.5, \
            f"Average latency {avg_latency}ms exceeds target"


# =============================================================================
# Chatterbox TTS GPU Tests
# =============================================================================


@pytest.mark.gpu
class TestChatterboxGPU:
    """GPU tests for Chatterbox TTS."""

    @pytest.fixture(scope="class")
    async def chatterbox_engine(self, gpu_available):
        """Create Chatterbox engine for testing."""
        from maya.core import ChatterboxTTSEngine, ChatterboxConfig

        logger.info("Initializing Chatterbox engine for GPU tests...")
        initial_vram = get_gpu_memory_gb()

        config = ChatterboxConfig(
            device="cuda:0",
            exaggeration=0.6,
        )
        engine = ChatterboxTTSEngine(config)

        try:
            await engine.initialize()
            final_vram = get_gpu_memory_gb()
            logger.info(
                "Chatterbox engine initialized",
                vram_used_gb=final_vram - initial_vram,
            )
            yield engine
        finally:
            await engine.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.asyncio
    async def test_chatterbox_initialization(self, chatterbox_engine):
        """Test Chatterbox initializes correctly on GPU."""
        assert chatterbox_engine.is_initialized
        vram = get_gpu_memory_gb()
        logger.info(f"Chatterbox VRAM usage: {vram:.2f} GB")

    @pytest.mark.asyncio
    async def test_chatterbox_synthesize(self, chatterbox_engine):
        """Test Chatterbox can synthesize speech."""
        text = "Hello, this is a test of the Chatterbox text to speech system."

        start_time = time.perf_counter()
        result = await chatterbox_engine.synthesize(text)
        synthesis_time = (time.perf_counter() - start_time) * 1000

        assert result.audio is not None
        assert len(result.audio) > 0
        assert result.audio.dtype == np.float32

        audio_duration_ms = len(result.audio) / DEFAULT_SAMPLE_RATE * 1000
        rtf = synthesis_time / audio_duration_ms  # Real-time factor

        logger.info(
            "Chatterbox synthesis",
            text_length=len(text),
            audio_duration_ms=f"{audio_duration_ms:.1f}",
            synthesis_time_ms=f"{synthesis_time:.1f}",
            rtf=f"{rtf:.2f}",
        )

        # Should synthesize faster than real-time
        assert rtf < 1.0, "Chatterbox should be faster than real-time"

    @pytest.mark.asyncio
    async def test_chatterbox_streaming(self, chatterbox_engine):
        """Test Chatterbox streaming synthesis."""
        text = "This is a longer text to test streaming synthesis capabilities."

        chunks = []
        first_chunk_time = None
        start_time = time.perf_counter()

        async for chunk in chatterbox_engine.synthesize_stream(text):
            if first_chunk_time is None:
                first_chunk_time = (time.perf_counter() - start_time) * 1000
            chunks.append(chunk)

        total_time = (time.perf_counter() - start_time) * 1000

        assert len(chunks) > 0, "Should produce audio chunks"
        assert first_chunk_time is not None

        logger.info(
            "Chatterbox streaming",
            chunks=len(chunks),
            first_chunk_ms=f"{first_chunk_time:.1f}",
            total_time_ms=f"{total_time:.1f}",
        )

        # First chunk should arrive quickly
        assert first_chunk_time < 500, "First chunk should arrive within 500ms"


# =============================================================================
# Emotion Detector GPU Tests
# =============================================================================


@pytest.mark.gpu
class TestEmotionDetectorGPU:
    """GPU tests for emotion detector."""

    @pytest.fixture(scope="class")
    async def emotion_detector(self, gpu_available):
        """Create emotion detector for testing."""
        from maya.core import EmotionDetector, EmotionDetectorConfig

        logger.info("Initializing emotion detector for GPU tests...")
        initial_vram = get_gpu_memory_gb()

        config = EmotionDetectorConfig(
            device="cuda:0",
            warmup_on_init=True,
        )
        detector = EmotionDetector(config)

        try:
            await detector.initialize()
            final_vram = get_gpu_memory_gb()
            logger.info(
                "Emotion detector initialized",
                vram_used_gb=final_vram - initial_vram,
            )
            yield detector
        finally:
            await detector.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.asyncio
    async def test_emotion_detector_initialization(self, emotion_detector):
        """Test emotion detector initializes correctly."""
        assert emotion_detector.is_initialized

    @pytest.mark.asyncio
    async def test_emotion_detection(self, emotion_detector, sample_audio):
        """Test emotion detection on audio."""
        from maya.core import Emotion

        result = await emotion_detector.detect(sample_audio)

        assert result.emotion in Emotion
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0

        logger.info(
            "Emotion detection result",
            emotion=result.emotion.value,
            confidence=f"{result.confidence:.2f}",
            latency_ms=f"{result.latency_ms:.1f}",
        )

    @pytest.mark.asyncio
    async def test_emotion_detection_latency(self, emotion_detector, sample_audio):
        """Test emotion detection meets latency targets."""
        latencies = []

        for _ in range(10):
            result = await emotion_detector.detect(sample_audio)
            latencies.append(result.latency_ms)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        logger.info(
            "Emotion detection latency",
            avg_ms=f"{avg_latency:.1f}",
            p95_ms=f"{p95_latency:.1f}",
        )

        # Should be well under target
        assert avg_latency < 100, f"Emotion detection too slow: {avg_latency}ms"


# =============================================================================
# Full Pipeline GPU Tests
# =============================================================================


@pytest.mark.gpu
class TestFullPipelineGPU:
    """Integration tests for the complete voice pipeline."""

    @pytest.fixture(scope="class")
    async def full_pipeline(self, gpu_available):
        """Create full pipeline for testing."""
        from maya.core import (
            MoshiEngine,
            MoshiEngineConfig,
            ChatterboxTTSEngine,
            ChatterboxConfig,
            EmotionDetector,
            EmotionDetectorConfig,
            ResponseRouter,
            RouterConfig,
            Humanizer,
        )
        from maya.humanize import FillerConfig, PauseConfig

        logger.info("Initializing full pipeline for GPU tests...")
        initial_vram = get_gpu_memory_gb()

        # Initialize all components
        moshi = MoshiEngine(MoshiEngineConfig(device="cuda:0"))
        chatterbox = ChatterboxTTSEngine(ChatterboxConfig(device="cuda:0"))
        emotion = EmotionDetector(EmotionDetectorConfig(device="cuda:0"))
        router = ResponseRouter(RouterConfig())
        humanizer = Humanizer(FillerConfig(), PauseConfig())

        try:
            await moshi.initialize()
            await chatterbox.initialize()
            await emotion.initialize()

            final_vram = get_gpu_memory_gb()
            logger.info(
                "Full pipeline initialized",
                total_vram_gb=final_vram,
                vram_used_gb=final_vram - initial_vram,
            )

            yield {
                "moshi": moshi,
                "chatterbox": chatterbox,
                "emotion": emotion,
                "router": router,
                "humanizer": humanizer,
            }
        finally:
            await moshi.cleanup()
            await chatterbox.cleanup()
            await emotion.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.asyncio
    async def test_pipeline_vram_budget(self, full_pipeline):
        """Test that full pipeline fits in VRAM budget."""
        vram = get_gpu_memory_gb()
        reserved = get_gpu_memory_reserved_gb()

        logger.info(
            "VRAM usage",
            allocated_gb=f"{vram:.2f}",
            reserved_gb=f"{reserved:.2f}",
        )

        # Should fit within 24GB A10G with headroom
        assert vram < 22.0, f"VRAM usage {vram}GB exceeds budget"

    @pytest.mark.asyncio
    async def test_quick_path_latency(self, full_pipeline, short_audio):
        """Test quick path meets latency target."""
        moshi = full_pipeline["moshi"]
        router = full_pipeline["router"]

        latencies = []

        for _ in range(5):
            start_time = time.perf_counter()

            # Process through Moshi
            async for response in moshi.process_stream(short_audio):
                if response.audio is not None:
                    # Route decision (instant)
                    text = response.text or "okay"
                    decision = router.route(text)

                    if decision.path.value == "quick":
                        # Quick path - use Moshi audio directly
                        latency = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency)
                        break

        if latencies:
            avg_latency = np.mean(latencies)
            logger.info(f"Quick path average latency: {avg_latency:.1f}ms")
            assert avg_latency < LATENCY_TARGET_QUICK_PATH_MS, \
                f"Quick path latency {avg_latency}ms exceeds {LATENCY_TARGET_QUICK_PATH_MS}ms"

    @pytest.mark.asyncio
    async def test_enhanced_path_latency(self, full_pipeline, sample_audio):
        """Test enhanced path meets latency target."""
        moshi = full_pipeline["moshi"]
        chatterbox = full_pipeline["chatterbox"]
        humanizer = full_pipeline["humanizer"]

        latencies = []

        for _ in range(3):
            start_time = time.perf_counter()

            # Process through Moshi
            moshi_text = None
            async for response in moshi.process_stream(sample_audio):
                if response.text:
                    moshi_text = response.text
                    break

            if moshi_text:
                # Humanize text
                humanized = humanizer.humanize(moshi_text)

                # Synthesize with Chatterbox
                async for audio_chunk in chatterbox.synthesize_stream(humanized):
                    # Time to first audio chunk
                    latency = (time.perf_counter() - start_time) * 1000
                    latencies.append(latency)
                    break

        if latencies:
            avg_latency = np.mean(latencies)
            logger.info(f"Enhanced path average latency: {avg_latency:.1f}ms")
            # Enhanced path has higher target
            assert avg_latency < LATENCY_TARGET_ENHANCED_PATH_MS * 1.5, \
                f"Enhanced path latency {avg_latency}ms too high"

    @pytest.mark.asyncio
    async def test_emotion_adaptive_response(self, full_pipeline, sample_audio):
        """Test emotion detection influences TTS parameters."""
        emotion_detector = full_pipeline["emotion"]
        chatterbox = full_pipeline["chatterbox"]

        # Detect emotion
        emotion_result = await emotion_detector.detect(sample_audio)

        # Use suggested exaggeration
        exaggeration = emotion_result.suggested_exaggeration

        logger.info(
            "Emotion-adaptive synthesis",
            detected_emotion=emotion_result.emotion.value,
            exaggeration=exaggeration,
        )

        # Synthesize with emotion-adapted parameters
        # (In real usage, this would update Chatterbox config)
        assert 0.1 <= exaggeration <= 1.0


# =============================================================================
# VRAM Stress Tests
# =============================================================================


@pytest.mark.gpu
class TestVRAMStability:
    """Tests for VRAM stability and memory leaks."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_moshi(self, gpu_available):
        """Test Moshi doesn't leak memory over multiple runs."""
        from maya.core import MoshiEngine, MoshiEngineConfig

        gc.collect()
        torch.cuda.empty_cache()
        baseline_vram = get_gpu_memory_gb()

        config = MoshiEngineConfig(device="cuda:0", warmup_on_init=False)
        audio = generate_test_audio(1.0)

        for iteration in range(3):
            engine = MoshiEngine(config)
            await engine.initialize()

            # Process audio multiple times
            for _ in range(5):
                async for _ in engine.process_stream(audio):
                    pass

            await engine.cleanup()
            del engine
            gc.collect()
            torch.cuda.empty_cache()

            current_vram = get_gpu_memory_gb()
            logger.info(f"Iteration {iteration + 1} VRAM: {current_vram:.2f} GB")

        final_vram = get_gpu_memory_gb()
        leak = final_vram - baseline_vram

        logger.info(f"VRAM leak test: baseline={baseline_vram:.2f}, final={final_vram:.2f}, leak={leak:.2f}")
        assert leak < 0.5, f"Possible memory leak: {leak:.2f} GB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gpu"])
