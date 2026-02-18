"""
Unit tests for MoshiEngine.

Tests cover:
- Configuration and initialization
- State management
- Audio processing
- Context reset
- Error handling
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from maya.core.moshi_engine import (
    MoshiEngine,
    MoshiEngineConfig,
    MoshiResponse,
    MoshiState,
    create_moshi_engine,
)
from maya.constants import MOSHI_SAMPLE_RATE


class TestMoshiEngineConfig:
    """Tests for MoshiEngineConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MoshiEngineConfig()

        assert config.device == "cuda:0"
        assert config.dtype == torch.bfloat16
        assert config.use_sampling is True
        assert config.temperature == 0.8
        assert config.temperature_text == 0.7
        assert config.warmup_on_init is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MoshiEngineConfig(
            device="cuda:1",
            temperature=0.5,
            warmup_on_init=False,
        )

        assert config.device == "cuda:1"
        assert config.temperature == 0.5
        assert config.warmup_on_init is False


class TestMoshiResponse:
    """Tests for MoshiResponse dataclass."""

    def test_empty_response(self) -> None:
        """Test response with default values."""
        response = MoshiResponse()

        assert response.audio is None
        assert response.text is None
        assert response.is_speaking is False
        assert response.latency_ms == 0.0

    def test_audio_response(self) -> None:
        """Test response with audio data."""
        audio = np.random.randn(1920).astype(np.float32)
        response = MoshiResponse(
            audio=audio,
            is_speaking=True,
            latency_ms=150.0,
        )

        assert response.audio is not None
        assert len(response.audio) == 1920
        assert response.is_speaking is True
        assert response.latency_ms == 150.0

    def test_text_response(self) -> None:
        """Test response with text."""
        response = MoshiResponse(
            text="Hello, how can I help you?",
            is_speaking=True,
        )

        assert response.text == "Hello, how can I help you?"


class TestMoshiEngineState:
    """Tests for MoshiEngine state management."""

    def test_initial_state(self) -> None:
        """Test engine starts in uninitialized state."""
        engine = MoshiEngine()

        assert engine.state == MoshiState.UNINITIALIZED
        assert not engine.is_initialized

    def test_state_properties(self) -> None:
        """Test state-related properties."""
        engine = MoshiEngine()

        assert engine.context_elapsed_seconds == 0.0
        assert not engine.should_reset_context


class TestMoshiEngineWithMocks:
    """Tests using mocked models."""

    @pytest.fixture
    def mock_moshi_modules(self) -> MagicMock:
        """Create mock Moshi modules."""
        with patch("maya.core.moshi_engine.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = 10 * 1024**3  # 10GB

            yield mock_torch

    @pytest.mark.asyncio
    async def test_cleanup(self) -> None:
        """Test cleanup releases resources."""
        engine = MoshiEngine()
        engine._state = MoshiState.READY
        engine._mimi = MagicMock()
        engine._lm_model = MagicMock()
        engine._lm_gen = MagicMock()

        await engine.cleanup()

        assert engine.state == MoshiState.UNINITIALIZED
        assert engine._mimi is None
        assert engine._lm_model is None
        assert engine._lm_gen is None

    @pytest.mark.asyncio
    async def test_reset_context(self) -> None:
        """Test context reset clears state."""
        engine = MoshiEngine()
        engine._state = MoshiState.READY
        engine._total_frames_processed = 100
        engine._accumulated_text = ["some", "text"]
        engine._mimi = MagicMock()
        engine._lm_gen = MagicMock()

        await engine.reset_context()

        assert engine._total_frames_processed == 0
        assert len(engine._accumulated_text) == 0

    @pytest.mark.asyncio
    async def test_process_stream_not_initialized(self) -> None:
        """Test process_stream raises when not initialized."""
        engine = MoshiEngine()
        audio = np.zeros(MOSHI_SAMPLE_RATE, dtype=np.float32)

        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in engine.process_stream(audio):
                pass

    def test_get_stats(self) -> None:
        """Test stats dictionary structure."""
        engine = MoshiEngine()
        stats = engine.get_stats()

        assert "state" in stats
        assert "context_elapsed_seconds" in stats
        assert "total_frames_processed" in stats
        assert "vram_usage_gb" in stats
        assert "should_reset_context" in stats

        assert stats["state"] == "uninitialized"
        assert stats["total_frames_processed"] == 0


class TestMoshiEngineConfig:
    """Test configuration edge cases."""

    def test_config_with_cpu_device(self) -> None:
        """Test configuration with CPU device."""
        config = MoshiEngineConfig(device="cpu")
        engine = MoshiEngine(config)

        assert engine.config.device == "cpu"

    def test_config_defaults_from_settings(self) -> None:
        """Test that config can use values from settings."""
        config = MoshiEngineConfig()

        # Should have sensible defaults
        assert config.max_context_seconds > 0
        assert 0 < config.temperature <= 1.0


@pytest.mark.gpu
@pytest.mark.slow
class TestMoshiEngineIntegration:
    """
    Integration tests requiring GPU.

    These tests actually load models and run inference.
    Skip with: pytest -m "not gpu"
    """

    @pytest.fixture
    async def engine(self) -> MoshiEngine:
        """Create and initialize engine for testing."""
        config = MoshiEngineConfig(
            warmup_on_init=False,  # Skip warmup for faster tests
        )
        engine = MoshiEngine(config)

        # Only run if GPU available
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        return engine

    @pytest.mark.asyncio
    async def test_full_initialization(self, engine: MoshiEngine) -> None:
        """Test full model initialization."""
        await engine.initialize()

        assert engine.is_initialized
        assert engine.state == MoshiState.READY
        assert engine._mimi is not None
        assert engine._lm_model is not None
        assert engine._lm_gen is not None

        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_process_silence(self, engine: MoshiEngine) -> None:
        """Test processing silence."""
        await engine.initialize()

        # Create 1 second of silence
        audio = np.zeros(MOSHI_SAMPLE_RATE, dtype=np.float32)

        responses = []
        async for response in engine.process_stream(audio):
            responses.append(response)

        assert len(responses) > 0

        await engine.cleanup()


class TestCreateMoshiEngineContextManager:
    """Tests for the context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_error(self) -> None:
        """Test that context manager cleans up on error."""
        engine = MoshiEngine()

        # Mock initialize to succeed
        engine.initialize = AsyncMock()
        engine.cleanup = AsyncMock()

        with patch(
            "maya.core.moshi_engine.MoshiEngine",
            return_value=engine,
        ):
            try:
                async with create_moshi_engine() as e:
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Cleanup should still be called
            engine.cleanup.assert_called_once()
