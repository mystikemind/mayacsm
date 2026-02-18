"""
Integration tests for Project Maya pipeline components.

These tests verify that all components work together correctly.
They can run without GPU by mocking the heavy models.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from maya.config import settings
from maya.constants import DEFAULT_SAMPLE_RATE


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate sample audio data for testing."""
    duration_seconds = 2.0
    t = np.linspace(0, duration_seconds, int(DEFAULT_SAMPLE_RATE * duration_seconds))
    # Generate a simple tone with some variation
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    return audio.astype(np.float32)


@pytest.fixture
def short_audio() -> np.ndarray:
    """Generate short audio clip."""
    duration_seconds = 0.5
    t = np.linspace(0, duration_seconds, int(DEFAULT_SAMPLE_RATE * duration_seconds))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


# =============================================================================
# Component Import Tests
# =============================================================================


class TestComponentImports:
    """Test that all components can be imported."""

    def test_import_core_components(self):
        """Test core component imports."""
        from maya.core import (
            MoshiEngine,
            MoshiEngineConfig,
            MoshiResponse,
            MoshiState,
            ChatterboxTTSEngine,
            ChatterboxConfig,
            ResponseRouter,
            RouterConfig,
            RouteDecision,
            ResponsePath,
            VoicePipeline,
            PipelineConfig,
            EmotionDetector,
            EmotionDetectorConfig,
            Emotion,
        )

        assert MoshiEngine is not None
        assert ChatterboxTTSEngine is not None
        assert ResponseRouter is not None
        assert VoicePipeline is not None
        assert EmotionDetector is not None

    def test_import_humanize_components(self):
        """Test humanize component imports."""
        from maya.humanize import (
            FillerInjector,
            FillerConfig,
            PauseInjector,
            PauseConfig,
            BreathSynthesizer,
            BreathConfig,
        )

        assert FillerInjector is not None
        assert PauseInjector is not None
        assert BreathSynthesizer is not None

    def test_import_transport_components(self):
        """Test transport component imports."""
        from maya.transport import (
            WebSocketServer,
            WebSocketConfig,
            AudioCodec,
            CodecConfig,
            AudioFormat,
        )

        assert WebSocketServer is not None
        assert AudioCodec is not None
        assert AudioFormat is not None

    def test_import_session_components(self):
        """Test session component imports."""
        from maya.session import (
            SessionManager,
            SessionConfig,
            Session,
            SessionState,
        )

        assert SessionManager is not None
        assert Session is not None

    def test_import_api_components(self):
        """Test API component imports."""
        from maya.api import (
            app,
            HealthChecker,
            HealthStatus,
        )

        assert app is not None
        assert HealthChecker is not None


# =============================================================================
# Response Router Integration Tests
# =============================================================================


class TestResponseRouterIntegration:
    """Test ResponseRouter with realistic scenarios."""

    def test_router_with_various_inputs(self):
        """Test router handles various input types."""
        from maya.core import ResponseRouter, RouterConfig, ResponsePath

        router = ResponseRouter(RouterConfig())

        test_cases = [
            # (input, expected_path)
            ("uh-huh", ResponsePath.QUICK),
            ("okay", ResponsePath.QUICK),
            ("That's a very good point you're making.", ResponsePath.ENHANCED),
            ("I love that idea!", ResponsePath.ENHANCED),
            ("Let me think about that...", ResponsePath.ENHANCED),
        ]

        for text, expected_path in test_cases:
            result = router.route(text)
            assert result.path == expected_path, f"Failed for: {text}"

    def test_router_with_emotion_context(self):
        """Test router considers emotion context."""
        from maya.core import ResponseRouter, RouterConfig, ResponsePath

        router = ResponseRouter(RouterConfig())

        # Test that emotion metadata is captured
        text = "okay"

        result_neutral = router.route(text, user_emotion="neutral")
        result_angry = router.route(text, user_emotion="angry")

        # Both route to quick (it's a quick acknowledgment)
        assert result_neutral.path == ResponsePath.QUICK
        assert result_angry.path == ResponsePath.QUICK

        # But metadata should capture the emotion
        assert result_neutral.metadata.get("user_emotion") == "neutral"
        assert result_angry.metadata.get("user_emotion") == "angry"

    def test_router_batch_processing(self):
        """Test router can process batches efficiently."""
        from maya.core import ResponseRouter, RouterConfig

        router = ResponseRouter(RouterConfig())

        texts = [
            "yes",
            "no",
            "Tell me more about that please",
            "uh-huh",
            "What do you think about this topic?",
        ]

        results = router.route_batch(texts)
        assert len(results) == len(texts)

        # First, second, and fourth should be quick (short acknowledgments)
        assert results[0].path.value == "quick"
        assert results[1].path.value == "quick"
        assert results[3].path.value == "quick"
        # Third and fifth should be enhanced (longer)
        assert results[2].path.value == "enhanced"
        assert results[4].path.value == "enhanced"


# =============================================================================
# Humanizer Integration Tests
# =============================================================================


class TestHumanizerIntegration:
    """Test humanizer components work together."""

    def test_full_humanization_pipeline(self):
        """Test filler + pause injection pipeline."""
        from maya.humanize import FillerInjector, PauseInjector, FillerConfig, PauseConfig

        filler = FillerInjector(FillerConfig(
            sentence_start_probability=1.0,  # Always add filler for testing
        ))
        pause = PauseInjector(PauseConfig())

        original_text = "Hello. How are you doing today?"

        # Apply filler injection
        filler_result = filler.inject(original_text)

        # Apply pause injection
        pause_result = pause.inject(filler_result.text)

        # Should have added fillers and pauses
        assert len(pause_result.text) > len(original_text)
        # Should have pause markers
        assert "<pause" in pause_result.text or pause_result.pause_count > 0

    @pytest.mark.asyncio
    async def test_breath_synthesizer_initialization(self):
        """Test breath synthesizer can be initialized."""
        from maya.humanize import BreathSynthesizer, BreathConfig

        synth = BreathSynthesizer(BreathConfig())
        await synth.initialize()

        assert synth.is_initialized

        # Get a breath sample
        breath = synth.get_breath()
        assert breath is not None
        assert len(breath) > 0
        assert breath.dtype == np.float32

        # Note: BreathSynthesizer doesn't have cleanup, it's a lightweight component


# =============================================================================
# Audio Codec Integration Tests
# =============================================================================


class TestAudioCodecIntegration:
    """Test audio codec functionality."""

    def test_codec_roundtrip_pcm16(self, sample_audio):
        """Test PCM16 encode/decode roundtrip."""
        from maya.transport import AudioCodec, CodecConfig, AudioFormat

        codec = AudioCodec(CodecConfig(format=AudioFormat.PCM16))

        encoded = codec.encode(sample_audio)
        assert isinstance(encoded, bytes)

        decoded = codec.decode(encoded)
        assert decoded.dtype == np.float32

        # Should be approximately equal (some quantization loss)
        np.testing.assert_allclose(sample_audio, decoded, rtol=1e-3, atol=1e-3)

    def test_codec_roundtrip_pcm32(self, sample_audio):
        """Test PCM32 encode/decode roundtrip."""
        from maya.transport import AudioCodec, CodecConfig, AudioFormat

        codec = AudioCodec(CodecConfig(format=AudioFormat.PCM32))

        encoded = codec.encode(sample_audio, format=AudioFormat.PCM32)
        decoded = codec.decode(encoded, format=AudioFormat.PCM32)

        # PCM32 should be very close to original
        np.testing.assert_allclose(sample_audio, decoded, rtol=1e-6, atol=1e-6)


# =============================================================================
# Session Manager Integration Tests
# =============================================================================


class TestSessionManagerIntegration:
    """Test session manager functionality."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test full session lifecycle."""
        from maya.session import SessionManager, SessionConfig, SessionState

        manager = SessionManager(SessionConfig(
            idle_timeout_seconds=60,
        ))
        await manager.start()

        try:
            # Create session
            session = await manager.create_session("test-conn-1")
            assert session is not None
            # Session starts in CREATED state
            assert session.state in (SessionState.CREATED, SessionState.ACTIVE)

            # Get session (sync method)
            retrieved = manager.get_session(session.session_id)
            assert retrieved is not None
            assert retrieved.session_id == session.session_id

            # Add turn (sync method)
            session.add_turn("user", "Hello")
            session.add_turn("assistant", "Hi there!")

            # Check conversation history
            retrieved = manager.get_session(session.session_id)
            assert len(retrieved.conversation_history) == 2

            # End session
            await manager.end_session(session.session_id)
            ended = manager.get_session(session.session_id)
            assert ended is None or ended.state == SessionState.ENDED

        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        """Test managing multiple concurrent sessions."""
        from maya.session import SessionManager, SessionConfig

        manager = SessionManager(SessionConfig())
        await manager.start()

        try:
            sessions = []
            for i in range(3):
                session = await manager.create_session(f"conn-{i}")
                sessions.append(session)

            assert len(sessions) == 3
            assert len(set(s.session_id for s in sessions)) == 3  # All unique

            # All should exist (sync method)
            for session in sessions:
                retrieved = manager.get_session(session.session_id)
                assert retrieved is not None

        finally:
            await manager.stop()


# =============================================================================
# Full Pipeline Integration Tests (Mocked)
# =============================================================================


class TestPipelineIntegration:
    """Test pipeline integration with mocked heavy components."""

    @pytest.mark.asyncio
    async def test_pipeline_config_creation(self):
        """Test pipeline configuration."""
        from maya.core import PipelineConfig

        config = PipelineConfig(
            enable_humanizer=True,
            enable_quick_path=True,
        )

        assert config.enable_humanizer is True
        assert config.enable_quick_path is True

    def test_humanizer_class(self):
        """Test Humanizer helper class."""
        from maya.core import Humanizer
        from maya.humanize import FillerConfig, PauseConfig

        humanizer = Humanizer(
            filler_config=FillerConfig(),
            pause_config=PauseConfig(),
        )

        # Test humanization
        result = humanizer.humanize("Hello, how are you today?")
        assert result is not None
        assert len(result) > 0


# =============================================================================
# Emotion Detector Tests (Config Only)
# =============================================================================


class TestEmotionDetectorConfig:
    """Test emotion detector configuration."""

    def test_emotion_enum(self):
        """Test Emotion enum values."""
        from maya.core import Emotion

        assert Emotion.NEUTRAL.value == "neutral"
        assert Emotion.HAPPY.value == "happy"
        assert Emotion.SAD.value == "sad"
        assert Emotion.ANGRY.value == "angry"

        # Test from_label
        assert Emotion.from_label("happy") == Emotion.HAPPY
        assert Emotion.from_label("fear") == Emotion.FEARFUL
        assert Emotion.from_label("surprise") == Emotion.SURPRISED
        assert Emotion.from_label("unknown") == Emotion.NEUTRAL

    def test_emotion_result_properties(self):
        """Test EmotionResult properties."""
        from maya.core import EmotionResult, Emotion

        result = EmotionResult(
            emotion=Emotion.HAPPY,
            confidence=0.85,
            all_scores={Emotion.HAPPY: 0.85, Emotion.NEUTRAL: 0.15},
            latency_ms=45.0,
            suggested_exaggeration=0.7,
        )

        assert result.is_confident is True
        assert result.is_positive is True
        assert result.is_negative is False

        # Test negative emotion
        sad_result = EmotionResult(emotion=Emotion.SAD, confidence=0.6)
        assert sad_result.is_negative is True
        assert sad_result.is_positive is False


# =============================================================================
# Configuration Integration Tests
# =============================================================================


class TestConfigurationIntegration:
    """Test configuration loading and validation."""

    def test_settings_load(self):
        """Test settings can be loaded."""
        from maya.config import settings, Settings

        assert settings is not None
        assert isinstance(settings, Settings)

    def test_nested_config_access(self):
        """Test nested configuration access."""
        from maya.config import settings

        # Access nested configs
        assert settings.moshi.sample_rate == 24000
        assert settings.server.http_port == 8080
        assert settings.humanize.enable_fillers is True

    def test_config_defaults(self):
        """Test configuration defaults are sensible."""
        from maya.config import settings
        from maya.constants import (
            DEFAULT_SAMPLE_RATE,
            CHATTERBOX_EXAGGERATION_DEFAULT,
        )

        assert settings.moshi.sample_rate == DEFAULT_SAMPLE_RATE
        assert settings.chatterbox.exaggeration == CHATTERBOX_EXAGGERATION_DEFAULT


# =============================================================================
# Metrics Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Test metrics collection."""

    def test_metrics_available(self):
        """Test metrics singleton is available."""
        from maya.utils.metrics import metrics

        assert metrics is not None
        assert hasattr(metrics, "latency_seconds")
        assert hasattr(metrics, "model_load_seconds")
        assert hasattr(metrics, "routing_decisions")

    def test_record_error(self):
        """Test error recording."""
        from maya.utils.metrics import metrics

        # Should not raise
        metrics.record_error("test_component", "TestError")


# =============================================================================
# Logging Integration Tests
# =============================================================================


class TestLoggingIntegration:
    """Test logging configuration."""

    def test_logger_creation(self):
        """Test logger can be created."""
        from maya.utils.logging import get_logger

        logger = get_logger("test.integration")
        assert logger is not None

        # Should not raise
        logger.info("Test message", test_key="test_value")

    def test_logger_mixin(self):
        """Test LoggerMixin functionality."""
        from maya.utils.logging import LoggerMixin

        class TestClass(LoggerMixin):
            def do_something(self):
                self.logger.info("Doing something")
                return True

        obj = TestClass()
        assert hasattr(obj, "logger")
        assert obj.do_something() is True


# =============================================================================
# End-to-End Flow Tests (Mocked)
# =============================================================================


class TestEndToEndFlow:
    """Test complete flow with mocked components."""

    @pytest.mark.asyncio
    async def test_audio_to_routing_flow(self, sample_audio):
        """Test flow from audio input to routing decision."""
        from maya.core import ResponseRouter, RouterConfig, ResponsePath

        # In real usage, Moshi would process audio and output text
        # Here we simulate that output
        simulated_moshi_output = "I understand what you're saying."

        # Route the response
        router = ResponseRouter(RouterConfig())
        decision = router.route(simulated_moshi_output)

        assert decision.path == ResponsePath.ENHANCED
        assert decision.reason is not None

    @pytest.mark.asyncio
    async def test_text_humanization_flow(self):
        """Test text humanization flow."""
        from maya.humanize import FillerInjector, PauseInjector, FillerConfig, PauseConfig

        # Original response text
        text = "That's a great question. Let me explain how this works."

        # Apply humanization
        filler = FillerInjector(FillerConfig())
        pause = PauseInjector(PauseConfig())

        result = filler.inject(text)
        result = pause.inject(result.text)

        # Should have some modifications
        assert result.text is not None
        assert len(result.text) >= len(text)  # At least same length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
