"""
Pytest configuration and fixtures for Project Maya tests.

This module provides shared fixtures for:
- Audio test data generation
- Configuration mocking
- Component initialization
- Async event loop handling
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Generator

import numpy as np
import pytest

from maya.config import Settings, get_settings
from maya.constants import DEFAULT_SAMPLE_RATE
from maya.utils.audio import AudioProcessor, generate_silence, generate_test_tone
from maya.utils.logging import setup_logging

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging() -> None:
    """Configure logging for test sessions."""
    setup_logging(level="DEBUG", json_output=False)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Get test settings with safe defaults.

    Returns settings configured for testing that won't affect production.
    """
    return Settings(
        environment="development",
        debug=True,
        log_level="DEBUG",
    )


@pytest.fixture(scope="session")
def audio_processor() -> AudioProcessor:
    """Get a shared audio processor instance."""
    return AudioProcessor(target_sample_rate=DEFAULT_SAMPLE_RATE)


# =============================================================================
# Audio Test Data Fixtures
# =============================================================================


@pytest.fixture
def silence_1s() -> NDArray[np.float32]:
    """Generate 1 second of silence at default sample rate."""
    return generate_silence(duration_seconds=1.0, sample_rate=DEFAULT_SAMPLE_RATE)


@pytest.fixture
def silence_100ms() -> NDArray[np.float32]:
    """Generate 100ms of silence."""
    return generate_silence(duration_seconds=0.1, sample_rate=DEFAULT_SAMPLE_RATE)


@pytest.fixture
def test_tone_440hz() -> NDArray[np.float32]:
    """Generate 1 second of 440Hz test tone."""
    return generate_test_tone(
        frequency=440.0,
        duration_seconds=1.0,
        sample_rate=DEFAULT_SAMPLE_RATE,
        amplitude=0.5,
    )


@pytest.fixture
def test_tone_1khz() -> NDArray[np.float32]:
    """Generate 1 second of 1kHz test tone."""
    return generate_test_tone(
        frequency=1000.0,
        duration_seconds=1.0,
        sample_rate=DEFAULT_SAMPLE_RATE,
        amplitude=0.5,
    )


@pytest.fixture
def random_audio_1s() -> NDArray[np.float32]:
    """Generate 1 second of random noise."""
    np.random.seed(42)
    return np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32) * 0.1


@pytest.fixture
def speech_like_audio() -> NDArray[np.float32]:
    """
    Generate audio that mimics speech characteristics.

    Creates audio with speech-like amplitude modulation and
    frequency characteristics for testing VAD and processing.
    """
    duration = 2.0
    sr = DEFAULT_SAMPLE_RATE
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Create speech-like signal with varying amplitude
    # Combine multiple frequencies (formants)
    f0 = 150  # Fundamental frequency
    audio = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * f0 * 2 * t)
        + 0.2 * np.sin(2 * np.pi * f0 * 3 * t)
    )

    # Add amplitude modulation (syllables)
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))  # 4 Hz modulation
    audio = audio * modulation

    # Add some noise
    noise = np.random.randn(len(t)) * 0.02
    audio = (audio + noise).astype(np.float32)

    # Normalize
    audio = audio / np.abs(audio).max() * 0.7

    return audio.astype(np.float32)


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def test_fixtures_dir() -> Path:
    """Get the path to test fixtures directory."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    return fixtures_dir


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """Get a temporary directory for audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """
    Create mock settings for testing.

    Use this when you need to override settings for specific tests.
    """
    settings = Settings(
        environment="development",
        debug=True,
        log_level="DEBUG",
        hf_token=None,  # Don't use real token in tests
    )

    # Clear the cache and set our mock
    get_settings.cache_clear()
    monkeypatch.setattr("maya.config.settings", settings)

    return settings


# =============================================================================
# Async Fixtures
# =============================================================================


@pytest.fixture
async def async_timeout() -> float:
    """Default timeout for async operations in tests."""
    return 30.0


# =============================================================================
# Marker Definitions
# =============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


# =============================================================================
# Skip Conditions
# =============================================================================


def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


skip_without_gpu = pytest.mark.skipif(
    not gpu_available(),
    reason="GPU not available",
)
