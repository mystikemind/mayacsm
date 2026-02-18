"""
Audio Enhancement Engine - Noise Reduction & Echo Suppression

Implements audio cleanup for better STT accuracy:
1. Spectral noise reduction (noisereduce)
2. Echo detection and suppression
3. Adaptive gain control

This is critical for voice call quality where:
- Echo from Maya's audio can be picked up by mic
- Background noise degrades STT accuracy
- Volume variations affect VAD detection
"""

import numpy as np
import torch
import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass
import noisereduce as nr

logger = logging.getLogger(__name__)


@dataclass
class AudioEnhancerConfig:
    """Configuration for audio enhancement."""
    # Noise reduction
    noise_reduce_enabled: bool = True
    noise_reduce_stationary: bool = False  # Non-stationary for voice
    noise_reduce_prop_decrease: float = 0.8  # Reduction amount

    # Echo detection
    echo_detect_enabled: bool = True
    echo_correlation_threshold: float = 0.5
    echo_window_ms: float = 100  # Window for correlation check

    # Gain control - SESAME LEVEL: Professional broadcast standards
    # -16 LUFS is standard for voice agents (approximated via RMS)
    # RMS of 0.15 ≈ -16.5 dBFS which approximates -16 LUFS for speech
    agc_enabled: bool = True
    agc_target_level: float = 0.15  # Target RMS (~-16 LUFS for speech)
    agc_max_gain: float = 4.0  # Maximum gain multiplier
    true_peak_limit: float = 0.89  # -1 dBTP ceiling


class AudioEnhancer:
    """
    Audio enhancement for voice calls.

    Features:
    1. Spectral noise reduction - removes background noise
    2. Echo detection - identifies when Maya's audio is in mic signal
    3. Adaptive gain control - normalizes volume

    Usage:
        enhancer = AudioEnhancer()
        clean_audio = enhancer.enhance(audio, maya_audio=reference)
    """

    def __init__(self, config: Optional[AudioEnhancerConfig] = None):
        self._config = config or AudioEnhancerConfig()
        self._initialized = False

        # Echo reference buffer
        self._maya_audio_buffer: Optional[np.ndarray] = None
        self._noise_profile: Optional[np.ndarray] = None

        # Stats
        self._total_processed = 0
        self._total_time_ms = 0.0
        self._echo_detections = 0

    def initialize(self) -> None:
        """Initialize the enhancer."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("INITIALIZING AUDIO ENHANCER")
        logger.info(f"Noise reduction: {self._config.noise_reduce_enabled}")
        logger.info(f"Echo detection: {self._config.echo_detect_enabled}")
        logger.info(f"AGC: {self._config.agc_enabled}")
        logger.info("=" * 60)

        self._initialized = True

    def set_maya_reference(self, audio: np.ndarray) -> None:
        """
        Set Maya's audio as reference for echo detection.

        Call this when Maya speaks to detect echo in user input.
        """
        if audio is not None and len(audio) > 0:
            self._maya_audio_buffer = audio.copy()
        else:
            self._maya_audio_buffer = None

    def clear_maya_reference(self) -> None:
        """Clear Maya's audio reference."""
        self._maya_audio_buffer = None

    def _detect_echo(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[bool, float]:
        """
        Detect if Maya's audio is present in input (echo).

        Uses cross-correlation to find echo.
        """
        if self._maya_audio_buffer is None or len(self._maya_audio_buffer) == 0:
            return False, 0.0

        # Use a window of Maya's recent audio
        window_samples = int(self._config.echo_window_ms * sample_rate / 1000)
        ref_window = self._maya_audio_buffer[-window_samples:] if len(self._maya_audio_buffer) > window_samples else self._maya_audio_buffer

        # Normalize
        ref_norm = ref_window / (np.linalg.norm(ref_window) + 1e-8)
        audio_norm = audio / (np.linalg.norm(audio) + 1e-8)

        # Cross-correlation
        min_len = min(len(ref_norm), len(audio_norm))
        if min_len < 100:
            return False, 0.0

        correlation = np.correlate(
            audio_norm[:min_len],
            ref_norm[:min_len],
            mode='valid'
        )

        max_corr = np.abs(correlation).max() if len(correlation) > 0 else 0.0

        is_echo = max_corr > self._config.echo_correlation_threshold

        if is_echo:
            logger.debug(f"Echo detected: correlation={max_corr:.3f}")
            self._echo_detections += 1

        return is_echo, max_corr

    def _reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Apply spectral noise reduction."""
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=self._config.noise_reduce_stationary,
                prop_decrease=self._config.noise_reduce_prop_decrease,
                n_jobs=1  # Single-threaded for low latency
            )
            return reduced
        except Exception as e:
            logger.debug(f"Noise reduction failed: {e}")
            return audio

    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """Apply adaptive gain control with true peak limiting.

        SESAME-LEVEL: Use professional broadcast standards:
        - Target -16 LUFS (approximated via RMS for speech)
        - True peak limit at -1 dBTP (0.89 linear)
        - Soft limiting to prevent harsh clipping
        """
        # Remove DC offset first
        audio = audio - np.mean(audio)

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio

        # Calculate gain needed to reach target level
        gain = self._config.agc_target_level / rms
        gain = min(gain, self._config.agc_max_gain)
        gain = max(gain, 0.1)  # Don't attenuate too much

        audio = audio * gain

        # True peak limiting (soft knee limiter)
        peak = np.max(np.abs(audio))
        if peak > self._config.true_peak_limit:
            # Soft limiting: compress peaks above threshold
            threshold = self._config.true_peak_limit * 0.9
            mask = np.abs(audio) > threshold
            if np.any(mask):
                over_threshold = np.abs(audio) - threshold
                limit_headroom = self._config.true_peak_limit - threshold
                compressed = threshold + np.tanh(over_threshold / limit_headroom) * limit_headroom
                audio = np.where(mask, np.sign(audio) * compressed, audio)

        return audio

    def enhance(
        self,
        audio: torch.Tensor,
        sample_rate: int = 24000,
        check_echo: bool = True
    ) -> Tuple[torch.Tensor, bool]:
        """
        Enhance audio quality.

        Args:
            audio: Input audio tensor
            sample_rate: Sample rate
            check_echo: Whether to check for echo

        Returns:
            Tuple of (enhanced_audio, is_echo_detected)
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        is_echo = False

        # Echo detection (before noise reduction)
        if self._config.echo_detect_enabled and check_echo:
            is_echo, _ = self._detect_echo(audio_np, sample_rate)

        # Skip enhancement if echo detected (will be filtered by pipeline)
        if is_echo:
            elapsed = (time.time() - start) * 1000
            self._total_processed += 1
            self._total_time_ms += elapsed
            return torch.tensor(audio_np, dtype=torch.float32), True

        # Noise reduction
        if self._config.noise_reduce_enabled:
            audio_np = self._reduce_noise(audio_np, sample_rate)

        # Adaptive gain control
        if self._config.agc_enabled:
            audio_np = self._apply_agc(audio_np)

        # Clip to valid range
        audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)

        elapsed = (time.time() - start) * 1000
        self._total_processed += 1
        self._total_time_ms += elapsed

        logger.debug(f"Enhanced audio in {elapsed:.1f}ms")

        return torch.tensor(audio_np, dtype=torch.float32), is_echo

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def average_latency_ms(self) -> float:
        if self._total_processed == 0:
            return 0.0
        return self._total_time_ms / self._total_processed

    def get_stats(self) -> dict:
        return {
            "total_processed": self._total_processed,
            "average_latency_ms": self.average_latency_ms,
            "echo_detections": self._echo_detections,
        }
