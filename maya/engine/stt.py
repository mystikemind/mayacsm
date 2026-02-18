"""
Speech-to-Text Engine - Whisper

Transcribes user speech to text with high accuracy.

Features:
- Uses OpenAI Whisper (turbo model for speed)
- GPU acceleration
- Optimized for English
"""

import torch
import numpy as np
import logging
from typing import Optional, List
import time

from ..config import AUDIO

logger = logging.getLogger(__name__)


class STTEngine:
    """
    Whisper wrapper for speech transcription.

    Optimizations:
    - turbo model (fastest)
    - English-only mode
    - GPU acceleration
    """

    def __init__(self):
        self._model = None
        self._initialized = False

        # Performance tracking
        self._total_transcriptions = 0
        self._total_time = 0.0

    def initialize(self) -> None:
        """Load Whisper model."""
        if self._initialized:
            return

        logger.info("Loading Whisper turbo...")
        start = time.time()

        try:
            import whisper

            # Use turbo for speed, or large-v3 for accuracy
            self._model = whisper.load_model("turbo", device="cuda")

            elapsed = time.time() - start
            logger.info(f"Whisper loaded in {elapsed:.1f}s")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            # Try fallback to base model
            try:
                import whisper
                self._model = whisper.load_model("base", device="cuda")
                logger.info("Loaded Whisper base as fallback")
                self._initialized = True
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise

    def transcribe(self, audio: torch.Tensor) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio tensor at 24kHz

        Returns:
            Transcribed text
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        # Convert to numpy float32
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        # Ensure 1D
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        # Resample from 24kHz to 16kHz (Whisper's expected rate)
        # CRITICAL: Use scipy.signal.resample for proper anti-aliasing
        # Linear interpolation (np.interp) causes aliasing artifacts that hurt accuracy
        if AUDIO.sample_rate == 24000:
            from scipy import signal
            target_length = int(len(audio_np) * 16000 / 24000)
            # scipy.signal.resample applies proper low-pass filter before downsampling
            audio_np = signal.resample(audio_np, target_length).astype(np.float32)

        # Ensure float32 for Whisper
        audio_np = audio_np.astype(np.float32)

        # Transcribe
        result = self._model.transcribe(
            audio_np,
            language="en",
            fp16=True,
            verbose=False,
        )

        text = result["text"].strip()

        # Track performance
        elapsed = time.time() - start
        self._total_transcriptions += 1
        self._total_time += elapsed

        logger.debug(f"Transcribed in {elapsed*1000:.0f}ms: '{text[:50]}...'")

        return text

    def transcribe_stream(self, audio_chunks: List[torch.Tensor]) -> str:
        """
        Transcribe streaming audio chunks.

        Args:
            audio_chunks: List of audio tensors

        Returns:
            Transcribed text
        """
        # Concatenate chunks
        if not audio_chunks:
            return ""

        combined = torch.cat(audio_chunks, dim=0)
        return self.transcribe(combined)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def average_latency_ms(self) -> float:
        """Average transcription latency in milliseconds."""
        if self._total_transcriptions == 0:
            return 0.0
        return (self._total_time / self._total_transcriptions) * 1000

    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "total_transcriptions": self._total_transcriptions,
            "total_time": self._total_time,
            "average_latency_ms": self.average_latency_ms,
        }
