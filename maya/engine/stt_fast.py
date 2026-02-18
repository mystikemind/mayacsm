"""
Fast STT Engine - faster-whisper via Docker.

Uses faster-whisper (CTranslate2) which is 2-4x faster than OpenAI Whisper.
Runs in Docker to bypass dependency conflicts.

Target: ~50-70ms (vs 125ms with OpenAI Whisper)
"""

import torch
import numpy as np
import logging
import time
import requests
import io
import wave
from typing import Optional

from ..config import AUDIO

logger = logging.getLogger(__name__)


class FastSTTEngine:
    """
    Ultra-fast STT using faster-whisper in Docker.

    Architecture:
        Audio → HTTP POST → faster-whisper Docker → Transcript

    Benchmark: ~50-70ms (vs 125ms with OpenAI Whisper)
    """

    WHISPER_URL = "http://localhost:8002"
    # small.en: Good accuracy, reasonable speed (~150ms after warmup)
    # Options: tiny.en (~200ms), base.en (~200ms), small.en (~150ms), medium.en (~250ms)
    MODEL = "Systran/faster-whisper-small.en"

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._initialized = False
        self._total_transcriptions = 0
        self._total_time = 0.0

    def _create_session(self) -> requests.Session:
        """Create HTTP session with connection pooling."""
        session = requests.Session()
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(total=2, backoff_factor=0.1)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
        session.mount("http://", adapter)
        return session

    def _check_health(self) -> bool:
        """Check if faster-whisper server is healthy."""
        try:
            resp = self._session.get(f"{self.WHISPER_URL}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def initialize(self) -> None:
        """Initialize connection to faster-whisper server."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("CONNECTING TO FASTER-WHISPER SERVER")
        logger.info(f"URL: {self.WHISPER_URL}")
        logger.info("Target: ~50-70ms per transcription")
        logger.info("=" * 60)

        self._session = self._create_session()

        # Check health
        if not self._check_health():
            logger.warning("faster-whisper server not responding")
        else:
            logger.info("faster-whisper server is healthy")

        # Warmup
        logger.info("Warming up...")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        for i in range(2):
            start = time.time()
            try:
                _ = self._transcribe_internal(dummy_audio)
                elapsed = (time.time() - start) * 1000
                logger.info(f"  Warmup {i+1}/2: {elapsed:.0f}ms")
            except Exception as e:
                logger.warning(f"  Warmup {i+1}/2 failed: {e}")

        logger.info("=" * 60)
        logger.info("FAST STT READY")
        logger.info("=" * 60)

        self._initialized = True

    def _audio_to_wav_bytes(self, audio_np: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert numpy audio to WAV bytes for API."""
        # Ensure float32 and normalize
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        # Convert to int16 for WAV
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        buffer.seek(0)
        return buffer.read()

    def _transcribe_internal(self, audio_np: np.ndarray) -> str:
        """Internal transcription via HTTP."""
        # Convert to WAV bytes
        wav_bytes = self._audio_to_wav_bytes(audio_np, sample_rate=16000)

        # Call API
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "language": "en",
            "response_format": "json",
        }
        # Only specify model if set (otherwise use server default)
        if self.MODEL:
            data["model"] = self.MODEL

        try:
            resp = self._session.post(
                f"{self.WHISPER_URL}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=10.0
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

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

        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        # Resample from 24kHz to 16kHz
        if AUDIO.sample_rate == 24000:
            target_length = int(len(audio_np) * 16000 / 24000)
            indices = np.linspace(0, len(audio_np) - 1, target_length)
            audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np).astype(np.float32)

        # Transcribe
        text = self._transcribe_internal(audio_np)

        # Track stats
        elapsed = time.time() - start
        self._total_transcriptions += 1
        self._total_time += elapsed

        logger.debug(f"Transcribed in {elapsed*1000:.0f}ms: '{text[:50]}...'")

        return text

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def average_latency_ms(self) -> float:
        if self._total_transcriptions == 0:
            return 0.0
        return (self._total_time / self._total_transcriptions) * 1000

    def get_stats(self) -> dict:
        return {
            "total_transcriptions": self._total_transcriptions,
            "average_latency_ms": self.average_latency_ms,
        }
