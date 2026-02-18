"""
Local STT Engine - faster-whisper with CUDA.

Uses faster-whisper (CTranslate2) directly for ultra-low latency.
No HTTP overhead = ~85ms vs ~237ms with Docker.

Target: ~85ms for 1.5s audio (Sesame-level)

Requires: ctranslate2 3.x (for cuDNN 8 compatibility)
CUDA library symlinks may be needed for cuBLAS 11/12 compatibility.
"""

import os
import torch
import numpy as np
import logging
import time
from typing import Optional
from faster_whisper import WhisperModel

from ..config import AUDIO

# Ensure CUDA libraries are found (cuBLAS in cuda-12.1, cuDNN in cuda-12.2)
cuda_paths = [
    "/usr/local/cuda-12.1/lib64",  # cuBLAS 12
    "/usr/local/cuda-12.2/lib",    # cuDNN 8
    "/usr/local/cuda/lib64",
]
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
for path in cuda_paths:
    if os.path.exists(path) and path not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

logger = logging.getLogger(__name__)


class LocalSTTEngine:
    """
    Ultra-fast STT using faster-whisper locally.

    Architecture:
        Audio → faster-whisper (CTranslate2) → Transcript

    Benefits over Docker:
    - No HTTP overhead (~100-150ms saved)
    - No serialization/deserialization
    - Direct GPU access
    - Lower latency: ~80-100ms vs ~250ms

    Model options:
    - tiny.en: ~50ms, lower accuracy
    - base.en: ~70ms, good accuracy
    - small.en: ~100ms, better accuracy (default)
    - medium.en: ~200ms, best accuracy
    """

    # Model selection - base.en for quality/latency balance
    # Options: tiny.en (~70ms, 10% WER), base.en (~100ms, 5% WER), small.en (~130ms, 4% WER)
    # Using base.en: 2x better accuracy than tiny.en with only +30ms latency
    # This is critical for conversation quality - mishearing destroys UX
    MODEL_SIZE = "base.en"
    COMPUTE_TYPE = "float16"  # Best for CUDA

    def __init__(self):
        self._model: Optional[WhisperModel] = None
        self._initialized = False
        self._total_transcriptions = 0
        self._total_time = 0.0

    def initialize(self) -> None:
        """Load faster-whisper model."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("LOADING LOCAL FASTER-WHISPER")
        logger.info(f"Model: {self.MODEL_SIZE}")
        logger.info(f"Compute type: {self.COMPUTE_TYPE}")
        logger.info("Target: ~80-100ms per transcription")
        logger.info("=" * 60)

        start = time.time()

        # Load model with CUDA acceleration (ctranslate2 3.x + cuDNN 8)
        self._model = WhisperModel(
            self.MODEL_SIZE,
            device="cuda",
            compute_type="float16",  # Best for GPU
            download_root="/home/ec2-user/SageMaker/project_maya/models/whisper"
        )
        logger.info("Using CUDA with float16")

        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.1f}s")

        # Warmup
        logger.info("Warming up...")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second
        for i in range(3):
            start = time.time()
            try:
                segments, _ = self._model.transcribe(
                    dummy_audio,
                    language="en",
                    beam_size=1,
                    vad_filter=False
                )
                _ = list(segments)  # Consume generator
                elapsed = (time.time() - start) * 1000
                logger.info(f"  Warmup {i+1}/3: {elapsed:.0f}ms")
            except Exception as e:
                logger.warning(f"  Warmup {i+1}/3 failed: {e}")

        logger.info("=" * 60)
        logger.info("LOCAL STT READY")
        logger.info("=" * 60)

        self._initialized = True

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

        # Resample from 24kHz to 16kHz (Whisper native rate)
        if AUDIO.sample_rate == 24000:
            target_length = int(len(audio_np) * 16000 / 24000)
            indices = np.linspace(0, len(audio_np) - 1, target_length)
            audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np).astype(np.float32)

        # Transcribe with optimized settings + hallucination prevention
        segments, info = self._model.transcribe(
            audio_np,
            language="en",
            beam_size=1,  # Greedy decoding for speed
            vad_filter=True,  # Enable VAD to filter silence/noise
            word_timestamps=False,  # Don't need word timing
            condition_on_previous_text=False,  # Faster without context
            no_speech_threshold=0.6,  # Higher = stricter no-speech detection
            log_prob_threshold=-1.0,  # Filter low-confidence outputs
            compression_ratio_threshold=2.4,  # Detect repetitive hallucinations
        )

        # Collect text from segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()

        # Track stats
        elapsed = time.time() - start
        self._total_transcriptions += 1
        self._total_time += elapsed

        logger.debug(f"Transcribed in {elapsed*1000:.0f}ms: '{text[:50]}...'")

        # Clear CUDA cache periodically to prevent memory fragmentation
        if self._total_transcriptions % 10 == 0:
            torch.cuda.empty_cache()

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
            "model": self.MODEL_SIZE,
        }
