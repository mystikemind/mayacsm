"""
Local Faster-Whisper STT Engine - CTranslate2 backend for maximum speed.

Uses faster-whisper directly (no Docker) with CTranslate2 which is 2-4x faster
than OpenAI's Whisper implementation.

Target: ~100ms per 1-2s transcription
"""

import torch
import torchaudio
import numpy as np
import logging
import time
from typing import Optional

from ..config import AUDIO, STT

logger = logging.getLogger(__name__)

# Pre-create resampler for 24kHz -> 16kHz (Whisper's native rate)
# Using torchaudio is faster than scipy for resampling
_resampler_24k_to_16k = None


class FasterSTTEngine:
    """
    Local faster-whisper STT using CTranslate2 backend.

    Key advantages over OpenAI Whisper:
    - CTranslate2 inference engine (2-4x faster)
    - INT8 quantization support
    - Batched decoding
    - Lower memory usage

    Target: ~50-80ms per transcription
    """

    def __init__(self, device: str = "cuda", device_index: int = 0):
        self._model = None
        self._initialized = False
        self._device = device
        self._device_index = device_index

        # Performance tracking
        self._total_transcriptions = 0
        self._total_time = 0.0

    def initialize(self) -> None:
        """Load faster-whisper model."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("LOADING FASTER-WHISPER (CTranslate2 backend)")
        logger.info(f"Device: {self._device}:{self._device_index}")
        logger.info("Target: ~50-80ms per transcription")
        logger.info("=" * 60)

        start = time.time()

        from faster_whisper import WhisperModel

        # Use model from config (large-v3-turbo recommended for speed)
        # CTranslate2 supports: int8, int8_float16, float16, float32
        model_name = STT.model_size
        compute_type = STT.compute_type
        logger.info(f"  Model: {model_name}, compute_type: {compute_type}")
        self._model = WhisperModel(
            model_name,
            device=self._device,
            device_index=self._device_index,
            compute_type=compute_type,
        )

        load_time = time.time() - start
        logger.info(f"  Model loaded in {load_time:.1f}s")

        # Warmup (CTranslate2 compiles kernels on first run)
        logger.info("  Warming up...")
        dummy = np.zeros(16000, dtype=np.float32)  # 1s silence at 16kHz
        for i in range(3):
            warmup_start = time.time()
            segments, _ = self._model.transcribe(
                dummy,
                language="en",
                beam_size=1,
                vad_filter=False,
            )
            # Consume the generator
            for _ in segments:
                pass
            warmup_ms = (time.time() - warmup_start) * 1000
            logger.info(f"    Warmup {i+1}/3: {warmup_ms:.0f}ms")

        total_time = time.time() - start
        logger.info("=" * 60)
        logger.info(f"FASTER-WHISPER READY in {total_time:.1f}s")
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

        # Ensure CPU tensor (resampling is fast on CPU for small arrays)
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        elif audio.is_cuda:
            audio = audio.cpu()

        if audio.dtype != torch.float32:
            audio = audio.float()

        if audio.dim() > 1:
            audio = audio.squeeze()

        # Fast resampling using torchaudio
        global _resampler_24k_to_16k
        if AUDIO.sample_rate != 16000:
            if _resampler_24k_to_16k is None:
                _resampler_24k_to_16k = torchaudio.transforms.Resample(
                    orig_freq=AUDIO.sample_rate,
                    new_freq=16000,
                    lowpass_filter_width=16,  # Reduced from default for speed
                    rolloff=0.85,
                    resampling_method="sinc_interp_kaiser",
                )
            audio = _resampler_24k_to_16k(audio)

        # Convert to numpy for faster-whisper
        audio_np = audio.numpy().astype(np.float32)

        # Transcribe with fastest settings
        segments, info = self._model.transcribe(
            audio_np,
            language="en",
            beam_size=1,           # Greedy decoding (fastest)
            best_of=1,
            vad_filter=True,       # Filter silence segments
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100,
            ),
        )

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()

        # Track performance
        elapsed = time.time() - start
        self._total_transcriptions += 1
        self._total_time += elapsed

        logger.debug(f"Transcribed in {elapsed*1000:.0f}ms: '{text[:50]}'")

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
