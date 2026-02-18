"""
Streaming STT Engine - Prefetch transcription during speech.

Key optimization: Start transcribing audio chunks WHILE user is still speaking,
so when they finish, we already have partial transcription ready.

Flow:
1. User starts speaking
2. After 1s of speech, transcribe first chunk in background
3. Continue collecting audio
4. When user stops, only transcribe remaining portion
5. Combine partial + final transcripts

This can save ~150-200ms by overlapping transcription with speech.
"""

import torch
import numpy as np
import logging
import time
import threading
import queue
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

from .stt_local import LocalSTTEngine
from ..config import AUDIO

logger = logging.getLogger(__name__)


class StreamingSTTEngine:
    """
    Streaming STT with prefetch optimization.

    While user is speaking:
    - Buffer audio chunks
    - After 1s of audio, start transcribing first chunk in background
    - When speech ends, only transcribe remaining audio
    - Combine results

    This overlaps ~200ms of STT latency with speech time.
    """

    # Configuration - ULTRA OPTIMIZED for Sesame-level latency
    PREFETCH_THRESHOLD_MS = 300   # Start prefetch after 300ms - maximum overlap with speech
    CHUNK_DURATION_MS = 800       # Prefetch chunks - smaller for faster processing
    MIN_FINAL_CHUNK_MS = 150      # Minimum remaining audio - tighter threshold

    def __init__(self):
        self._stt = LocalSTTEngine()  # Uses local CUDA (~85ms vs Docker ~237ms)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt_")
        self._initialized = False

        # Streaming state
        self._audio_buffer: List[torch.Tensor] = []
        self._prefetch_future: Optional[Future] = None
        self._prefetch_result: Optional[str] = None
        self._prefetch_samples: int = 0  # Samples already prefetched
        self._lock = threading.Lock()

        # Stats
        self._total_transcriptions = 0
        self._prefetch_hits = 0
        self._total_time_saved_ms = 0.0

    def initialize(self) -> None:
        """Initialize underlying STT engine."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("INITIALIZING STREAMING STT")
        logger.info("Prefetch threshold: %dms", self.PREFETCH_THRESHOLD_MS)
        logger.info("Chunk duration: %dms", self.CHUNK_DURATION_MS)
        logger.info("=" * 60)

        self._stt.initialize()
        self._initialized = True

        logger.info("Streaming STT ready")

    def buffer_audio(self, audio_chunk: torch.Tensor) -> None:
        """
        Buffer incoming audio and trigger prefetch if threshold reached.

        Called continuously while user is speaking.
        """
        if not self._initialized:
            self.initialize()

        with self._lock:
            self._audio_buffer.append(audio_chunk)

            # Calculate total buffered audio
            total_samples = sum(len(c) for c in self._audio_buffer)
            total_ms = (total_samples / AUDIO.sample_rate) * 1000

            # Start prefetch if threshold reached and not already prefetching
            if total_ms >= self.PREFETCH_THRESHOLD_MS and self._prefetch_future is None:
                self._start_prefetch()

    def _start_prefetch(self) -> None:
        """Start prefetch transcription in background thread."""
        # Collect chunk for prefetch
        chunk_samples = int(self.CHUNK_DURATION_MS * AUDIO.sample_rate / 1000)
        samples_collected = 0
        chunks_to_prefetch = []

        for chunk in self._audio_buffer:
            if samples_collected >= chunk_samples:
                break
            chunks_to_prefetch.append(chunk)
            samples_collected += len(chunk)

        if not chunks_to_prefetch:
            return

        prefetch_audio = torch.cat(chunks_to_prefetch)
        self._prefetch_samples = len(prefetch_audio)

        logger.debug(f"Starting prefetch of {self._prefetch_samples} samples ({self._prefetch_samples/AUDIO.sample_rate*1000:.0f}ms)")

        # Start background transcription
        self._prefetch_future = self._executor.submit(
            self._stt.transcribe, prefetch_audio
        )

    def _wait_for_prefetch(self, timeout: float = 0.5) -> Optional[str]:
        """Wait for prefetch to complete and return result."""
        if self._prefetch_future is None:
            return None

        try:
            result = self._prefetch_future.result(timeout=timeout)
            self._prefetch_result = result
            return result
        except Exception as e:
            logger.debug(f"Prefetch failed: {e}")
            return None

    def transcribe(self, audio: Optional[torch.Tensor] = None) -> str:
        """
        Complete transcription - combine prefetch with remaining audio.

        Args:
            audio: Optional full audio. If None, uses buffered audio.

        Returns:
            Full transcription
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        with self._lock:
            # Get full audio
            if audio is not None:
                full_audio = audio
            elif self._audio_buffer:
                full_audio = torch.cat(self._audio_buffer)
            else:
                return ""

            # Check if we have prefetch result
            prefetch_text = None
            if self._prefetch_future is not None:
                prefetch_text = self._wait_for_prefetch()

            # Calculate remaining audio (after prefetched portion)
            remaining_samples = len(full_audio) - self._prefetch_samples
            remaining_ms = (remaining_samples / AUDIO.sample_rate) * 1000

            if prefetch_text and remaining_ms < self.MIN_FINAL_CHUNK_MS:
                # Prefetch covered most/all audio - use it directly
                final_text = prefetch_text
                self._prefetch_hits += 1
                logger.debug("Using prefetch result directly")

            elif prefetch_text and remaining_samples > 0:
                # Transcribe remaining portion and combine
                remaining_audio = full_audio[self._prefetch_samples:]
                remaining_text = self._stt.transcribe(remaining_audio)

                # Combine (prefetch might overlap with remaining, so be careful)
                if remaining_text.strip():
                    final_text = f"{prefetch_text} {remaining_text}".strip()
                else:
                    final_text = prefetch_text

                self._prefetch_hits += 1
                logger.debug(f"Combined: prefetch + {remaining_ms:.0f}ms remaining")

            else:
                # No prefetch or it failed - transcribe full audio
                final_text = self._stt.transcribe(full_audio)
                logger.debug("Full transcription (no prefetch)")

            # Reset state
            self._reset_state()

        elapsed = time.time() - start
        self._total_transcriptions += 1

        # Estimate time saved (prefetch ran in parallel with speech)
        if prefetch_text:
            # Saved roughly the time of prefetch chunk transcription
            saved_ms = self.CHUNK_DURATION_MS * 0.15  # ~15% of chunk time saved
            self._total_time_saved_ms += saved_ms
            logger.debug(f"Transcribed in {elapsed*1000:.0f}ms (est. {saved_ms:.0f}ms saved by prefetch)")

        return final_text

    def _reset_state(self) -> None:
        """Reset streaming state for next utterance."""
        self._audio_buffer.clear()
        self._prefetch_future = None
        self._prefetch_result = None
        self._prefetch_samples = 0

    def clear(self) -> None:
        """Clear all buffers and cancel pending prefetch."""
        with self._lock:
            if self._prefetch_future is not None:
                self._prefetch_future.cancel()
            self._reset_state()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def prefetch_hit_rate(self) -> float:
        if self._total_transcriptions == 0:
            return 0.0
        return self._prefetch_hits / self._total_transcriptions

    def get_stats(self) -> dict:
        return {
            "total_transcriptions": self._total_transcriptions,
            "prefetch_hits": self._prefetch_hits,
            "prefetch_hit_rate": self.prefetch_hit_rate,
            "total_time_saved_ms": self._total_time_saved_ms,
            "stt_stats": self._stt.get_stats(),
        }

    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
