"""
Voice Activity Detection Engine - Silero VAD + Smart Turn Detection

Detects:
- When user starts speaking
- When user stops speaking (turn end) - using prosodic analysis
- When user interrupts Maya

Combines:
1. Silero VAD for speech/silence detection
2. Prosody Turn Detector for intelligent turn boundary detection

This prevents:
- Interrupting users mid-thought during pauses
- Waiting too long when user is clearly done speaking
"""

import torch
import torchaudio.functional as F  # Import at module level, not in hot path
import numpy as np
import logging
import threading
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass
from enum import Enum, auto
import time

from ..config import VAD, AUDIO

logger = logging.getLogger(__name__)


class SpeechState(Enum):
    """Current speech state."""
    SILENCE = auto()
    SPEAKING = auto()
    JUST_STARTED = auto()
    JUST_ENDED = auto()


@dataclass
class VADResult:
    """Result of VAD processing."""
    is_speech: bool
    confidence: float
    state: SpeechState
    speech_duration_ms: float
    silence_duration_ms: float


class VADEngine:
    """
    Silero VAD + Smart Turn Detection.

    Features:
    - Low latency (~30ms per chunk)
    - Accurate speech/silence detection
    - **Smart turn boundary detection using prosodic analysis**
    - Interruption detection

    Smart Turn Detection:
    - Analyzes pitch contour (falling = statement complete)
    - Analyzes energy decay (falling = turn ending)
    - Detects final pauses
    - Adapts silence timeout based on turn completeness
    """

    # Silence thresholds - aligned with config
    MIN_SILENCE_MS = VAD.min_silence_ms  # From config (default 350ms) - faster turn detection
    MAX_SILENCE_MS = 2000  # Force turn end after this silence
    QUICK_COMPLETE_MS = 300  # If turn is clearly complete, use shorter timeout

    def __init__(self):
        self._model = None
        self._turn_detector = None
        self._initialized = False

        # Thread safety lock for state mutations
        self._lock = threading.RLock()

        # State tracking (protected by _lock)
        self._is_speech = False
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._last_speech_end_time: Optional[float] = None

        # Audio buffer for turn detection
        self._speech_audio_buffer: List[torch.Tensor] = []

        # Callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        self._on_interruption: Optional[Callable] = None

    def initialize(self) -> None:
        """Load Silero VAD model and turn detector."""
        if self._initialized:
            return

        logger.info("Loading Silero VAD...")

        # Load VAD model
        self._model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self._model.eval()

        # Get utilities
        (
            self._get_speech_timestamps,
            self._save_audio,
            self._read_audio,
            self._VADIterator,
            self._collect_chunks
        ) = utils

        logger.info("Silero VAD loaded successfully")

        # Load smart turn detector
        logger.info("Loading Smart Turn Detector...")
        from .turn_detector import ProsodyTurnDetector
        self._turn_detector = ProsodyTurnDetector()
        self._turn_detector.initialize()

        self._initialized = True
        self._silence_start_time = time.time()

        logger.info("VAD + Smart Turn Detection ready")

    def process(self, audio_chunk: torch.Tensor) -> VADResult:
        """
        Process an audio chunk with smart turn detection.

        Uses:
        1. Silero VAD for speech/silence detection
        2. Prosody analysis for intelligent turn boundary detection

        Args:
            audio_chunk: Audio tensor at 24kHz

        Returns:
            VADResult with speech detection info
        """
        if not self._initialized:
            self.initialize()

        # Ensure correct format
        if audio_chunk.dtype != torch.float32:
            audio_chunk = audio_chunk.float()

        if audio_chunk.dim() > 1:
            audio_chunk = audio_chunk.squeeze()

        # Keep original for turn detection (24kHz)
        original_chunk = audio_chunk.clone()

        # Resample from 24kHz to 16kHz for Silero VAD (F imported at module level)
        if AUDIO.sample_rate == 24000 and len(audio_chunk) > 0:
            audio_chunk = F.resample(audio_chunk, AUDIO.sample_rate, VAD.sample_rate)

        # Silero VAD requires exactly 512 samples at 16kHz (32ms chunks)
        if len(audio_chunk) < 512:
            audio_chunk = torch.nn.functional.pad(audio_chunk, (0, 512 - len(audio_chunk)))

        vad_input = audio_chunk[:512]

        # Run VAD
        with torch.no_grad():
            speech_prob = self._model(vad_input, VAD.sample_rate).item()

        is_speech = speech_prob > VAD.threshold

        if speech_prob > 0.3:
            logger.debug(f"VAD: speech_prob={speech_prob:.3f}, is_speech={is_speech}")

        current_time = time.time()
        state = SpeechState.SILENCE
        speech_duration_ms = 0.0
        silence_duration_ms = 0.0

        if is_speech and not self._is_speech:
            # Speech just started
            state = SpeechState.JUST_STARTED
            self._is_speech = True
            self._speech_start_time = current_time
            self._silence_start_time = None
            self._speech_audio_buffer.clear()
            self._speech_audio_buffer.append(original_chunk)

            if self._on_speech_start:
                self._on_speech_start()

            logger.debug("Speech started")

        elif is_speech and self._is_speech:
            # Continuing to speak - buffer audio for turn detection
            state = SpeechState.SPEAKING
            self._speech_audio_buffer.append(original_chunk)

            # Limit buffer to last 4 seconds (for memory)
            max_chunks = int(4.0 * AUDIO.sample_rate / len(original_chunk))
            if len(self._speech_audio_buffer) > max_chunks:
                self._speech_audio_buffer = self._speech_audio_buffer[-max_chunks:]

            if self._speech_start_time:
                speech_duration_ms = (current_time - self._speech_start_time) * 1000

        elif not is_speech and self._is_speech:
            # Might be ending speech - use smart turn detection
            if self._silence_start_time is None:
                self._silence_start_time = current_time

            silence_duration_ms = (current_time - self._silence_start_time) * 1000

            # Smart turn detection logic
            should_end_turn = False

            if silence_duration_ms >= self.MAX_SILENCE_MS:
                # Force end after max silence
                should_end_turn = True
                logger.debug("Turn end: max silence reached")

            elif silence_duration_ms >= self.MIN_SILENCE_MS:
                # Check prosodic turn completion
                if self._speech_audio_buffer and self._turn_detector:
                    speech_audio = torch.cat(self._speech_audio_buffer)
                    audio_np = speech_audio.cpu().numpy()

                    is_complete, confidence = self._turn_detector.is_turn_complete(
                        audio_np, sample_rate=AUDIO.sample_rate
                    )

                    if is_complete and confidence > 0.6:
                        should_end_turn = True
                        logger.debug(f"Turn end: prosody complete (conf={confidence:.2f})")
                    elif silence_duration_ms >= VAD.min_silence_ms:
                        # Fallback to config silence threshold
                        should_end_turn = True
                        logger.debug("Turn end: silence threshold")
                else:
                    # No turn detector, use silence threshold
                    if silence_duration_ms >= VAD.min_silence_ms:
                        should_end_turn = True

            if should_end_turn:
                state = SpeechState.JUST_ENDED
                self._is_speech = False
                self._last_speech_end_time = current_time

                if self._speech_start_time:
                    speech_duration_ms = (current_time - self._speech_start_time) * 1000

                if self._on_speech_end:
                    self._on_speech_end()

                self._speech_audio_buffer.clear()
                logger.debug(f"Speech ended after {speech_duration_ms:.0f}ms")
            else:
                # Still in speech, just a pause
                state = SpeechState.SPEAKING

        else:
            # Continuing silence
            state = SpeechState.SILENCE
            if self._silence_start_time:
                silence_duration_ms = (current_time - self._silence_start_time) * 1000

        return VADResult(
            is_speech=is_speech,
            confidence=speech_prob,
            state=state,
            speech_duration_ms=speech_duration_ms,
            silence_duration_ms=silence_duration_ms
        )

    def check_interruption(self, maya_is_speaking: bool) -> bool:
        """
        Check if user is interrupting Maya.

        Thread-safe check of speech state.

        Args:
            maya_is_speaking: Whether Maya is currently speaking

        Returns:
            True if user is interrupting
        """
        with self._lock:
            if maya_is_speaking and self._is_speech:
                if self._speech_start_time:
                    speech_duration = (time.time() - self._speech_start_time) * 1000
                    # User needs to speak for at least 200ms to count as interruption
                    if speech_duration >= 200:
                        if self._on_interruption:
                            self._on_interruption()
                        return True
        return False

    def set_callbacks(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_interruption: Optional[Callable] = None
    ) -> None:
        """Set event callbacks."""
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_interruption = on_interruption

    def reset(self) -> None:
        """Reset VAD state (thread-safe)."""
        with self._lock:
            self._is_speech = False
            self._speech_start_time = None
            self._silence_start_time = time.time()
            self._last_speech_end_time = None
            self._speech_audio_buffer.clear()

            if self._model is not None:
                self._model.reset_states()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_speech(self) -> bool:
        """Thread-safe check if speech is active."""
        with self._lock:
            return self._is_speech
