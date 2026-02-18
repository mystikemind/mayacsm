"""
Seamless Pipeline Orchestrator

Optimized for natural, human-like responses with low latency.
Uses CSM with torch.compile for fast, high-quality audio.

Flow:
1. User stops speaking
2. STT (faster-whisper CTranslate2) - ~50-80ms
3. LLM (Llama 3.2 3B compiled) - ~200-300ms
4. TTS (CSM compiled) - ~400-600ms
5. Send complete audio

Target: <1.2s total latency with natural human-like audio.
"""

import torch
import asyncio
import time
import logging
import concurrent.futures
from typing import Optional, Callable, Awaitable, List
from dataclasses import dataclass

from ..engine import VADEngine
from ..engine.stt_faster import FasterSTTEngine
from ..engine.vad import SpeechState
# Use RealStreamingTTSEngine which has MIMI STREAMING CONTEXT
# This is CRITICAL for click-free audio (GitHub Issue #105)
# CompiledTTSEngine decodes frames without streaming context → clicks
from ..engine.tts_streaming_real import RealStreamingTTSEngine
from ..engine.llm_optimized import OptimizedLLMEngine
from ..engine.audio_humanizer import humanize_audio
from ..conversation import ConversationManager
from ..config import AUDIO

logger = logging.getLogger(__name__)

AudioCallback = Callable[[torch.Tensor], Awaitable[None]]


# Known Whisper hallucination phrases - filter these out
# ONLY exact matches - be VERY careful not to filter real user input!
# These appear when Whisper processes silence/noise with no real speech
WHISPER_HALLUCINATIONS = frozenset([
    # YouTube/video artifacts (very distinctive, not normal speech)
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",

    # Media annotations (never real speech)
    "music",
    "applause",
    "laughter",
    "silence",
    "inaudible",
    "unintelligible",
    "foreign",
    "speaking foreign language",

    # Blank/filler
    "",
    "...",
    ".",
])


def is_whisper_hallucination(transcript: str) -> bool:
    """
    Check if a transcript is likely a Whisper hallucination.

    CONSERVATIVE - only filter obvious hallucinations, not real user input!

    Args:
        transcript: The transcribed text to check

    Returns:
        True if this looks like a hallucination
    """
    if not transcript:
        return True

    # Normalize for comparison
    normalized = transcript.lower().strip()

    # Remove punctuation for comparison
    import re
    normalized = re.sub(r'[^\w\s]', '', normalized).strip()

    # ONLY exact matches - no partial matches to avoid filtering real speech
    if normalized in WHISPER_HALLUCINATIONS:
        logger.debug(f"Filtered hallucination: '{transcript}'")
        return True

    # Only filter VERY short transcripts (single meaningless sounds)
    if len(normalized) < 2:
        logger.debug(f"Filtered empty transcript")
        return True

    return False


@dataclass
class Metrics:
    """Pipeline metrics."""
    total_turns: int = 0
    total_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    total_time_to_first_audio_ms: float = 0.0
    avg_time_to_first_audio_ms: float = 0.0
    interruptions: int = 0


class SeamlessMayaPipeline:
    """
    Maya pipeline optimized for natural, human-like audio.

    Uses CSM with torch.compile for fast, high-quality speech.
    Target: <1.2s latency with natural prosody.

    Features:
    - Barge-in support: User can interrupt Maya mid-sentence
    - Echo cancellation: Brief cooldown after Maya speaks
    - Context-aware TTS: Passes conversation history for prosodic consistency
    - Faster-Whisper STT: ~50-80ms latency (CTranslate2 backend)
    """

    def __init__(self):
        self.vad = VADEngine()
        self.stt = FasterSTTEngine()
        self.llm = OptimizedLLMEngine()
        self.tts = RealStreamingTTSEngine()
        self.conversation = ConversationManager()

        self._initialized = False
        self._is_processing = False
        self._is_sending_audio = False
        self._interrupted = False
        self._audio_callback: Optional[AudioCallback] = None
        self._user_audio_buffer: List[torch.Tensor] = []
        self._metrics = Metrics()
        self._maya_stop_time: Optional[float] = None

        # Thread pool for parallel operations
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("INITIALIZING MAYA - OPTIMIZED FOR SPEED")
        logger.info("=" * 60)

        start = time.time()

        logger.info("Loading VAD...")
        self.vad.initialize()

        logger.info("Loading STT (Faster-Whisper CTranslate2)...")
        self.stt.initialize()

        logger.info("Loading LLM (Llama 3.2 3B compiled)...")
        self.llm.initialize()

        logger.info("Loading TTS (Compiled CSM)...")
        self.tts.initialize()

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"MAYA READY in {elapsed:.1f}s")
        logger.info("=" * 60)

        self._initialized = True

    def set_audio_callback(self, callback: AudioCallback) -> None:
        self._audio_callback = callback

    async def _send_audio(self, audio: torch.Tensor) -> None:
        """Send audio with single normalization pass. Supports interruption."""
        if self._audio_callback and audio is not None and len(audio) > 0:
            # Check for interruption before sending
            if self._interrupted:
                logger.info("Audio send cancelled due to interruption")
                return

            self._is_sending_audio = True

            try:
                if audio.dtype != torch.float32:
                    audio = audio.float()
                if audio.dim() > 1:
                    audio = audio.squeeze()

                # Skip invalid audio
                if torch.isnan(audio).any() or torch.isinf(audio).any():
                    return
                if audio.abs().max() == 0:
                    return

                # Quality gate: detect failed generations (very quiet audio)
                rms = torch.sqrt(torch.mean(audio ** 2))
                if rms < 0.005:
                    logger.warning(f"Audio quality gate: RMS={rms:.4f} too low, likely failed generation")
                    return

                # Single normalization: peak normalize to -6dB (0.5 linear)
                peak = audio.abs().max()
                if peak > 0:
                    target_peak = 0.5  # -6dB
                    audio = audio * (target_peak / peak)

                # Final interruption check
                if self._interrupted:
                    logger.info("Audio send cancelled due to interruption")
                    return

                await self._audio_callback(audio)
            finally:
                self._is_sending_audio = False

    async def process_audio_chunk(self, audio_chunk: torch.Tensor) -> None:
        """Process incoming audio with barge-in support."""
        if not self._initialized:
            await self.initialize()

        # Echo cooldown after Maya stops speaking - prevents picking up Maya's own audio
        # 150ms is sufficient with proper audio normalization (was 600ms - too conservative)
        # Sesame research: 100-150ms is optimal for natural conversation flow
        if self._maya_stop_time and (time.time() - self._maya_stop_time < 0.15):
            return

        if audio_chunk.dtype != torch.float32:
            audio_chunk = audio_chunk.float()
        if audio_chunk.dim() > 1:
            audio_chunk = audio_chunk.squeeze()

        # ALWAYS pass audio to VAD - even when Maya is speaking (for barge-in detection)
        vad_result = self.vad.process(audio_chunk)

        # Barge-in: User starts speaking while Maya is speaking/processing
        if vad_result.state == SpeechState.JUST_STARTED:
            if self._is_sending_audio or self._is_processing:
                # User is interrupting Maya!
                logger.info(">>> BARGE-IN DETECTED - User interrupted Maya <<<")
                self._interrupted = True
                self._metrics.interruptions += 1
                self.conversation.maya_stopped_speaking("", torch.tensor([]))  # End Maya's turn
                self._maya_stop_time = None  # No echo cooldown for interruption
            await self._handle_speech_start()

        elif vad_result.state == SpeechState.SPEAKING:
            # Only buffer if not being interrupted mid-process
            if not self._is_processing or self._interrupted:
                self._user_audio_buffer.append(audio_chunk)
                self.conversation.buffer_audio(audio_chunk)

        elif vad_result.state == SpeechState.JUST_ENDED:
            # Only process if we're not already processing (unless it's a barge-in)
            if not self._is_processing or self._interrupted:
                await self._handle_speech_end()

    async def _handle_speech_start(self) -> None:
        logger.info(">>> USER SPEAKING <<<")
        self.conversation.user_started_speaking()
        self._user_audio_buffer.clear()

    async def _handle_speech_end(self) -> None:
        """
        Handle user stopping - generate streaming response.

        Flow:
        1. STT (~100ms)
        2. LLM (~300ms)
        3. TTS STREAMING - send first chunk ASAP, continue generating
        4. Each chunk sent to client as generated

        First audio arrives at: STT + LLM + first_chunk_gen (~700-800ms)
        vs previous: STT + LLM + full_TTS (~1300ms)
        """
        if self._is_processing and not self._interrupted:
            return

        was_interrupted = self._interrupted
        self._interrupted = False
        self._is_processing = True
        response_start = time.time()

        if was_interrupted:
            logger.info("Processing user speech after barge-in")

        try:
            # Get user audio
            if self._user_audio_buffer:
                user_audio = torch.cat(self._user_audio_buffer)
            else:
                self._is_processing = False
                return

            # Filter out very short audio
            min_samples = int(AUDIO.sample_rate * 0.3)
            if len(user_audio) < min_samples:
                logger.debug(f"Audio too short ({len(user_audio)} samples), skipping")
                self._is_processing = False
                return

            # STEP 1: STT
            stt_start = time.time()
            transcript = self.stt.transcribe(user_audio)
            stt_time = (time.time() - stt_start) * 1000

            logger.info(f"[{stt_time:.0f}ms] STT: '{transcript}'")

            if is_whisper_hallucination(transcript):
                logger.info(f"Filtered potential hallucination: '{transcript}'")
                self._is_processing = False
                return

            self.conversation.user_stopped_speaking(transcript)
            self.tts.add_context(transcript, user_audio, is_user=True)

            if self._interrupted:
                logger.info("Interrupted before LLM")
                self._is_processing = False
                return

            # STEP 2: LLM
            llm_start = time.time()
            response = self.llm.generate(transcript)
            llm_time = (time.time() - llm_start) * 1000

            logger.info(f"[{llm_time:.0f}ms] LLM: '{response}'")

            if not response or not response.strip():
                logger.warning("Empty LLM response, using fallback")
                response = "tell me more about that"

            if self._interrupted:
                logger.info("Interrupted before TTS generation")
                self._is_processing = False
                return

            # STEP 3: TTS (TRUE STREAMING)
            # Send audio chunks as they are generated instead of waiting for all
            self.conversation.maya_started_speaking()
            tts_start = time.time()
            all_chunks = []
            first_chunk_sent = False
            first_audio_time = 0

            self._is_sending_audio = True

            for chunk in self.tts.generate_stream(response, use_context=True):
                # Check for barge-in between chunks
                if self._interrupted:
                    logger.info("Barge-in during TTS streaming, stopping")
                    break

                all_chunks.append(chunk)

                # Log and track first chunk timing
                if not first_chunk_sent:
                    first_audio_time = (time.time() - response_start) * 1000
                    logger.info(f">>> FIRST AUDIO at {first_audio_time:.0f}ms <<<")
                    first_chunk_sent = True

                # Humanize chunk: add jitter/shimmer/warmth for natural "aliveness"
                # Skip breath insertion (needs full audio context, not per-chunk)
                chunk = humanize_audio(
                    chunk, sample_rate=AUDIO.sample_rate,
                    jitter=0.3, shimmer=1.0, breaths=False, warmth=0.08
                )

                # Send this chunk to client immediately
                await self._send_audio(chunk)

            self._is_sending_audio = False
            tts_time = (time.time() - tts_start) * 1000

            # Reconstruct full audio for context
            if all_chunks:
                response_audio = torch.cat(all_chunks)
            else:
                response_audio = torch.tensor([])

            total_time = (time.time() - response_start) * 1000
            audio_duration = len(response_audio) / AUDIO.sample_rate if len(response_audio) > 0 else 0

            self.conversation.maya_stopped_speaking(response, response_audio)
            self._maya_stop_time = time.time()

            # Add Maya's response to TTS context (enables prosodic consistency)
            if len(response_audio) > 0:
                self.tts.add_context(response, response_audio, is_user=False)
            self.llm.add_context("assistant", response)

            # Metrics
            self._metrics.total_turns += 1
            self._metrics.total_response_time_ms += total_time
            self._metrics.avg_response_time_ms = (
                self._metrics.total_response_time_ms / self._metrics.total_turns
            )
            self._metrics.total_time_to_first_audio_ms += first_audio_time
            self._metrics.avg_time_to_first_audio_ms = (
                self._metrics.total_time_to_first_audio_ms / self._metrics.total_turns
            )

            rtf = tts_time / 1000 / audio_duration if audio_duration > 0 else 0
            logger.info(
                f"DONE: STT={stt_time:.0f}ms, LLM={llm_time:.0f}ms, "
                f"TTS={tts_time:.0f}ms (RTF={rtf:.2f}x), "
                f"First audio={first_audio_time:.0f}ms, "
                f"Total={total_time:.0f}ms, Audio={audio_duration:.1f}s"
            )

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            self._is_processing = False
            self._is_sending_audio = False
            self._user_audio_buffer.clear()

    async def play_greeting(self) -> None:
        """Play initial greeting."""
        greeting = "Hi, I'm Maya. How can I help you?"

        logger.info(f"Playing greeting: '{greeting}'")
        self.conversation.maya_started_speaking()

        # Generate greeting audio (fast compiled path)
        greeting_audio = self.tts.generate(greeting, use_context=False)

        # Full humanization on greeting (breaths + jitter + shimmer + warmth)
        greeting_audio = humanize_audio(
            greeting_audio, sample_rate=AUDIO.sample_rate,
            jitter=0.3, shimmer=1.0, breaths=True, warmth=0.08
        )

        await self._send_audio(greeting_audio)

        self.conversation.maya_stopped_speaking(greeting, greeting_audio)

        # Add greeting to context for prosodic consistency
        self.tts.add_context(greeting, greeting_audio, is_user=False)
        self.llm.add_context("assistant", greeting)
        self._maya_stop_time = time.time()  # Track when Maya stopped for echo cooldown

    async def start_conversation(self) -> None:
        """Start a new conversation with greeting."""
        await self.play_greeting()

    async def reset(self) -> None:
        """Reset conversation."""
        self.conversation.reset()
        self.tts.clear_context()
        self.llm.clear_history()
        self._is_processing = False
        self._is_sending_audio = False
        self._interrupted = False
        self._user_audio_buffer.clear()
        self._maya_stop_time = None

    def get_stats(self) -> dict:
        return {
            "initialized": self._initialized,
            "is_processing": self._is_processing,
            "total_turns": self._metrics.total_turns,
            "avg_response_ms": self._metrics.avg_response_time_ms,
            "avg_first_audio_ms": self._metrics.avg_time_to_first_audio_ms,
            "interruptions": self._metrics.interruptions,
        }

    @property
    def is_initialized(self) -> bool:
        """Check if pipeline is initialized."""
        return self._initialized
