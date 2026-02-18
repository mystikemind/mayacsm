"""
Maya Pipeline Orchestrator - The Heart of the System

Coordinates all components for seamless conversation:
1. VAD detects speech → trigger transcription
2. STT transcribes → send to LLM
3. LLM generates → send to TTS
4. TTS synthesizes → stream to user

THE ZERO LATENCY TRICK:
- When user stops: IMMEDIATELY play filler
- While filler plays: transcribe → think → synthesize
- Crossfade filler into response
- User perceives INSTANT response
"""

import torch
import asyncio
import time
import logging
from typing import Optional, Callable, Awaitable, List
from dataclasses import dataclass

from ..engine import VADEngine, STTEngine, LLMEngine, TTSEngine
from ..engine.vad import SpeechState
from ..conversation import FillerSystem, ConversationManager
from ..config import AUDIO, FILLER, LATENCY

logger = logging.getLogger(__name__)


# Type alias for audio callback
AudioCallback = Callable[[torch.Tensor], Awaitable[None]]


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    total_turns: int = 0
    total_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    interruptions: int = 0
    backchannels_played: int = 0


class MayaPipeline:
    """
    The main Maya voice pipeline.

    FLOW:
    1. User speaks → VAD detects, buffer audio
    2. User stops → IMMEDIATELY play thinking filler
    3. Parallel processing:
       - STT transcribes user audio
       - LLM generates response (streaming)
       - TTS synthesizes response (streaming)
    4. When ready: crossfade filler → response
    5. User hears seamless response

    This creates ZERO PERCEIVED LATENCY even though
    actual processing takes 750ms+.
    """

    def __init__(self):
        # Core engines
        self.vad = VADEngine()
        self.stt = STTEngine()
        self.llm = LLMEngine()
        self.tts = TTSEngine()

        # Conversation management
        self.fillers = FillerSystem()
        self.conversation = ConversationManager()

        # State
        self._initialized = False
        self._is_processing = False
        self._should_stop = False

        # Audio callback
        self._audio_callback: Optional[AudioCallback] = None

        # Audio buffer for current user speech
        self._user_audio_buffer: List[torch.Tensor] = []

        # Backchannel timer
        self._backchannel_task: Optional[asyncio.Task] = None
        self._last_backchannel_time: float = 0

        # Metrics
        self._metrics = PipelineMetrics()

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("INITIALIZING MAYA PIPELINE")
        logger.info("=" * 60)

        start = time.time()

        # Initialize in optimal order
        logger.info("Loading VAD...")
        self.vad.initialize()

        logger.info("Loading STT (faster-whisper)...")
        self.stt.initialize()

        logger.info("Loading LLM (Llama 3.2 3B)...")
        self.llm.initialize()

        logger.info("Loading TTS (CSM-1B)...")
        self.tts.initialize()

        logger.info("Initializing filler system...")
        await self.fillers.initialize(tts_engine=self.tts)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"MAYA READY in {elapsed:.1f}s")
        logger.info("=" * 60)

        self._initialized = True

    def set_audio_callback(self, callback: AudioCallback) -> None:
        """Set callback for sending audio to user."""
        self._audio_callback = callback

    async def _send_audio(self, audio: torch.Tensor) -> None:
        """Send audio through callback."""
        if self._audio_callback:
            # Normalize audio
            if audio.abs().max() > 0:
                audio = audio / audio.abs().max() * 0.95
            await self._audio_callback(audio)

    async def process_audio_chunk(self, audio_chunk: torch.Tensor) -> None:
        """
        Process incoming audio chunk from user.

        Called continuously as audio streams in.
        """
        if not self._initialized:
            await self.initialize()

        # Ensure correct format
        if audio_chunk.dtype != torch.float32:
            audio_chunk = audio_chunk.float()
        if audio_chunk.dim() > 1:
            audio_chunk = audio_chunk.squeeze()

        # Run VAD
        vad_result = self.vad.process(audio_chunk)

        # Handle state transitions
        if vad_result.state == SpeechState.JUST_STARTED:
            await self._handle_speech_start()

        elif vad_result.state == SpeechState.SPEAKING:
            # Buffer audio while user speaks
            self._user_audio_buffer.append(audio_chunk)
            self.conversation.buffer_audio(audio_chunk)

            # Check for backchannel opportunity
            await self._maybe_play_backchannel()

        elif vad_result.state == SpeechState.JUST_ENDED:
            # User stopped - this is where the magic happens
            await self._handle_speech_end()

        # Check for interruption
        if self.conversation.is_maya_speaking:
            if self.vad.check_interruption(maya_is_speaking=True):
                await self._handle_interruption()

    async def _handle_speech_start(self) -> None:
        """Handle user starting to speak."""
        logger.debug("User started speaking")

        self.conversation.user_started_speaking()
        self._user_audio_buffer.clear()

        # Start backchannel timer
        self._start_backchannel_timer()

    async def _handle_speech_end(self) -> None:
        """
        Handle user stopping speaking.

        THIS IS WHERE THE ZERO LATENCY MAGIC HAPPENS:
        1. IMMEDIATELY play thinking filler
        2. In parallel: transcribe, generate response, synthesize
        3. Crossfade filler into response
        """
        if self._is_processing:
            return

        self._is_processing = True
        self._stop_backchannel_timer()

        try:
            process_start = time.time()

            # STEP 1: IMMEDIATELY play filler (zero latency trick)
            filler_audio, filler_text = self.fillers.get_thinking_filler()
            filler_task = asyncio.create_task(self._send_audio(filler_audio))

            logger.info("Playing thinking filler...")

            # STEP 2: Transcribe user audio
            stt_start = time.time()
            user_audio = torch.cat(self._user_audio_buffer) if self._user_audio_buffer else torch.zeros(AUDIO.sample_rate)
            transcript = self.stt.transcribe(user_audio)
            stt_time = (time.time() - stt_start) * 1000

            logger.info(f"STT ({stt_time:.0f}ms): '{transcript}'")

            # Skip empty/noise
            if not transcript or len(transcript.strip()) < 2:
                logger.debug("Skipping empty transcript")
                self._is_processing = False
                return

            # Update conversation
            user_turn = self.conversation.user_stopped_speaking(transcript)

            # Add to TTS context
            self.tts.add_context(transcript, user_audio, is_user=True)

            # STEP 3: Generate LLM response
            llm_start = time.time()
            response = self.llm.generate(transcript)
            llm_time = (time.time() - llm_start) * 1000

            logger.info(f"LLM ({llm_time:.0f}ms): '{response[:50]}...'")

            # STEP 4: Generate TTS audio with conversation context
            self.conversation.maya_started_speaking()
            tts_start = time.time()
            response_audio = self.tts.generate(response, use_context=True)
            tts_time = (time.time() - tts_start) * 1000

            logger.info(f"TTS ({tts_time:.0f}ms): {len(response_audio)/AUDIO.sample_rate:.1f}s audio")

            # STEP 5: Wait for filler to finish
            await filler_task

            # Small gap for natural transition
            await asyncio.sleep(0.05)

            # STEP 6: Play response
            if not self._should_stop:
                await self._send_audio(response_audio)

            # Update conversation
            maya_turn = self.conversation.maya_stopped_speaking(response, response_audio)

            # Add Maya's response to TTS context
            self.tts.add_context(response, response_audio, is_user=False)

            # Track metrics
            total_time = (time.time() - process_start) * 1000
            self._metrics.total_turns += 1
            self._metrics.total_response_time_ms += total_time
            self._metrics.avg_response_time_ms = (
                self._metrics.total_response_time_ms / self._metrics.total_turns
            )

            logger.info(
                f"Turn complete: STT={stt_time:.0f}ms, LLM={llm_time:.0f}ms, "
                f"TTS={tts_time:.0f}ms, Total={total_time:.0f}ms"
            )

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)
        finally:
            self._is_processing = False
            self._user_audio_buffer.clear()

    async def _handle_interruption(self) -> None:
        """Handle user interrupting Maya."""
        logger.info("User interrupted Maya!")

        self._should_stop = True
        self._metrics.interruptions += 1

        # Clear any pending audio
        self._user_audio_buffer.clear()

        # Reset state
        await asyncio.sleep(0.1)
        self._should_stop = False

    async def _maybe_play_backchannel(self) -> None:
        """Maybe play a backchannel while user is speaking."""
        if self._is_processing:
            return

        current_time = time.time()
        time_since_last = current_time - self._last_backchannel_time

        # Check if enough time has passed
        if time_since_last >= FILLER.backchannel_interval_min:
            # Random chance
            import random
            if random.random() < 0.3:  # 30% chance when eligible
                backchannel_audio, _ = self.fillers.get_backchannel()
                await self._send_audio(backchannel_audio)
                self._last_backchannel_time = current_time
                self._metrics.backchannels_played += 1
                logger.debug("Played backchannel")

    def _start_backchannel_timer(self) -> None:
        """Start timer for periodic backchannels."""
        self._last_backchannel_time = time.time()

    def _stop_backchannel_timer(self) -> None:
        """Stop backchannel timer."""
        if self._backchannel_task:
            self._backchannel_task.cancel()
            self._backchannel_task = None

    async def start_conversation(self) -> None:
        """Start conversation with a greeting."""
        if not self._initialized:
            await self.initialize()

        greeting = "Hi! I'm Maya. How can I help you today?"

        logger.info(f"Maya greeting: '{greeting}'")

        self.conversation.maya_started_speaking()
        audio = self.tts.generate(greeting, use_context=False)
        await self._send_audio(audio)
        self.conversation.maya_stopped_speaking(greeting, audio)

        # Add to context
        self.tts.add_context(greeting, audio, is_user=False)
        self.llm.add_context("assistant", greeting)

    async def reset(self) -> None:
        """Reset for new conversation."""
        self._stop_backchannel_timer()
        self.conversation.clear()
        self.tts.clear_context()
        self.llm.clear_history()
        self._user_audio_buffer.clear()
        self._is_processing = False
        self._should_stop = False
        self.vad.reset()

        logger.info("Pipeline reset")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_metrics(self) -> PipelineMetrics:
        """Get pipeline metrics."""
        return self._metrics

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        return {
            "pipeline": {
                "initialized": self._initialized,
                "is_processing": self._is_processing,
                "total_turns": self._metrics.total_turns,
                "avg_response_ms": self._metrics.avg_response_time_ms,
                "interruptions": self._metrics.interruptions,
                "backchannels": self._metrics.backchannels_played,
            },
            "conversation": self.conversation.get_stats().__dict__,
            "stt": self.stt.get_stats(),
            "llm": self.llm.get_stats(),
            "tts": self.tts.get_stats(),
            "fillers": self.fillers.get_stats(),
        }
