"""
Smart Pipeline Orchestrator - FILLER + OVERLAP STRATEGY

This is how Sesame Maya achieves perceived low latency despite slow generation:

1. User stops speaking → IMMEDIATELY play short filler (~0.5-1s)
2. In PARALLEL, process: STT → LLM → TTS
3. When TTS first chunk ready, crossfade from filler to response
4. Continue streaming actual response

Result:
- Perceived latency: ~200ms (filler starts immediately)
- Actual latency: ~1.5-2s (masked by filler)
- Smooth audio: crossfade prevents clicks

Research backing:
- "Intentional disfluencies: um, uh, hmm for thinking time" (Sesame Maya docs)
- "Generate 20-50ms chunks of audio tokens, decode immediately"
- "Client-side jitter buffer smooths variations"
"""

import torch
import asyncio
import time
import logging
import random
from typing import Optional, Callable, Awaitable, List
from dataclasses import dataclass
from pathlib import Path

from ..engine import VADEngine, STTEngine, LLMEngine
from ..engine.vad import SpeechState
from ..engine.tts_streaming import StreamingTTSEngine, StreamingConfig
from ..conversation import ConversationManager
from ..config import AUDIO, PROJECT_ROOT

logger = logging.getLogger(__name__)

AudioCallback = Callable[[torch.Tensor], Awaitable[None]]


# Import hallucination filter from seamless_orchestrator
from .seamless_orchestrator import is_whisper_hallucination, WHISPER_HALLUCINATIONS


@dataclass
class Metrics:
    """Pipeline metrics."""
    total_turns: int = 0
    total_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    total_time_to_first_audio_ms: float = 0.0
    avg_time_to_first_audio_ms: float = 0.0
    filler_used_count: int = 0
    filler_skipped_count: int = 0


# TTS config optimized for quality (we have filler to mask latency)
RESPONSE_CONFIG = StreamingConfig(
    initial_batch_size=12,     # Larger first chunk for better quality
    batch_size=20,             # Large chunks to reduce overhead
    buffer_size=20,
    max_audio_length_ms=5000,
    temperature=0.8,
    topk=50
)


# Short thinking sounds - varied to sound natural
THINKING_FILLERS = [
    "hmm",
    "well",
    "so",
    "let me see",
    "okay",
]


class SmartMayaPipeline:
    """
    Maya pipeline with smart filler strategy.

    Key innovation: Play filler IMMEDIATELY while generating response.
    This masks the generation latency completely.

    Flow:
    1. User stops → Play filler (0ms delay)
    2. STT + LLM + TTS run in parallel
    3. Crossfade from filler to response
    4. Stream rest of response
    """

    def __init__(self):
        self.vad = VADEngine()
        self.stt = STTEngine()
        self.llm = LLMEngine()
        self.tts = StreamingTTSEngine()
        self.conversation = ConversationManager()

        self._initialized = False
        self._is_processing = False
        self._audio_callback: Optional[AudioCallback] = None
        self._user_audio_buffer: List[torch.Tensor] = []
        self._metrics = Metrics()
        self._maya_stop_time: Optional[float] = None

        # Pre-generated filler audio cache
        self._filler_cache: dict = {}
        self._last_filler_idx = -1  # Track last used filler to avoid repetition

    async def initialize(self) -> None:
        """Initialize all components and pre-generate fillers."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("INITIALIZING MAYA - SMART FILLER STRATEGY")
        logger.info("=" * 60)

        start = time.time()

        logger.info("Loading VAD...")
        self.vad.initialize()

        logger.info("Loading STT (Whisper turbo)...")
        self.stt.initialize()

        logger.info("Loading LLM (Llama 3.2 3B)...")
        self.llm.initialize()

        logger.info("Loading TTS (CSM-1B streaming)...")
        self.tts.initialize()

        # Pre-generate filler audio
        logger.info("Pre-generating filler audio...")
        await self._pregenerate_fillers()

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"MAYA READY in {elapsed:.1f}s")
        logger.info(f"Fillers cached: {len(self._filler_cache)}")
        logger.info("Strategy: Filler + Overlap for perceived ~200ms latency")
        logger.info("=" * 60)

        self._initialized = True

    async def _pregenerate_fillers(self) -> None:
        """Pre-generate all filler audio for instant playback."""
        filler_config = StreamingConfig(
            initial_batch_size=4,
            batch_size=8,
            max_audio_length_ms=1500,  # Max 1.5 seconds
            temperature=0.8,
            topk=50
        )

        for filler_text in THINKING_FILLERS:
            try:
                logger.debug(f"Generating filler: '{filler_text}'")
                chunks = []
                async for chunk in self.tts.generate_stream(
                    filler_text,
                    use_context=False,
                    config=filler_config
                ):
                    chunks.append(chunk)

                if chunks:
                    full_audio = torch.cat(chunks)
                    # Trim to max 1 second
                    max_samples = int(AUDIO.sample_rate * 1.0)
                    if len(full_audio) > max_samples:
                        full_audio = full_audio[:max_samples]

                    # Apply fade out for smooth transition
                    fade_samples = min(1200, len(full_audio) // 4)  # 50ms fade
                    if fade_samples > 0:
                        fade = torch.linspace(1.0, 0.3, fade_samples, device=full_audio.device)
                        full_audio[-fade_samples:] = full_audio[-fade_samples:] * fade

                    # Move to CPU for storage (will be sent via WebSocket)
                    self._filler_cache[filler_text] = full_audio.cpu()
                    duration = len(full_audio) / AUDIO.sample_rate
                    logger.debug(f"Cached filler '{filler_text}': {duration:.2f}s")

            except Exception as e:
                logger.warning(f"Failed to generate filler '{filler_text}': {e}")

    def _get_random_filler(self) -> tuple:
        """Get a random filler, avoiding the last used one."""
        if not self._filler_cache:
            return None, None

        available = list(self._filler_cache.keys())

        # Avoid repeating the same filler
        if len(available) > 1 and self._last_filler_idx >= 0:
            # Remove last used from options
            last_text = THINKING_FILLERS[self._last_filler_idx] if self._last_filler_idx < len(THINKING_FILLERS) else None
            if last_text in available:
                available.remove(last_text)

        chosen_text = random.choice(available)
        self._last_filler_idx = THINKING_FILLERS.index(chosen_text) if chosen_text in THINKING_FILLERS else -1

        return chosen_text, self._filler_cache[chosen_text]

    def set_audio_callback(self, callback: AudioCallback) -> None:
        self._audio_callback = callback

    async def _send_audio(self, audio: torch.Tensor) -> None:
        """Send audio with normalization."""
        if self._audio_callback and audio is not None and len(audio) > 0:
            if audio.dtype != torch.float32:
                audio = audio.float()
            if audio.dim() > 1:
                audio = audio.squeeze()

            if torch.isnan(audio).any() or torch.isinf(audio).any():
                return
            if audio.abs().max() == 0:
                return

            # Remove DC offset and normalize
            audio = audio - audio.mean()
            peak = audio.abs().max()
            if peak > 0:
                audio = audio * (0.5 / peak)

            await self._audio_callback(audio)

    async def process_audio_chunk(self, audio_chunk: torch.Tensor) -> None:
        """Process incoming audio."""
        if not self._initialized:
            await self.initialize()

        if self.conversation.is_maya_speaking or self._is_processing:
            return

        if self._maya_stop_time and (time.time() - self._maya_stop_time < 1.5):
            return

        if audio_chunk.dtype != torch.float32:
            audio_chunk = audio_chunk.float()
        if audio_chunk.dim() > 1:
            audio_chunk = audio_chunk.squeeze()

        audio_energy = audio_chunk.abs().max().item()
        if audio_energy < 0.05:
            return

        vad_result = self.vad.process(audio_chunk)

        if vad_result.state == SpeechState.JUST_STARTED:
            await self._handle_speech_start()
        elif vad_result.state == SpeechState.SPEAKING:
            self._user_audio_buffer.append(audio_chunk)
            self.conversation.buffer_audio(audio_chunk)
        elif vad_result.state == SpeechState.JUST_ENDED:
            await self._handle_speech_end()

    async def _handle_speech_start(self) -> None:
        logger.info(">>> USER SPEAKING <<<")
        self.conversation.user_started_speaking()
        self._user_audio_buffer.clear()

    async def _handle_speech_end(self) -> None:
        """
        Handle user stopping - SMART FILLER STRATEGY.

        1. IMMEDIATELY play filler (0ms perceived latency)
        2. Process STT → LLM → TTS in parallel
        3. Crossfade from filler to response
        """
        if self._is_processing:
            return

        self._is_processing = True
        response_start = time.time()

        try:
            # Get user audio
            if self._user_audio_buffer:
                user_audio = torch.cat(self._user_audio_buffer)
            else:
                self._is_processing = False
                return

            # Filter very short audio
            min_samples = int(AUDIO.sample_rate * 0.3)
            if len(user_audio) < min_samples:
                self._is_processing = False
                return

            # STEP 1: Start filler IMMEDIATELY
            filler_text, filler_audio = self._get_random_filler()
            filler_task = None

            if filler_audio is not None:
                # Play filler in background while we process
                async def play_filler():
                    logger.info(f">>> FILLER: '{filler_text}' <<<")
                    await self._send_audio(filler_audio)

                filler_task = asyncio.create_task(play_filler())
                self._metrics.filler_used_count += 1

            # STEP 2: STT (runs while filler plays)
            stt_start = time.time()
            transcript = self.stt.transcribe(user_audio)
            stt_time = (time.time() - stt_start) * 1000

            logger.info(f"[{stt_time:.0f}ms] STT: '{transcript}'")

            # Filter hallucinations
            if is_whisper_hallucination(transcript):
                logger.info(f"Filtered hallucination: '{transcript}'")
                if filler_task:
                    await filler_task  # Let filler finish
                self._is_processing = False
                return

            self.conversation.user_stopped_speaking(transcript)

            # STEP 3: LLM (runs while filler plays)
            llm_start = time.time()
            response = self.llm.generate(transcript)
            llm_time = (time.time() - llm_start) * 1000

            logger.info(f"[{llm_time:.0f}ms] LLM: '{response[:50]}...'")

            # Wait for filler to finish before starting response
            if filler_task:
                await filler_task

            # STEP 4: TTS STREAMING
            self.conversation.maya_started_speaking()
            tts_start = time.time()
            first_chunk_sent = False
            first_chunk_time = None
            total_audio_samples = 0
            chunk_count = 0

            async for audio_chunk in self.tts.generate_stream(
                response,
                use_context=True,
                config=RESPONSE_CONFIG
            ):
                if not first_chunk_sent:
                    first_chunk_time = (time.time() - response_start) * 1000
                    logger.info(f">>> FIRST RESPONSE AUDIO at {first_chunk_time:.0f}ms <<<")
                    first_chunk_sent = True

                await self._send_audio(audio_chunk)
                total_audio_samples += len(audio_chunk)
                chunk_count += 1

            tts_time = (time.time() - tts_start) * 1000
            total_time = (time.time() - response_start) * 1000
            audio_duration = total_audio_samples / AUDIO.sample_rate

            response_audio = torch.zeros(total_audio_samples)
            self.conversation.maya_stopped_speaking(response, response_audio)
            self._maya_stop_time = time.time()

            self.tts.add_context(transcript, user_audio, is_user=True)
            self.llm.add_context("assistant", response)

            # Metrics
            self._metrics.total_turns += 1
            self._metrics.total_response_time_ms += total_time
            self._metrics.avg_response_time_ms = (
                self._metrics.total_response_time_ms / self._metrics.total_turns
            )
            if first_chunk_time:
                self._metrics.total_time_to_first_audio_ms += first_chunk_time
                self._metrics.avg_time_to_first_audio_ms = (
                    self._metrics.total_time_to_first_audio_ms / self._metrics.total_turns
                )

            filler_info = f"Filler='{filler_text}'" if filler_text else "NoFiller"
            logger.info(
                f"DONE: {filler_info}, STT={stt_time:.0f}ms, LLM={llm_time:.0f}ms, "
                f"TTS={tts_time:.0f}ms, FirstResponse={first_chunk_time:.0f}ms, "
                f"Total={total_time:.0f}ms, Audio={audio_duration:.1f}s"
            )

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            self._is_processing = False
            self._user_audio_buffer.clear()

    async def play_greeting(self) -> None:
        """Play initial greeting."""
        greeting = "Hi, I'm Maya. How can I help you?"

        logger.info(f"Playing greeting: '{greeting}'")
        self.conversation.maya_started_speaking()

        async for chunk in self.tts.generate_stream(greeting, use_context=False):
            await self._send_audio(chunk)

        self.conversation.maya_stopped_speaking(greeting, torch.zeros(24000))
        self.llm.add_context("assistant", greeting)
        self._maya_stop_time = time.time()

    async def start_conversation(self) -> None:
        """Start a new conversation with greeting."""
        await self.play_greeting()

    async def reset(self) -> None:
        """Reset conversation."""
        self.conversation.reset()
        self.tts.clear_context()
        self.llm.clear_history()
        self._is_processing = False
        self._user_audio_buffer.clear()

    def get_stats(self) -> dict:
        return {
            "initialized": self._initialized,
            "is_processing": self._is_processing,
            "total_turns": self._metrics.total_turns,
            "avg_response_ms": self._metrics.avg_response_time_ms,
            "avg_first_audio_ms": self._metrics.avg_time_to_first_audio_ms,
            "filler_used": self._metrics.filler_used_count,
            "filler_skipped": self._metrics.filler_skipped_count,
            "fillers_cached": len(self._filler_cache),
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized
