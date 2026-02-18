"""
Text-to-Speech Engine - Sesame CSM-1B

Generates human-like speech with conversation context awareness.

THE KEY TO HUMAN-LIKE VOICE:
- CSM uses conversation context (audio + text) to match emotional tone
- Same model that powers Sesame Maya
"""

import torch
import torchaudio
import logging
from typing import Optional, List, Tuple
import time
import sys

# Add CSM to path
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from ..config import TTS, AUDIO, CSM_ROOT

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Sesame CSM-1B wrapper for voice synthesis.

    THE MAGIC:
    - Pass conversation context (previous audio + text)
    - CSM automatically adjusts prosody to match conversation tone
    - User sounds sad → Maya sounds empathetic (automatically)
    - User sounds excited → Maya matches energy (automatically)
    """

    SAMPLE_RATE = 24000

    def __init__(self):
        self._generator = None
        self._initialized = False

        # Voice prompt (establishes Maya's voice identity)
        self._voice_prompt: Optional[object] = None

        # Conversation context for CSM
        self._context: List[object] = []

        # Performance tracking
        self._total_generations = 0
        self._total_audio_seconds = 0.0
        self._total_time = 0.0

    def initialize(self) -> None:
        """Load CSM-1B model."""
        if self._initialized:
            return

        logger.info("Loading Sesame CSM-1B...")
        start = time.time()

        try:
            from generator import load_csm_1b

            self._generator = load_csm_1b(device=TTS.device)

            elapsed = time.time() - start
            logger.info(f"CSM-1B loaded in {elapsed:.1f}s")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load CSM: {e}")
            raise

    def set_voice_prompt(self, text: str, audio: torch.Tensor) -> None:
        """
        Set Maya's voice identity prompt.

        This establishes consistent voice characteristics across all generations.

        Args:
            text: What Maya says in the prompt
            audio: Audio of Maya saying the text
        """
        if not self._initialized:
            self.initialize()

        from generator import Segment

        if audio.dim() > 1:
            audio = audio.squeeze()

        self._voice_prompt = Segment(
            text=text,
            speaker=TTS.speaker_id,
            audio=audio
        )

        logger.info("Voice prompt set for Maya")

    def add_context(self, text: str, audio: torch.Tensor, is_user: bool = True) -> None:
        """
        Add a turn to conversation context.

        Args:
            text: What was said
            audio: Audio of what was said
            is_user: True if user, False if Maya
        """
        if not self._initialized:
            self.initialize()

        from generator import Segment

        if audio.dim() > 1:
            audio = audio.squeeze()

        speaker_id = 1 if is_user else TTS.speaker_id

        segment = Segment(
            text=text,
            speaker=speaker_id,
            audio=audio
        )

        self._context.append(segment)

        # Keep only last N turns
        if len(self._context) > TTS.context_turns:
            self._context = self._context[-TTS.context_turns:]

        logger.debug(f"Context now has {len(self._context)} turns")

    def generate(
        self,
        text: str,
        use_context: bool = True
    ) -> torch.Tensor:
        """
        Generate speech for text.

        Args:
            text: What Maya should say
            use_context: Whether to use conversation context

        Returns:
            Audio tensor at 24kHz
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        # Build context
        context = []

        # Add voice prompt first (establishes voice identity)
        if self._voice_prompt is not None:
            context.append(self._voice_prompt)

        # Add conversation context
        if use_context and self._context:
            context.extend(self._context)

        logger.debug(f"Generating '{text[:50]}...' with {len(context)} context segments")

        # Generate audio
        audio = self._generator.generate(
            text=text,
            speaker=TTS.speaker_id,
            context=context,
            max_audio_length_ms=TTS.max_audio_length_ms,
        )

        # Track performance
        elapsed = time.time() - start
        audio_duration = len(audio) / self.SAMPLE_RATE

        self._total_generations += 1
        self._total_audio_seconds += audio_duration
        self._total_time += elapsed

        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        logger.debug(
            f"Generated {audio_duration:.2f}s audio in {elapsed:.2f}s "
            f"(RTF={rtf:.2f}x)"
        )

        return audio

    def generate_short(self, text: str) -> torch.Tensor:
        """
        Generate short utterance (fillers, backchannels).

        Args:
            text: Short text to speak

        Returns:
            Audio tensor
        """
        if not self._initialized:
            self.initialize()

        # No context for fillers - should be neutral
        audio = self._generator.generate(
            text=text,
            speaker=TTS.speaker_id,
            context=[self._voice_prompt] if self._voice_prompt else [],
            max_audio_length_ms=3000,  # 3 seconds max for fillers
        )

        return audio

    def clear_context(self) -> None:
        """Clear conversation context (keep voice prompt)."""
        self._context.clear()
        logger.info("TTS context cleared")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def get_stats(self) -> dict:
        """Get performance statistics."""
        avg_rtf = 0.0
        if self._total_audio_seconds > 0:
            avg_rtf = self._total_time / self._total_audio_seconds

        return {
            "total_generations": self._total_generations,
            "total_audio_seconds": self._total_audio_seconds,
            "total_time": self._total_time,
            "average_rtf": avg_rtf,
            "context_turns": len(self._context),
        }
