"""
Filler System - Zero Perceived Latency

THE GENIUS TRICK:
- When user stops speaking, IMMEDIATELY play a filler
- While filler plays (5+ seconds), generate the real response
- Crossfade filler into response
- User perceives INSTANT response

Types:
- Thinking: "Hmm...", "Let me think..." (play when user stops)
- Backchannels: "Mm-hmm", "Yeah" (play while user speaks)
"""

import torch
import torchaudio
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import asyncio

from ..config import FILLER, FILLERS_DIR, AUDIO

logger = logging.getLogger(__name__)


class FillerType(Enum):
    """Types of fillers."""
    THINKING = auto()      # Play when user stops speaking
    BACKCHANNEL = auto()   # Play while user is speaking
    TRANSITION = auto()    # Play between topics
    EMPATHY = auto()       # Play for emotional moments


@dataclass
class Filler:
    """A single filler with metadata."""
    name: str
    type: FillerType
    text: str
    audio: torch.Tensor
    duration_ms: float


class FillerSystem:
    """
    Pre-generated filler audio for instant playback.

    THE KEY INSIGHT:
    - Human conversation has natural "thinking" sounds
    - Playing these INSTANTLY when user stops creates perception of immediate response
    - 5+ second fillers give enough time to generate real response
    """

    def __init__(self):
        self._fillers: Dict[str, Filler] = {}
        self._by_type: Dict[FillerType, List[Filler]] = {
            t: [] for t in FillerType
        }
        self._initialized = False

        # Anti-repeat tracking
        self._last_thinking: Optional[str] = None
        self._last_backchannel: Optional[str] = None
        self._thinking_history: List[str] = []

    async def initialize(self, tts_engine=None) -> None:
        """
        Initialize filler system.

        If filler audio files don't exist, generate them with TTS.
        """
        FILLERS_DIR.mkdir(parents=True, exist_ok=True)

        # Define all fillers
        filler_definitions = [
            # THINKING FILLERS (5+ seconds of natural thinking)
            ("thinking_1", FillerType.THINKING, "Hmm, that's a really interesting question... let me think about that for a moment..."),
            ("thinking_2", FillerType.THINKING, "Well, you know... there are a few things I want to say about that..."),
            ("thinking_3", FillerType.THINKING, "Let me see... okay, so here's what I'm thinking..."),
            ("thinking_4", FillerType.THINKING, "So... that's actually something I find really fascinating... let me share my thoughts..."),
            ("thinking_5", FillerType.THINKING, "Okay, so... I want to make sure I give you a good answer here..."),
            ("thinking_6", FillerType.THINKING, "Hmm... you know, that reminds me of something... let me think..."),

            # BACKCHANNELS (short, play while user speaks)
            ("backchannel_1", FillerType.BACKCHANNEL, "Mm-hmm."),
            ("backchannel_2", FillerType.BACKCHANNEL, "Yeah."),
            ("backchannel_3", FillerType.BACKCHANNEL, "Right."),
            ("backchannel_4", FillerType.BACKCHANNEL, "I see."),
            ("backchannel_5", FillerType.BACKCHANNEL, "Uh-huh."),

            # TRANSITIONS
            ("transition_1", FillerType.TRANSITION, "Okay, so..."),
            ("transition_2", FillerType.TRANSITION, "Alright..."),
            ("transition_3", FillerType.TRANSITION, "So anyway..."),

            # EMPATHY
            ("empathy_1", FillerType.EMPATHY, "Oh... I understand..."),
            ("empathy_2", FillerType.EMPATHY, "I see... that sounds really..."),
        ]

        for name, ftype, text in filler_definitions:
            filepath = FILLERS_DIR / f"{name}.wav"

            if filepath.exists():
                # Load existing
                audio, sr = torchaudio.load(str(filepath))
                if sr != AUDIO.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, AUDIO.sample_rate)
                audio = audio.squeeze(0)
                logger.debug(f"Loaded filler: {name}")

            elif tts_engine is not None:
                # Generate with TTS
                logger.info(f"Generating filler: {name}")
                audio = await self._generate_filler(tts_engine, text)

                # Save for future
                torchaudio.save(
                    str(filepath),
                    audio.unsqueeze(0).cpu(),
                    AUDIO.sample_rate
                )
            else:
                logger.warning(f"Skipping filler {name} - no TTS engine")
                continue

            # Create filler object
            duration_ms = len(audio) / AUDIO.sample_rate * 1000

            filler = Filler(
                name=name,
                type=ftype,
                text=text,
                audio=audio,
                duration_ms=duration_ms
            )

            self._fillers[name] = filler
            self._by_type[ftype].append(filler)

        self._initialized = True
        logger.info(f"Filler system initialized with {len(self._fillers)} fillers")

    async def _generate_filler(self, tts_engine, text: str) -> torch.Tensor:
        """Generate filler audio with TTS."""
        # Run in executor to not block
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: tts_engine.generate_short(text)
        )
        return audio

    def get_thinking_filler(self) -> Tuple[torch.Tensor, str]:
        """
        Get a thinking filler (play when user stops speaking).

        Returns:
            (audio, text) - Audio tensor and the text that was spoken

        Smart selection:
        - Never repeat the last filler
        - Cycle through all before repeating any
        """
        fillers = self._by_type[FillerType.THINKING]

        if not fillers:
            # Emergency fallback: 3 seconds of near-silence
            logger.warning("No thinking fillers available!")
            return torch.zeros(AUDIO.sample_rate * 3), ""

        # Filter out recently used
        available = [f for f in fillers if f.name not in self._thinking_history[-3:]]

        if not available:
            # Reset history
            self._thinking_history = []
            available = fillers

        # Select random
        filler = random.choice(available)
        self._thinking_history.append(filler.name)
        self._last_thinking = filler.name

        logger.debug(f"Selected thinking filler: {filler.name}")

        return filler.audio.clone(), filler.text

    def get_backchannel(self) -> Tuple[torch.Tensor, str]:
        """
        Get a backchannel (play while user speaks).

        Returns:
            (audio, text) at reduced volume
        """
        fillers = self._by_type[FillerType.BACKCHANNEL]

        if not fillers:
            return torch.zeros(int(AUDIO.sample_rate * 0.5)), ""

        # Don't repeat last
        available = [f for f in fillers if f.name != self._last_backchannel]
        if not available:
            available = fillers

        filler = random.choice(available)
        self._last_backchannel = filler.name

        # Reduce volume
        audio = filler.audio.clone() * FILLER.backchannel_volume

        return audio, filler.text

    def get_empathy_filler(self) -> Tuple[torch.Tensor, str]:
        """Get an empathy filler for emotional moments."""
        fillers = self._by_type[FillerType.EMPATHY]

        if not fillers:
            return self.get_thinking_filler()

        filler = random.choice(fillers)
        return filler.audio.clone(), filler.text

    def get_transition_filler(self) -> Tuple[torch.Tensor, str]:
        """Get a transition filler."""
        fillers = self._by_type[FillerType.TRANSITION]

        if not fillers:
            return self.get_thinking_filler()

        filler = random.choice(fillers)
        return filler.audio.clone(), filler.text

    def get_contextual_filler(self, transcript: str) -> Tuple[torch.Tensor, str]:
        """
        Get a filler appropriate for the user's message content.

        Analyzes the transcript to select the most natural filler:
        - Questions → Thinking fillers ("Hmm, good question...")
        - Sad/emotional → Empathy fillers ("Oh, I understand...")
        - Excited/positive → Transition fillers ("Oh wow...")
        - General → Random thinking filler

        Also ensures variety by tracking history.
        """
        transcript_lower = transcript.lower().strip()

        # Detect question
        is_question = (
            "?" in transcript or
            transcript_lower.startswith(("what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should", "do", "does", "is", "are", "will"))
        )

        # Detect emotional/sad content
        sad_keywords = ["sad", "upset", "depressed", "angry", "frustrated", "worried", "anxious", "scared", "hurt", "lost", "died", "sick", "bad day", "terrible", "awful", "hate", "crying"]
        is_emotional = any(kw in transcript_lower for kw in sad_keywords)

        # Detect positive/excited content
        positive_keywords = ["excited", "happy", "great", "amazing", "awesome", "wonderful", "fantastic", "love", "best", "won", "got the job", "engaged", "married", "baby", "promotion"]
        is_positive = any(kw in transcript_lower for kw in positive_keywords)

        # Select appropriate filler type
        if is_emotional:
            logger.debug(f"Detected emotional content, using empathy filler")
            return self.get_empathy_filler()
        elif is_positive:
            logger.debug(f"Detected positive content, using transition filler")
            return self.get_transition_filler()
        elif is_question:
            logger.debug(f"Detected question, using thinking filler")
            return self.get_thinking_filler()
        else:
            # Default: thinking filler with variety
            logger.debug(f"General content, using thinking filler")
            return self.get_thinking_filler()

    @staticmethod
    def crossfade_audio(
        audio1: torch.Tensor,
        audio2: torch.Tensor,
        crossfade_ms: float = 100.0,
        sample_rate: int = 24000
    ) -> torch.Tensor:
        """
        Crossfade between two audio segments for smooth transition.

        Args:
            audio1: First audio (e.g., filler ending)
            audio2: Second audio (e.g., response starting)
            crossfade_ms: Crossfade duration in milliseconds
            sample_rate: Audio sample rate

        Returns:
            Combined audio with smooth crossfade
        """
        crossfade_samples = int(crossfade_ms * sample_rate / 1000)

        # Ensure 1D
        if audio1.dim() > 1:
            audio1 = audio1.squeeze()
        if audio2.dim() > 1:
            audio2 = audio2.squeeze()

        # If either is too short for crossfade, just concatenate
        if len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
            return torch.cat([audio1, audio2])

        # Create fade curves
        fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
        fade_in = torch.linspace(0.0, 1.0, crossfade_samples)

        # Apply fades
        audio1_end = audio1[-crossfade_samples:] * fade_out
        audio2_start = audio2[:crossfade_samples] * fade_in

        # Combine crossfade region
        crossfaded = audio1_end + audio2_start

        # Build final audio
        result = torch.cat([
            audio1[:-crossfade_samples],
            crossfaded,
            audio2[crossfade_samples:]
        ])

        return result

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def filler_count(self) -> int:
        return len(self._fillers)

    def get_stats(self) -> dict:
        """Get filler statistics."""
        return {
            "total_fillers": len(self._fillers),
            "thinking_count": len(self._by_type[FillerType.THINKING]),
            "backchannel_count": len(self._by_type[FillerType.BACKCHANNEL]),
        }
