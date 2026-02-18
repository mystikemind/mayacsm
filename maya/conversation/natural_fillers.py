"""
Natural Filler System - Like Sesame Maya

Plays SHORT, NATURAL fillers that sound like human speech patterns:
- "Yeahhh..." (agreement)
- "Hmm..." (thinking)
- "Ohhh..." (empathy)
- "Okay..." (acknowledgment)

These give CSM time to generate while user perceives immediate response.
"""

import torch
import torchaudio
import random
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto

from ..config import FILLERS_DIR, AUDIO

logger = logging.getLogger(__name__)


class FillerType(Enum):
    """Types of natural fillers based on conversation context."""
    THINKING = auto()      # "Hmm..." - for questions
    AGREEMENT = auto()     # "Yeahhh..." - for statements
    EMPATHY = auto()       # "Ohhh..." - for emotional content
    ACKNOWLEDGMENT = auto() # "Okay..." - for instructions
    NEUTRAL = auto()       # "Mhmm..." - default


@dataclass
class NaturalFiller:
    """A natural filler with metadata."""
    name: str
    audio: torch.Tensor
    duration_ms: float
    filler_type: FillerType


# Filler definitions: (name_prefix, filler_type)
FILLER_DEFINITIONS = {
    "yeah": FillerType.AGREEMENT,
    "hmm": FillerType.THINKING,
    "oh": FillerType.EMPATHY,
    "okay": FillerType.ACKNOWLEDGMENT,
    "right": FillerType.AGREEMENT,
    "mhmm": FillerType.NEUTRAL,
    "uh": FillerType.THINKING,
    "ah": FillerType.EMPATHY,
}


class NaturalFillerSystem:
    """
    Natural filler system for seamless response.

    Selects contextually appropriate short fillers that sound
    like natural human speech patterns (how Sesame does it).
    """

    def __init__(self):
        self._fillers: Dict[str, NaturalFiller] = {}
        self._by_type: Dict[FillerType, List[NaturalFiller]] = {
            t: [] for t in FillerType
        }
        self._initialized = False
        self._history: List[str] = []  # Avoid repeating same filler

    def initialize(self) -> None:
        """Load pre-generated natural fillers."""
        if self._initialized:
            return

        if not FILLERS_DIR.exists():
            logger.warning(f"Fillers directory not found: {FILLERS_DIR}")
            return

        # Load all WAV files
        for wav_path in FILLERS_DIR.glob("*.wav"):
            name = wav_path.stem

            # Determine filler type from name prefix
            filler_type = FillerType.NEUTRAL
            for prefix, ft in FILLER_DEFINITIONS.items():
                if name.startswith(prefix):
                    filler_type = ft
                    break

            try:
                audio, sr = torchaudio.load(str(wav_path))
                if sr != AUDIO.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, AUDIO.sample_rate)
                audio = audio.squeeze(0)

                duration_ms = len(audio) / AUDIO.sample_rate * 1000

                filler = NaturalFiller(
                    name=name,
                    audio=audio,
                    duration_ms=duration_ms,
                    filler_type=filler_type
                )

                self._fillers[name] = filler
                self._by_type[filler_type].append(filler)

                logger.debug(f"Loaded filler: {name} ({filler_type.name}, {duration_ms:.0f}ms)")

            except Exception as e:
                logger.warning(f"Failed to load filler {wav_path}: {e}")

        self._initialized = True
        logger.info(f"Natural filler system initialized with {len(self._fillers)} fillers")

    def get_filler(self, filler_type: FillerType = FillerType.NEUTRAL) -> Tuple[torch.Tensor, str]:
        """
        Get a contextually appropriate filler.

        Args:
            filler_type: Type of filler based on conversation context

        Returns:
            (audio_tensor, filler_name)
        """
        if not self._initialized:
            self.initialize()

        fillers = self._by_type.get(filler_type, [])

        # Fall back to neutral if type not available
        if not fillers:
            fillers = self._by_type.get(FillerType.NEUTRAL, [])

        # Avoid recent repeats
        available = [f for f in fillers if f.name not in self._history[-3:]]
        if not available:
            self._history.clear()
            available = fillers

        if not available:
            # Emergency fallback
            logger.warning("No fillers available, returning silence")
            return torch.zeros(int(AUDIO.sample_rate * 0.3)), "silence"

        filler = random.choice(available)
        self._history.append(filler.name)

        logger.debug(f"Selected filler: {filler.name} ({filler.filler_type.name})")
        return filler.audio.clone(), filler.name

    def get_thinking_filler(self) -> Tuple[torch.Tensor, str]:
        """Get a thinking filler (for questions)."""
        return self.get_filler(FillerType.THINKING)

    def get_agreement_filler(self) -> Tuple[torch.Tensor, str]:
        """Get an agreement filler (for statements)."""
        return self.get_filler(FillerType.AGREEMENT)

    def get_empathy_filler(self) -> Tuple[torch.Tensor, str]:
        """Get an empathy filler (for emotional content)."""
        return self.get_filler(FillerType.EMPATHY)

    def get_acknowledgment_filler(self) -> Tuple[torch.Tensor, str]:
        """Get an acknowledgment filler (for instructions)."""
        return self.get_filler(FillerType.ACKNOWLEDGMENT)

    def get_random_filler(self) -> Tuple[torch.Tensor, str]:
        """Get any random filler for variety (used when playing multiple fillers)."""
        all_fillers = list(self._fillers.values())

        # Avoid recent repeats
        available = [f for f in all_fillers if f.name not in self._history[-3:]]
        if not available:
            self._history.clear()
            available = all_fillers

        if not available:
            return torch.zeros(int(AUDIO.sample_rate * 0.3)), "silence"

        filler = random.choice(available)
        self._history.append(filler.name)
        return filler.audio.clone(), filler.name

    def select_filler_for_input(self, user_input: str) -> Tuple[torch.Tensor, str]:
        """
        Select appropriate filler based on user input.

        Simple heuristics to determine filler type.
        """
        user_input = user_input.lower().strip()

        # Question words → thinking
        if any(word in user_input for word in ["what", "how", "why", "when", "where", "who", "which", "?"]):
            return self.get_thinking_filler()

        # Emotional words → empathy
        if any(word in user_input for word in ["sad", "happy", "upset", "worried", "excited", "love", "hate", "feel"]):
            return self.get_empathy_filler()

        # Instruction words → acknowledgment
        if any(word in user_input for word in ["please", "can you", "could you", "help me", "tell me", "show me"]):
            return self.get_acknowledgment_filler()

        # Statement with period → agreement
        if user_input.endswith(".") and len(user_input) > 20:
            return self.get_agreement_filler()

        # Default → neutral or thinking
        return random.choice([self.get_thinking_filler, self.get_filler])()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_stats(self) -> dict:
        """Get filler statistics."""
        type_counts = {t.name: len(f) for t, f in self._by_type.items()}
        return {
            "total_fillers": len(self._fillers),
            "by_type": type_counts,
            "history_length": len(self._history),
        }
