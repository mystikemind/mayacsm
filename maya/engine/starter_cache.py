"""
Starter Cache - Pre-generated Response Starters for Zero-Latency First Audio

THE HYBRID APPROACH:
1. Pre-generate common response starters with FULL 32 codebooks (high quality)
2. When user finishes speaking, IMMEDIATELY start playing cached starter
3. While starter plays (~500-800ms), generate actual response (also 32 codebooks)
4. Crossfade from cached starter to generated response

RESULT:
- Perceived latency: ~100ms (time to select and start playing cached audio)
- Audio quality: Full 32 codebooks throughout (no muffling)
- No breaks: Cached audio buffers while real generation catches up
"""

import torch
import torchaudio
import logging
import os
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Common response starters - these cover 80%+ of natural responses
# LONGER PHRASES = MORE BUFFER TIME for continuation generation
# Target: 800-1200ms of audio per starter to bridge the gap
STARTER_PHRASES = [
    # Thinking/processing starters (longer for more buffer)
    "well let me think",
    "hmm let me see",
    "oh i see",
    "ah yes",
    "so basically",

    # Affirmative starters
    "yes of course",
    "yeah definitely",
    "sure thing",
    "right so",
    "okay so",

    # Transitional starters (longer)
    "i think that",
    "let me explain",
    "thats a good",
    "i see what you",
    "you know what",

    # Question acknowledgment
    "good question let me",
    "thats interesting",
    "interesting question",
]

# Map LLM response patterns to starters
# Match the beginning of LLM responses to appropriate cached starters
STARTER_MAPPING = {
    # Thinking/processing patterns
    "well": "well let me think",
    "well,": "well let me think",
    "hmm": "hmm let me see",
    "oh i see": "oh i see",
    "oh": "oh i see",
    "ah": "ah yes",
    "so": "so basically",

    # Affirmative patterns
    "yes": "yes of course",
    "yes,": "yes of course",
    "yeah": "yeah definitely",
    "sure": "sure thing",
    "right": "right so",
    "okay": "okay so",
    "ok": "okay so",

    # Transitional patterns
    "i think": "i think that",
    "let me": "let me explain",
    "that's": "thats a good",
    "that is": "thats a good",
    "i see": "i see what you",
    "you know": "you know what",

    # Question acknowledgment
    "good question": "good question let me",
    "interesting": "thats interesting",
}


@dataclass
class CachedStarter:
    """A pre-generated audio starter."""
    text: str
    audio: torch.Tensor
    duration_ms: float


class StarterCache:
    """
    Pre-generated response starters for instant first audio.

    Usage:
        cache = StarterCache()
        cache.initialize(tts_engine)  # Pre-generates all starters

        # When responding:
        starter_audio, starter_text = cache.get_starter("I think that's interesting")
        # Start playing starter_audio immediately
        # Generate rest of response without starter_text prefix
    """

    CACHE_DIR = "/home/ec2-user/SageMaker/project_maya/cache/starters"
    SAMPLE_RATE = 24000

    def __init__(self):
        self._starters: Dict[str, CachedStarter] = {}
        self._initialized = False
        self._device = "cuda"

    def initialize(self, tts_engine) -> None:
        """
        Initialize the cache by pre-generating all starters.

        This should be called during server startup.
        Takes ~30-60 seconds but only runs once.
        """
        if self._initialized:
            return

        os.makedirs(self.CACHE_DIR, exist_ok=True)

        logger.info(f"Initializing starter cache with {len(STARTER_PHRASES)} phrases...")
        start_time = time.time()

        for phrase in STARTER_PHRASES:
            cache_path = os.path.join(self.CACHE_DIR, f"{phrase.replace(' ', '_')}.pt")

            # Try to load from disk cache first
            if os.path.exists(cache_path):
                try:
                    cached_data = torch.load(cache_path, weights_only=True)
                    audio = cached_data["audio"].to(self._device)
                    self._starters[phrase] = CachedStarter(
                        text=phrase,
                        audio=audio,
                        duration_ms=len(audio) / self.SAMPLE_RATE * 1000
                    )
                    logger.debug(f"Loaded cached starter: '{phrase}' ({self._starters[phrase].duration_ms:.0f}ms)")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cached starter '{phrase}': {e}")

            # Generate if not cached
            try:
                logger.info(f"Generating starter: '{phrase}'...")

                # Generate with full quality (32 codebooks via standard generate)
                audio = tts_engine.generate(phrase, use_context=False)

                # Normalize audio
                audio = audio - audio.mean()
                peak = audio.abs().max()
                if peak > 0:
                    audio = audio * (0.5 / peak)  # -6dB

                duration_ms = len(audio) / self.SAMPLE_RATE * 1000

                self._starters[phrase] = CachedStarter(
                    text=phrase,
                    audio=audio,
                    duration_ms=duration_ms
                )

                # Save to disk cache
                torch.save({"audio": audio.cpu()}, cache_path)
                logger.info(f"Generated and cached starter: '{phrase}' ({duration_ms:.0f}ms)")

            except Exception as e:
                logger.error(f"Failed to generate starter '{phrase}': {e}")

        elapsed = time.time() - start_time
        logger.info(f"Starter cache initialized with {len(self._starters)} phrases in {elapsed:.1f}s")
        self._initialized = True

    def get_starter(self, response_text: str) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """
        Get the appropriate starter audio for a response.

        Args:
            response_text: The full LLM response text

        Returns:
            (audio, matched_text) or (None, None) if no match
        """
        if not self._initialized or not self._starters:
            return None, None

        response_lower = response_text.lower().strip()

        # Find matching starter
        for pattern, starter_key in STARTER_MAPPING.items():
            if response_lower.startswith(pattern):
                if starter_key in self._starters:
                    starter = self._starters[starter_key]
                    return starter.audio.clone(), starter.text

        # Default: use a thinking filler if available
        if "hmm" in self._starters:
            starter = self._starters["hmm"]
            return starter.audio.clone(), None  # None = don't strip from response

        return None, None

    def get_continuation_text(self, response_text: str, matched_starter: Optional[str]) -> str:
        """
        Get the text that should be generated after the starter.

        If we matched "I think" and the response is "I think that's great",
        we only need to generate "that's great".
        """
        if matched_starter is None:
            return response_text

        response_lower = response_text.lower().strip()
        starter_lower = matched_starter.lower()

        if response_lower.startswith(starter_lower):
            continuation = response_text[len(matched_starter):].strip()
            return continuation if continuation else response_text

        return response_text

    def crossfade_audio(
        self,
        starter_audio: torch.Tensor,
        continuation_audio: torch.Tensor,
        crossfade_ms: int = 50
    ) -> torch.Tensor:
        """
        Crossfade from starter audio to continuation audio.

        Creates a seamless transition between pre-generated starter
        and real-time generated continuation.
        """
        crossfade_samples = int(crossfade_ms * self.SAMPLE_RATE / 1000)

        if len(starter_audio) < crossfade_samples or len(continuation_audio) < crossfade_samples:
            # Too short for crossfade, just concatenate
            return torch.cat([starter_audio, continuation_audio])

        # Create crossfade curves (equal power)
        t = torch.linspace(0, 1, crossfade_samples, device=starter_audio.device)
        fade_out = torch.cos(t * 3.14159 / 2)
        fade_in = torch.sin(t * 3.14159 / 2)

        # Trim starter (remove last crossfade_samples)
        starter_trimmed = starter_audio[:-crossfade_samples]

        # Get crossfade regions
        starter_tail = starter_audio[-crossfade_samples:]
        continuation_head = continuation_audio[:crossfade_samples]

        # Apply crossfade
        crossfaded = starter_tail * fade_out + continuation_head * fade_in

        # Combine: starter_trimmed + crossfaded + continuation_rest
        result = torch.cat([
            starter_trimmed,
            crossfaded,
            continuation_audio[crossfade_samples:]
        ])

        return result

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._starters:
            return {"initialized": False, "count": 0}

        total_duration = sum(s.duration_ms for s in self._starters.values())
        avg_duration = total_duration / len(self._starters)

        return {
            "initialized": self._initialized,
            "count": len(self._starters),
            "total_duration_ms": total_duration,
            "avg_duration_ms": avg_duration,
            "phrases": list(self._starters.keys())
        }


# Global singleton
_starter_cache: Optional[StarterCache] = None

def get_starter_cache() -> StarterCache:
    """Get the global starter cache instance."""
    global _starter_cache
    if _starter_cache is None:
        _starter_cache = StarterCache()
    return _starter_cache
