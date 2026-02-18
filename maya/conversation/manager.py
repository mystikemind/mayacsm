"""
Conversation Manager - Context and State Tracking

Tracks the full conversation for:
- LLM context (text history)
- TTS context (audio + text for prosody matching)
- State management (who's speaking, timing)
"""

import torch
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
from enum import Enum, auto

from ..config import AUDIO

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Current state of the conversation."""
    IDLE = auto()           # No one speaking
    USER_SPEAKING = auto()  # User is talking
    MAYA_THINKING = auto()  # Maya is processing (filler playing)
    MAYA_SPEAKING = auto()  # Maya is responding
    INTERRUPTED = auto()    # User interrupted Maya


@dataclass
class Turn:
    """A single conversation turn."""
    speaker: str           # "user" or "maya"
    text: str              # What was said
    audio: Optional[torch.Tensor] = None  # Audio recording
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    # Optional metadata
    emotion: Optional[str] = None
    interrupted: bool = False


@dataclass
class ConversationStats:
    """Conversation statistics."""
    total_turns: int = 0
    user_turns: int = 0
    maya_turns: int = 0
    total_duration_ms: float = 0.0
    interruptions: int = 0
    avg_response_time_ms: float = 0.0


class ConversationManager:
    """
    Manages conversation state and context.

    Responsibilities:
    - Track who is speaking
    - Store conversation history (text + audio)
    - Provide context to LLM and TTS
    - Handle state transitions
    """

    def __init__(self, max_turns: int = 20):
        self._max_turns = max_turns
        self._turns: deque = deque(maxlen=max_turns)

        # Current state
        self._state = ConversationState.IDLE
        self._state_start_time: float = time.time()

        # Audio buffer for current speech
        self._audio_buffer: List[torch.Tensor] = []
        self._current_text: str = ""

        # Timing
        self._user_stop_time: Optional[float] = None
        self._response_times: List[float] = []

        # Session
        self._session_start = time.time()

    def user_started_speaking(self) -> None:
        """Called when VAD detects user started speaking."""
        previous_state = self._state

        self._state = ConversationState.USER_SPEAKING
        self._state_start_time = time.time()
        self._audio_buffer.clear()
        self._current_text = ""

        # Check if this is an interruption
        if previous_state == ConversationState.MAYA_SPEAKING:
            self._state = ConversationState.INTERRUPTED
            logger.info("User interrupted Maya")

        logger.debug(f"User started speaking (was {previous_state.name})")

    def buffer_audio(self, audio_chunk: torch.Tensor) -> None:
        """Buffer incoming user audio."""
        if self._state in (ConversationState.USER_SPEAKING, ConversationState.INTERRUPTED):
            self._audio_buffer.append(audio_chunk)

    def user_stopped_speaking(self, transcript: str) -> Turn:
        """
        Called when VAD detects user stopped speaking.

        Args:
            transcript: Transcribed text from STT

        Returns:
            Completed turn
        """
        # Calculate duration
        duration_ms = (time.time() - self._state_start_time) * 1000

        # Combine buffered audio
        audio = None
        if self._audio_buffer:
            audio = torch.cat(self._audio_buffer, dim=0)

        # Create turn
        turn = Turn(
            speaker="user",
            text=transcript,
            audio=audio,
            duration_ms=duration_ms,
            interrupted=(self._state == ConversationState.INTERRUPTED)
        )

        # Store turn
        self._turns.append(turn)

        # Update state
        self._state = ConversationState.MAYA_THINKING
        self._state_start_time = time.time()
        self._user_stop_time = time.time()
        self._audio_buffer.clear()

        logger.debug(f"User turn complete: '{transcript[:50]}...' ({duration_ms:.0f}ms)")

        return turn

    def maya_started_speaking(self) -> None:
        """Called when Maya starts her response."""
        self._state = ConversationState.MAYA_SPEAKING
        self._state_start_time = time.time()

        # Track response time
        if self._user_stop_time:
            response_time = (time.time() - self._user_stop_time) * 1000
            self._response_times.append(response_time)
            logger.debug(f"Response time: {response_time:.0f}ms")

    def maya_stopped_speaking(self, text: str, audio: torch.Tensor) -> Turn:
        """
        Called when Maya finishes her response.

        Args:
            text: What Maya said
            audio: Maya's audio

        Returns:
            Completed turn
        """
        duration_ms = (time.time() - self._state_start_time) * 1000

        turn = Turn(
            speaker="maya",
            text=text,
            audio=audio,
            duration_ms=duration_ms
        )

        self._turns.append(turn)

        self._state = ConversationState.IDLE
        self._state_start_time = time.time()
        self._user_stop_time = None

        logger.debug(f"Maya turn complete: '{text[:50]}...' ({duration_ms:.0f}ms)")

        return turn

    def get_llm_context(self) -> List[Dict[str, str]]:
        """
        Get conversation history for LLM.

        Returns:
            List of {"role": "user"|"assistant", "content": text}
        """
        context = []
        for turn in self._turns:
            role = "user" if turn.speaker == "user" else "assistant"
            context.append({
                "role": role,
                "content": turn.text
            })
        return context

    def get_tts_context(self) -> List[tuple]:
        """
        Get conversation history for TTS (audio + text).

        Returns:
            List of (text, audio, speaker_id) tuples
        """
        context = []
        for turn in self._turns:
            if turn.audio is not None:
                speaker_id = 1 if turn.speaker == "user" else 0
                context.append((turn.text, turn.audio, speaker_id))
        return context

    def get_recent_turns(self, n: int = 5) -> List[Turn]:
        """Get the N most recent turns."""
        return list(self._turns)[-n:]

    def get_full_transcript(self) -> str:
        """Get full conversation as text."""
        lines = []
        for turn in self._turns:
            speaker = "User" if turn.speaker == "user" else "Maya"
            lines.append(f"{speaker}: {turn.text}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear conversation history."""
        self._turns.clear()
        self._audio_buffer.clear()
        self._state = ConversationState.IDLE
        self._response_times.clear()
        logger.info("Conversation cleared")

    def reset(self) -> None:
        """Reset conversation state (alias for clear)."""
        self.clear()

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def is_user_speaking(self) -> bool:
        return self._state == ConversationState.USER_SPEAKING

    @property
    def is_maya_speaking(self) -> bool:
        return self._state == ConversationState.MAYA_SPEAKING

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def get_stats(self) -> ConversationStats:
        """Get conversation statistics."""
        user_turns = sum(1 for t in self._turns if t.speaker == "user")
        maya_turns = sum(1 for t in self._turns if t.speaker == "maya")
        total_duration = sum(t.duration_ms for t in self._turns)
        interruptions = sum(1 for t in self._turns if t.interrupted)

        avg_response = 0.0
        if self._response_times:
            avg_response = sum(self._response_times) / len(self._response_times)

        return ConversationStats(
            total_turns=len(self._turns),
            user_turns=user_turns,
            maya_turns=maya_turns,
            total_duration_ms=total_duration,
            interruptions=interruptions,
            avg_response_time_ms=avg_response
        )
