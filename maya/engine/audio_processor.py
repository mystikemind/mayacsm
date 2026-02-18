"""
Stateful Audio Processor for Click-Free Streaming Enhancement.

This maintains filter state across chunks to prevent discontinuities.
Uses scipy's sosfilt with zi/zf state parameters for continuous filtering.

This is how professional audio DSP works - stateful processing that
maintains phase continuity across buffer boundaries.
"""

import numpy as np
import scipy.signal as signal
from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class FilterState:
    """State container for multi-stage filtering."""
    highpass_zi: Optional[np.ndarray] = None
    lowmid_zi: Optional[np.ndarray] = None
    presence_zi: Optional[np.ndarray] = None
    air_zi: Optional[np.ndarray] = None
    last_sample: float = 0.0  # For DC offset tracking


class StatefulAudioProcessor:
    """
    Stateful audio enhancement that maintains continuity across chunks.

    Unlike per-chunk processing which causes clicks at boundaries, this
    maintains filter states between calls, ensuring smooth transitions.

    Features:
    - Stateful high-pass filter (removes rumble/DC)
    - Stateful presence boost (2-5kHz for clarity)
    - Stateful air boost (6-10kHz for brightness)
    - Consistent peak normalization
    - Soft limiting for safety

    Usage:
        processor = StatefulAudioProcessor()
        for chunk in audio_chunks:
            processed = processor.process(chunk)
            yield processed
        processor.reset()  # Call between utterances
    """

    SAMPLE_RATE = 24000

    def __init__(self):
        self._state = FilterState()
        self._initialized = False

        # Pre-compute filter coefficients (SOS format for stability)
        self._init_filters()

    def _init_filters(self):
        """Initialize filter coefficients in SOS format."""
        nyq = self.SAMPLE_RATE / 2

        # Gentle high-pass at 80Hz - just removes DC and very low rumble
        # More aggressive filtering was causing issues
        self._hp_sos = signal.butter(2, 80/nyq, btype='high', output='sos')

        self._initialized = True

    def reset(self):
        """Reset filter states - call between utterances."""
        self._state = FilterState()

    def process(self, audio: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Process audio chunk with MINIMAL safe enhancement.

        After extensive testing, aggressive enhancement causes more clicks.
        This version uses only gentle high-pass to remove DC/rumble.
        The audio may be slightly muffled but will be artifact-free.

        Args:
            audio: Input audio tensor
            normalize: Whether to apply peak normalization

        Returns:
            Processed audio tensor (click-free!)
        """
        if len(audio) < 100:
            return audio

        device = audio.device
        audio_np = audio.cpu().numpy().astype(np.float64)

        # === ONLY STAGE: Gentle high-pass at 80Hz ===
        # Just remove DC offset and very low rumble
        # No presence/air boost - these cause clicks
        if self._state.highpass_zi is None:
            self._state.highpass_zi = signal.sosfilt_zi(self._hp_sos) * audio_np[0]

        audio_np, self._state.highpass_zi = signal.sosfilt(
            self._hp_sos, audio_np, zi=self._state.highpass_zi
        )

        # === Consistent normalization ===
        if normalize:
            # Remove any residual DC
            audio_np = audio_np - np.mean(audio_np)

            # Peak normalize to consistent level
            TARGET_PEAK = 0.7
            peak = np.max(np.abs(audio_np))
            if peak > 1e-6:
                audio_np = audio_np * (TARGET_PEAK / peak)

            # Soft limit for safety
            LIMIT = 0.9
            if np.max(np.abs(audio_np)) > LIMIT:
                audio_np = np.tanh(audio_np / LIMIT) * LIMIT

        return torch.from_numpy(audio_np.astype(np.float32)).to(device)

    def process_complete(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process complete audio (non-streaming).

        For complete audio, we can use zero-phase filtering (filtfilt)
        which has no edge effects at all.

        Args:
            audio: Complete audio tensor

        Returns:
            Enhanced audio tensor
        """
        if len(audio) < 1000:
            return audio

        device = audio.device
        audio_np = audio.cpu().numpy().astype(np.float64)
        nyq = self.SAMPLE_RATE / 2

        # High-pass at 100Hz (zero-phase)
        b, a = signal.butter(2, 100/nyq, btype='high')
        audio_np = signal.filtfilt(b, a, audio_np)

        # Low-mid cut 200-400Hz (zero-phase)
        b, a = signal.butter(2, [200/nyq, 400/nyq], btype='band')
        lowmid = signal.filtfilt(b, a, audio_np)
        audio_np = audio_np - 0.3 * lowmid

        # Presence boost (zero-phase) - STRONG
        b, a = signal.butter(2, [2000/nyq, 5000/nyq], btype='band')
        presence = signal.filtfilt(b, a, audio_np)
        audio_np = audio_np + 0.8 * presence

        # Air boost (zero-phase) - STRONG
        b, a = signal.butter(1, 6000/nyq, btype='high')
        air = signal.filtfilt(b, a, audio_np)
        audio_np = audio_np + 0.5 * air

        # Normalize
        audio_np = audio_np - np.mean(audio_np)
        peak = np.max(np.abs(audio_np))
        if peak > 1e-6:
            audio_np = audio_np * (0.7 / peak)

        # Soft limit
        if np.max(np.abs(audio_np)) > 0.9:
            audio_np = np.tanh(audio_np / 0.9) * 0.9

        return torch.from_numpy(audio_np.astype(np.float32)).to(device)


# Global instance for streaming use
_processor: Optional[StatefulAudioProcessor] = None


def get_processor() -> StatefulAudioProcessor:
    """Get or create the global stateful processor."""
    global _processor
    if _processor is None:
        _processor = StatefulAudioProcessor()
    return _processor


def process_chunk(audio: torch.Tensor) -> torch.Tensor:
    """Process a single chunk with stateful enhancement."""
    return get_processor().process(audio)


def reset_processor():
    """Reset processor state between utterances."""
    if _processor is not None:
        _processor.reset()


def repair_clicks(audio: torch.Tensor, threshold: float = 0.4) -> torch.Tensor:
    """
    Detect and repair severe clicks in audio.

    This is a last-resort repair for when the model generates
    audio with severe discontinuities (>0.4). It uses linear
    interpolation to smooth the click points.

    Args:
        audio: Input audio tensor
        threshold: Discontinuity threshold for repair (default 0.4)

    Returns:
        Repaired audio tensor
    """
    if len(audio) < 10:
        return audio

    device = audio.device
    audio_np = audio.cpu().numpy().astype(np.float64)

    # Find severe discontinuities
    diff = np.abs(np.diff(audio_np))
    click_indices = np.where(diff > threshold)[0]

    if len(click_indices) == 0:
        return audio  # No clicks to repair

    # Repair each click with local interpolation
    for idx in click_indices:
        # Get surrounding context (avoid edges)
        start = max(0, idx - 5)
        end = min(len(audio_np), idx + 7)

        if end - start < 4:
            continue

        # Linear interpolate across the click
        # Use 3 samples on each side
        left_idx = max(0, idx - 3)
        right_idx = min(len(audio_np) - 1, idx + 4)

        left_val = audio_np[left_idx]
        right_val = audio_np[right_idx]

        # Interpolate between left and right
        num_samples = right_idx - left_idx
        if num_samples > 0:
            interp = np.linspace(left_val, right_val, num_samples)
            audio_np[left_idx:right_idx] = interp

    return torch.from_numpy(audio_np.astype(np.float32)).to(device)
