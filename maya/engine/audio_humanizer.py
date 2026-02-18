"""
Audio Humanizer - Add human-like micro-features to TTS output.

Based on research into what makes speech sound "alive":
1. Jitter (pitch micro-variations) - Natural voice has ~0.5% jitter
2. Shimmer (amplitude micro-variations) - Natural voice has ~1-3% shimmer
3. Breath insertion at natural pause points
4. Vocal fry at phrase endings
5. Formant micro-modulation

These features are present in natural human speech but MISSING from TTS output,
which is what makes TTS sound "robotic" even when prosody is correct.

References:
- Jitter/shimmer: speechprocessingbook.aalto.fi/Representations/Jitter_and_shimmer.html
- Breath insertion: arxiv:2402.00288 (Respiro)
- Vocal fry: HMM-based synthesis of creaky voice (Researchgate)
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Pre-recorded breath sound (synthesized - filtered noise with natural envelope)
# This is a gentle, quiet breath sound that occurs naturally between phrases
_BREATH_CACHE = None


def _generate_breath_sound(sample_rate: int = 24000, duration_ms: int = 200) -> np.ndarray:
    """
    Synthesize a natural-sounding breath sound.

    Breath = filtered noise with:
    - Bandpass 200-2000 Hz (throat resonance)
    - Natural attack/sustain/decay envelope
    - Low amplitude (background, not prominent)
    """
    global _BREATH_CACHE
    if _BREATH_CACHE is not None:
        return _BREATH_CACHE.copy()

    n_samples = int(sample_rate * duration_ms / 1000)

    # Generate pink-ish noise (more natural than white noise)
    noise = np.random.randn(n_samples).astype(np.float32)

    # Simple low-pass to make it more breath-like
    # Use a running average as simple filter
    kernel_size = int(sample_rate / 2000)  # ~2kHz cutoff
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        noise = np.convolve(noise, kernel, mode='same')

    # High-pass to remove rumble (above 200Hz)
    kernel_size_hp = int(sample_rate / 200)
    if kernel_size_hp > 1:
        hp_kernel = np.ones(kernel_size_hp) / kernel_size_hp
        low_freq = np.convolve(noise, hp_kernel, mode='same')
        noise = noise - low_freq

    # Natural breath envelope: quick attack, sustain, gentle decay
    envelope = np.ones(n_samples, dtype=np.float32)
    attack = int(n_samples * 0.1)   # 10% attack
    decay = int(n_samples * 0.4)    # 40% decay

    # Attack (fade in)
    envelope[:attack] = np.linspace(0, 1, attack)
    # Decay (fade out)
    envelope[-decay:] = np.linspace(1, 0, decay) ** 1.5  # Slightly concave decay

    breath = noise * envelope

    # Normalize to low amplitude (breaths are quiet!)
    peak = np.abs(breath).max()
    if peak > 0:
        breath = breath * (0.03 / peak)  # Very quiet - 3% of full scale

    _BREATH_CACHE = breath.copy()
    return breath


def add_jitter(audio: np.ndarray, jitter_percent: float = 0.3,
               sample_rate: int = 24000) -> np.ndarray:
    """
    Add pitch micro-variations (jitter) to make speech sound alive.

    Natural speech has ~0.5-1.0% jitter. We add 0.3% (subtle but perceptible).

    Implementation: Resample local segments by tiny random amounts.
    This changes local pitch without affecting overall duration significantly.

    Args:
        audio: Audio array
        jitter_percent: Amount of jitter (0.3 = 0.3% pitch variation)
        sample_rate: Sample rate

    Returns:
        Audio with added jitter
    """
    if len(audio) < sample_rate * 0.1:  # Skip very short audio
        return audio

    # Work on 20ms segments (one pitch cycle at ~50Hz)
    segment_size = int(sample_rate * 0.02)
    result = np.zeros_like(audio)

    pos_in = 0
    pos_out = 0

    while pos_in < len(audio) - segment_size:
        segment = audio[pos_in:pos_in + segment_size]

        # Random pitch shift for this segment
        shift_factor = 1.0 + np.random.normal(0, jitter_percent / 100)
        shift_factor = np.clip(shift_factor, 0.995, 1.005)  # Limit range

        # Resample segment
        new_len = int(len(segment) * shift_factor)
        if new_len > 0 and new_len < segment_size * 2:
            indices = np.linspace(0, len(segment) - 1, new_len)
            resampled = np.interp(indices, np.arange(len(segment)), segment)

            end = min(pos_out + len(resampled), len(result))
            write_len = end - pos_out
            if write_len > 0:
                result[pos_out:end] = resampled[:write_len]

        pos_in += segment_size
        pos_out += segment_size  # Keep output aligned

    # Copy remaining
    remaining = len(audio) - pos_in
    if remaining > 0 and pos_out + remaining <= len(result):
        result[pos_out:pos_out + remaining] = audio[pos_in:pos_in + remaining]

    return result


def add_shimmer(audio: np.ndarray, shimmer_percent: float = 1.0,
                sample_rate: int = 24000) -> np.ndarray:
    """
    Add amplitude micro-variations (shimmer) to make speech sound alive.

    Natural speech has ~1-3% shimmer. We add 1% (subtle).

    Implementation: Modulate amplitude at ~4-8Hz (natural vocal fold instability rate).
    """
    if len(audio) < sample_rate * 0.1:
        return audio

    # Generate slow random modulation (4-8 Hz)
    modulation_freq = np.random.uniform(4, 8)
    t = np.arange(len(audio)) / sample_rate

    # Smooth random modulation
    modulation = 1.0 + (shimmer_percent / 100) * np.sin(2 * np.pi * modulation_freq * t)

    # Add some randomness to the modulation
    noise = np.random.normal(0, shimmer_percent / 200, len(audio))
    # Smooth the noise
    kernel_size = int(sample_rate * 0.01)  # 10ms smoothing
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        noise = np.convolve(noise, kernel, mode='same')

    modulation = modulation + noise

    return audio * modulation.astype(np.float32)


def insert_breaths(audio: np.ndarray, sample_rate: int = 24000,
                   min_pause_ms: int = 300, breath_probability: float = 0.6) -> np.ndarray:
    """
    Insert breath sounds at natural pause points in speech.

    Detects pauses (low energy regions) and inserts subtle breath sounds.
    Only inserts at pauses > min_pause_ms (clause boundaries).

    Args:
        audio: Audio array
        sample_rate: Sample rate
        min_pause_ms: Minimum pause length to consider for breath insertion
        breath_probability: Probability of inserting a breath at each valid pause

    Returns:
        Audio with breath sounds inserted at natural points
    """
    if len(audio) < sample_rate * 0.5:  # Skip very short audio
        return audio

    # Detect pause locations using frame-level energy
    frame_ms = 25  # 25ms frames
    frame_size = int(sample_rate * frame_ms / 1000)
    hop_size = frame_size // 2

    energies = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        energy = np.sqrt(np.mean(frame ** 2))
        energies.append(energy)

    energies = np.array(energies)
    if len(energies) == 0:
        return audio

    # Threshold: frames below 5% of max energy are "silent"
    threshold = max(np.percentile(energies, 10), 0.005)

    # Find pause regions (consecutive silent frames)
    is_silent = energies < threshold

    # Find pause boundaries
    pauses = []
    in_pause = False
    pause_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_pause:
            pause_start = i
            in_pause = True
        elif not silent and in_pause:
            pause_end = i
            pause_duration_ms = (pause_end - pause_start) * (hop_size / sample_rate * 1000)
            if pause_duration_ms >= min_pause_ms:
                pauses.append((pause_start, pause_end, pause_duration_ms))
            in_pause = False

    if not pauses:
        return audio

    # Generate breath sound
    breath = _generate_breath_sound(sample_rate, duration_ms=180)

    # Insert breaths at random subset of pauses
    result = audio.copy()
    breaths_inserted = 0

    for pause_start_frame, pause_end_frame, duration_ms in pauses:
        if np.random.random() > breath_probability:
            continue

        # Calculate sample position (middle of pause)
        pause_center = ((pause_start_frame + pause_end_frame) // 2) * hop_size
        breath_start = pause_center - len(breath) // 2
        breath_end = breath_start + len(breath)

        if breath_start < 0 or breath_end > len(result):
            continue

        # Add breath (mix, don't replace)
        result[breath_start:breath_end] += breath[:breath_end - breath_start]
        breaths_inserted += 1

    if breaths_inserted > 0:
        logger.debug(f"Inserted {breaths_inserted} breath sounds at {len(pauses)} pauses")

    return result


def add_warmth(audio: np.ndarray, sample_rate: int = 24000,
               amount: float = 0.1) -> np.ndarray:
    """
    Add subtle harmonic warmth (2nd/3rd order harmonics).

    This mimics the natural harmonic richness of human vocal cords,
    which is often missing in codec-based TTS.

    Implementation: Soft saturation adds even harmonics.
    """
    if len(audio) < 100:
        return audio

    # Soft saturation (tanh-based, adds even harmonics)
    # Scale up, saturate, scale down
    drive = 1.0 + amount * 3  # Subtle drive
    saturated = np.tanh(audio * drive) / drive

    # Mix original with saturated
    result = audio * (1 - amount) + saturated * amount

    return result.astype(np.float32)


def humanize_audio(audio_tensor: torch.Tensor, sample_rate: int = 24000,
                   jitter: float = 0.3, shimmer: float = 1.0,
                   breaths: bool = True, warmth: float = 0.08) -> torch.Tensor:
    """
    Full humanization pipeline for TTS output.

    Applies all naturalness enhancements in the correct order:
    1. Breath insertion (at pause points)
    2. Jitter (pitch micro-variations)
    3. Shimmer (amplitude micro-variations)
    4. Warmth (subtle harmonic enrichment)

    Args:
        audio_tensor: Input audio as torch tensor
        sample_rate: Audio sample rate
        jitter: Jitter amount (0.3 = 0.3% pitch variation)
        shimmer: Shimmer amount (1.0 = 1% amplitude variation)
        breaths: Whether to insert breath sounds
        warmth: Harmonic warmth amount (0.08 = 8% mix)

    Returns:
        Humanized audio as torch tensor
    """
    device = audio_tensor.device
    audio = audio_tensor.cpu().numpy().astype(np.float32)

    if len(audio) < sample_rate * 0.2:  # Skip very short audio
        return audio_tensor

    # 1. Insert breaths at natural pause points
    if breaths:
        audio = insert_breaths(audio, sample_rate)

    # 2. Add jitter (pitch micro-variations)
    if jitter > 0:
        audio = add_jitter(audio, jitter, sample_rate)

    # 3. Add shimmer (amplitude micro-variations)
    if shimmer > 0:
        audio = add_shimmer(audio, shimmer, sample_rate)

    # 4. Add warmth (subtle harmonics)
    if warmth > 0:
        audio = add_warmth(audio, sample_rate, warmth)

    # Prevent clipping
    peak = np.abs(audio).max()
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    return torch.from_numpy(audio).to(device)
