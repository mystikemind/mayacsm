#!/usr/bin/env python3
"""
Generate SHORT, NATURAL fillers for seamless response.

These are the natural human speech patterns Sesame uses:
- "Yeahhh..." (agreement)
- "Hmm..." (thinking)
- "Ohhh..." (empathy/realization)
- "Okay..." (acknowledgment)
- "Right..." (understanding)
- "Mhmm..." (neutral confirmation)

Keep them SHORT (0.3-1.0 seconds) so they feel natural, not forced.
"""

import torch
import torchaudio
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

from maya.engine import TTSEngine
from maya.config import FILLERS_DIR


# Natural short fillers - VERY short text for quick sounds
NATURAL_FILLERS = [
    # Agreement/Acknowledgment
    ("yeah_1", "Yeah."),
    ("yeah_2", "Yeahhh..."),
    ("yeah_3", "Yeah, yeah."),

    # Thinking
    ("hmm_1", "Hmm."),
    ("hmm_2", "Hmmm..."),
    ("hmm_3", "Hmm, hmm."),

    # Empathy/Realization
    ("oh_1", "Oh."),
    ("oh_2", "Ohhh..."),
    ("oh_3", "Oh, okay."),

    # Acknowledgment
    ("okay_1", "Okay."),
    ("okay_2", "Okay..."),

    # Understanding
    ("right_1", "Right."),
    ("right_2", "Right, right."),

    # Neutral
    ("mhmm_1", "Mhmm."),
    ("mhmm_2", "Mm-hmm."),

    # Interest
    ("uh_1", "Uh..."),
    ("ah_1", "Ah."),
    ("ah_2", "Ahhh..."),
]


def main():
    """Generate all natural fillers."""
    print("=" * 60)
    print("GENERATING NATURAL FILLERS")
    print("=" * 60)

    # Initialize TTS
    print("\nLoading TTS engine...")
    tts = TTSEngine()
    tts.initialize()

    FILLERS_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for name, text in NATURAL_FILLERS:
        filepath = FILLERS_DIR / f"{name}.wav"

        # Skip if already exists
        if filepath.exists():
            audio, sr = torchaudio.load(str(filepath))
            duration = audio.shape[1] / sr
            print(f"[EXISTS] {name}: {duration:.2f}s - '{text}'")
            skipped += 1
            continue

        print(f"\n[GENERATING] {name}: '{text}'")
        start = time.time()

        try:
            # Generate with short max duration
            audio = tts._generator.generate(
                text=text,
                speaker=0,
                context=[],
                max_audio_length_ms=2000,  # Max 2 seconds
            )

            # Trim silence from end
            audio = trim_silence(audio)

            # Save
            torchaudio.save(str(filepath), audio.unsqueeze(0).cpu(), 24000)

            duration = len(audio) / 24000
            elapsed = time.time() - start

            print(f"  Duration: {duration:.2f}s (generated in {elapsed:.1f}s)")
            generated += 1

            # Check if too long
            if duration > 1.5:
                print(f"  ⚠️  WARNING: {duration:.2f}s is too long for natural filler")
            elif duration < 0.2:
                print(f"  ⚠️  WARNING: {duration:.2f}s might be too short")
            else:
                print(f"  ✅ Good length!")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"COMPLETE! Generated: {generated}, Skipped: {skipped}")
    print(f"Total: {generated + skipped} fillers in {FILLERS_DIR}")
    print("=" * 60)


def trim_silence(audio: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Trim silence from end of audio."""
    # Find last non-silent sample
    abs_audio = audio.abs()
    for i in range(len(audio) - 1, -1, -1):
        if abs_audio[i] > threshold:
            # Keep a tiny bit of trailing audio
            end_idx = min(i + 2400, len(audio))  # +100ms
            return audio[:end_idx]
    return audio


if __name__ == "__main__":
    main()
