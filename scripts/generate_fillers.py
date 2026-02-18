#!/usr/bin/env python3
"""
Generate high-quality filler audio using CSM.

This creates natural-sounding thinking fillers that play
while Maya processes the user's speech.
"""

import torch
import torchaudio
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

from maya.engine import TTSEngine
from maya.config import FILLERS_DIR


# Filler definitions: (filename, text)
FILLERS = [
    # Thinking fillers (5-6 seconds)
    ("thinking_1", "Hmm, that's a really interesting question... let me think about that for a moment..."),
    ("thinking_2", "Well, you know... there are a few things I want to say about that..."),
    ("thinking_3", "Let me see... okay, so here's what I'm thinking..."),
    ("thinking_4", "So... that's actually something I find really fascinating... let me share my thoughts..."),
    ("thinking_5", "Okay, so... I want to make sure I give you a good answer here..."),
    ("thinking_6", "Hmm... you know, that reminds me of something... let me think..."),

    # Backchannels (short)
    ("backchannel_1", "Mm-hmm."),
    ("backchannel_2", "Yeah."),
    ("backchannel_3", "Right."),
    ("backchannel_4", "I see."),
    ("backchannel_5", "Uh-huh."),

    # Transitions
    ("transition_1", "Okay, so..."),
    ("transition_2", "Alright..."),
    ("transition_3", "So anyway..."),

    # Empathy
    ("empathy_1", "Oh... I understand..."),
    ("empathy_2", "I see... that sounds really..."),
]


def main():
    """Generate all filler audio files."""
    print("=" * 60)
    print("GENERATING MAYA FILLERS")
    print("=" * 60)

    # Initialize TTS
    print("\nLoading TTS engine...")
    tts = TTSEngine()
    tts.initialize()

    # Create output directory
    FILLERS_DIR.mkdir(parents=True, exist_ok=True)

    total_duration = 0

    for name, text in FILLERS:
        filepath = FILLERS_DIR / f"{name}.wav"

        # Skip if already exists
        if filepath.exists():
            audio, sr = torchaudio.load(str(filepath))
            duration = audio.shape[1] / sr
            total_duration += duration
            print(f"[EXISTS] {name}: {duration:.1f}s")
            continue

        print(f"[GENERATING] {name}: '{text[:40]}...'")
        start = time.time()

        # Generate
        audio = tts.generate_short(text)

        # Save
        torchaudio.save(str(filepath), audio.unsqueeze(0).cpu(), 24000)

        duration = len(audio) / 24000
        elapsed = time.time() - start
        total_duration += duration

        print(f"           → {duration:.1f}s audio in {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print(f"COMPLETE: {len(FILLERS)} fillers, {total_duration:.1f}s total audio")
    print(f"Saved to: {FILLERS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
