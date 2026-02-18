#!/usr/bin/env python3
"""
Generate LONG thinking fillers (5-6 seconds) for zero perceived latency.
"""

import torch
import torchaudio
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

from maya.engine import TTSEngine
from maya.config import FILLERS_DIR


# Long thinking fillers - need to be 5-6 seconds minimum
LONG_FILLERS = [
    ("thinking_1", "Hmm... that's a really interesting question. Let me think about that for a moment... There's actually a few different ways to look at this..."),
    ("thinking_2", "Well, you know what... I've been thinking about something like that recently. Let me see if I can put my thoughts together here..."),
    ("thinking_3", "Oh, that's something I find really fascinating actually. Give me just a second to think about the best way to explain this..."),
    ("thinking_4", "Hmm... okay, so there's a lot to unpack there. Let me make sure I give you a really good answer on this one..."),
    ("thinking_5", "You know, that's a great point you're making. I want to think through this carefully so I can give you a thoughtful response..."),
    ("thinking_6", "That's such an interesting perspective. Let me take a moment to consider this from a few different angles before I respond..."),
]


def main():
    """Generate long thinking fillers."""
    print("=" * 60)
    print("GENERATING LONG THINKING FILLERS")
    print("=" * 60)

    # Initialize TTS
    print("\nLoading TTS engine...")
    tts = TTSEngine()
    tts.initialize()

    FILLERS_DIR.mkdir(parents=True, exist_ok=True)

    for name, text in LONG_FILLERS:
        filepath = FILLERS_DIR / f"{name}.wav"

        print(f"\n[GENERATING] {name}")
        print(f"  Text: '{text[:60]}...'")

        start = time.time()

        # Generate with longer max duration
        audio = tts._generator.generate(
            text=text,
            speaker=0,
            context=[],
            max_audio_length_ms=8000,  # 8 seconds max
        )

        # Save
        torchaudio.save(str(filepath), audio.unsqueeze(0).cpu(), 24000)

        duration = len(audio) / 24000
        elapsed = time.time() - start

        print(f"  Duration: {duration:.1f}s (generated in {elapsed:.1f}s)")

        # Target: 5-6 seconds
        if duration < 4.5:
            print(f"  ⚠️  WARNING: Only {duration:.1f}s, might need longer text")
        else:
            print(f"  ✅ Good length!")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
