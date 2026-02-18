#!/usr/bin/env python3
"""Generate audio samples for quality evaluation."""

import sys
import time
import torch
import torchaudio

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

print("=" * 60)
print("GENERATING QUALITY SAMPLES")
print("=" * 60)

from maya.engine.tts_compiled import CompiledTTSEngine

tts = CompiledTTSEngine()
tts.initialize()

output_dir = "/home/ec2-user/SageMaker/project_maya/tests/outputs"

# Test phrases covering different types of responses
samples = [
    ("greeting", "Hi! Nice to meet you."),
    ("question", "What brings you here today?"),
    ("statement", "That sounds really interesting!"),
    ("acknowledgment", "I understand, thanks!"),
    ("thinking", "Hmm, let me think about that."),
    ("excitement", "Oh wow, that's amazing!"),
]

print("\nGenerating samples...")
print("-" * 60)

for name, text in samples:
    start = time.time()
    audio = tts.generate(text, use_context=False)
    elapsed = time.time() - start

    duration = len(audio) / 24000

    # Save
    path = f"{output_dir}/quality_{name}.wav"
    torchaudio.save(path, audio.unsqueeze(0).cpu(), 24000)

    print(f"  {name}: '{text}'")
    print(f"    -> {elapsed*1000:.0f}ms, {duration:.2f}s audio")
    print(f"    -> Saved: {path}")

print("\n" + "=" * 60)
print("QUALITY EVALUATION CHECKLIST")
print("=" * 60)
print("""
Listen to each sample and evaluate:

1. CLARITY - Is speech clear and understandable?
2. NATURALNESS - Does it sound human-like?
3. CONSISTENCY - Same voice across all samples?
4. PACING - Natural speech rhythm?
5. ARTIFACTS - Any clicks, pops, or distortions?

Compare with Sesame Maya demo:
- Voice warmth and expressiveness
- Natural pauses and rhythm
- Emotional appropriateness

Files saved to: tests/outputs/quality_*.wav
""")
