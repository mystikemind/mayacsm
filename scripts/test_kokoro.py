#!/usr/bin/env python3
"""
Test Kokoro TTS - Ultra-fast 82M parameter TTS.

Should be MUCH faster than CSM while maintaining quality.
"""

import torch
import time
import soundfile as sf
from kokoro import KPipeline

print("=" * 60)
print("TESTING KOKORO TTS - ULTRA FAST")
print("=" * 60)

# Test phrases
test_phrases = [
    "Hello! I'm Maya.",
    "That's a really interesting question. Let me think about that.",
    "I understand how you feel. It can be really difficult sometimes, but I'm here to help you through it.",
]

# Initialize
print("\nInitializing Kokoro pipeline...")
init_start = time.time()
pipeline = KPipeline(lang_code='a')  # 'a' for American English
init_time = time.time() - init_start
print(f"Initialized in {init_time*1000:.0f}ms")

# Test each phrase
for i, phrase in enumerate(test_phrases):
    print(f"\n{'='*60}")
    print(f"Phrase {i+1}: '{phrase[:50]}...'")
    print("=" * 60)

    start = time.time()
    first_chunk_time = None
    chunks = []

    # Generate with streaming
    generator = pipeline(phrase, voice='af_heart')

    for j, (gs, ps, audio) in enumerate(generator):
        chunk_time = time.time() - start
        if first_chunk_time is None:
            first_chunk_time = chunk_time
            print(f"\n>>> FIRST CHUNK in {first_chunk_time*1000:.0f}ms! <<<")

        chunks.append(audio)
        chunk_duration = len(audio) / 24000
        print(f"  Chunk {j}: {len(audio)} samples ({chunk_duration:.2f}s) at {chunk_time*1000:.0f}ms")

    total_time = time.time() - start

    # Combine chunks (audio can be tensor or numpy)
    if chunks:
        if isinstance(chunks[0], torch.Tensor):
            full_audio = torch.cat(chunks).numpy()
        else:
            full_audio = torch.cat([torch.from_numpy(c) for c in chunks]).numpy()
    else:
        full_audio = torch.zeros(24000).numpy()
    audio_duration = len(full_audio) / 24000

    print(f"\nResults:")
    print(f"  First chunk: {first_chunk_time*1000:.0f}ms")
    print(f"  Total time: {total_time*1000:.0f}ms")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Chunks: {len(chunks)}")
    print(f"  RTF: {total_time/audio_duration:.4f}x")
    print(f"  Speed: {audio_duration/total_time:.1f}x real-time")

    # Save
    output_path = f"/home/ec2-user/SageMaker/project_maya/tests/outputs/kokoro_test_{i+1}.wav"
    sf.write(output_path, full_audio, 24000)
    print(f"  Saved: {output_path}")

# Benchmark: Generate long text to measure sustained speed
print(f"\n{'='*60}")
print("BENCHMARK: Long text generation")
print("=" * 60)

long_text = """
Well, that's a really fascinating question, and I'm happy to help you think through it.
There are actually several different ways we could approach this problem.
First, we could try to break it down into smaller, more manageable pieces.
Then, we can work through each piece one at a time.
What do you think would be the best place to start?
"""

start = time.time()
first_chunk_time = None
total_audio_samples = 0
chunk_count = 0

generator = pipeline(long_text.strip(), voice='af_heart')

for j, (gs, ps, audio) in enumerate(generator):
    if first_chunk_time is None:
        first_chunk_time = time.time() - start
    total_audio_samples += len(audio)
    chunk_count += 1

total_time = time.time() - start
audio_duration = total_audio_samples / 24000

print(f"\nLong text benchmark:")
print(f"  Text length: {len(long_text)} chars")
print(f"  First chunk: {first_chunk_time*1000:.0f}ms")
print(f"  Total time: {total_time*1000:.0f}ms")
print(f"  Audio duration: {audio_duration:.2f}s")
print(f"  Chunks: {chunk_count}")
print(f"  RTF: {total_time/audio_duration:.4f}x")
print(f"  Speed: {audio_duration/total_time:.1f}x real-time")

print(f"\n{'='*60}")
print("KOKORO TTS TEST COMPLETE")
print("=" * 60)
