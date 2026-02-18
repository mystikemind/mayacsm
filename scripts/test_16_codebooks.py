#!/usr/bin/env python3
"""
Test 16-codebook mode for real-time audio.
Run this to verify RTF < 1.0 before starting the server.
"""
import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import time

print("=" * 60)
print("TESTING 16-CODEBOOK MODE")
print("Expected: RTF < 1.0 (real-time capable)")
print("=" * 60)

from maya.engine.tts_streaming import StreamingTTSEngine, StreamingConfig

tts = StreamingTTSEngine()
tts.initialize()

print(f"\nCodebooks configured: {tts._num_codebooks}")

config = StreamingConfig(
    initial_batch_size=4,
    batch_size=12,
    max_audio_length_ms=5000,
    temperature=0.8,
    topk=50
)

test_texts = [
    "Hello there",
    "How are you doing today",
    "I think that is a really interesting question",
]

results = []
for text in test_texts:
    print(f"\nTest: '{text}'")
    
    chunks = []
    start = time.time()
    first_chunk_time = None
    
    for chunk in tts._generate_frames_sync(
        text=tts.preprocess_text(text),
        speaker=0,
        context=[],
        config=config
    ):
        if first_chunk_time is None:
            first_chunk_time = (time.time() - start) * 1000
        chunks.append(chunk)
    
    total_time = time.time() - start
    if chunks:
        total_samples = sum(len(c) for c in chunks)
        audio_duration = total_samples / 24000
        rtf = total_time / audio_duration
        results.append(rtf)
        
        status = "REAL-TIME!" if rtf < 1.0 else ("CLOSE" if rtf < 1.5 else "TOO SLOW")
        print(f"  First chunk: {first_chunk_time:.0f}ms")
        print(f"  Audio: {audio_duration:.2f}s")
        print(f"  Gen time: {total_time:.2f}s")
        print(f"  RTF: {rtf:.2f}x - {status}")

print("\n" + "=" * 60)
if results:
    avg_rtf = sum(results) / len(results)
    print(f"AVERAGE RTF: {avg_rtf:.2f}x")
    if avg_rtf < 1.0:
        print("SUCCESS: Real-time streaming is possible!")
    elif avg_rtf < 1.5:
        print("CLOSE: May work with buffering, but expect some gaps")
    else:
        print("FAIL: Need to try fewer codebooks (8) or different approach")
print("=" * 60)
