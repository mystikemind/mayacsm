#!/usr/bin/env python3
"""Quick latency test after optimizations."""

import sys
import time
import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

print("=" * 60)
print("QUICK LATENCY TEST (After Optimizations)")
print("=" * 60)

# Initialize TTS only
from maya.engine.tts_compiled import CompiledTTSEngine

print("\n[1] Initializing TTS...")
tts = CompiledTTSEngine()
tts.initialize()

# Test with short responses
test_phrases = [
    "Hi! Nice to meet you.",     # 5 words
    "I'm great, thanks!",        # 3 words
    "That's really cool!",       # 3 words
    "Hmm, good question!",       # 3 words
]

print("\n[2] Testing TTS with short responses...")
print("-" * 60)

results = []
for phrase in test_phrases:
    start = time.time()
    audio = tts.generate(phrase, use_context=False)
    elapsed = time.time() - start

    duration = len(audio) / 24000
    rtf = elapsed / duration

    results.append({
        'phrase': phrase,
        'words': len(phrase.split()),
        'time_ms': elapsed * 1000,
        'audio_s': duration,
        'rtf': rtf
    })

    print(f"  '{phrase}' ({len(phrase.split())} words)")
    print(f"    -> {elapsed*1000:.0f}ms, {duration:.2f}s audio, RTF={rtf:.2f}x")

# Summary
import numpy as np
avg_time = np.mean([r['time_ms'] for r in results])
avg_rtf = np.mean([r['rtf'] for r in results])
avg_audio = np.mean([r['audio_s'] for r in results])

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Average TTS time: {avg_time:.0f}ms (target: <800ms)")
print(f"  Average audio: {avg_audio:.2f}s")
print(f"  Average RTF: {avg_rtf:.2f}x (must be < 1.0)")
print(f"  Real-time: {'YES ✅' if avg_rtf < 1.0 else 'NO ❌'}")

# Projected full pipeline
est_stt = 150
est_llm = 300
est_total = est_stt + est_llm + avg_time

print(f"\n  Projected full pipeline:")
print(f"    STT: ~{est_stt}ms")
print(f"    LLM: ~{est_llm}ms")
print(f"    TTS: ~{avg_time:.0f}ms")
print(f"    Total: ~{est_total:.0f}ms")
print(f"    vs Sesame (~750ms): {'+' if est_total > 750 else '-'}{abs(est_total - 750):.0f}ms")
