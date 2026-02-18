#!/usr/bin/env python3
"""
Test TTS latency and first chunk timing with the click fix.
"""
import sys
import os
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch

print('='*70)
print('  TTS LATENCY TEST - With Click Fix')
print('='*70)

from maya.engine.tts_streaming_real import RealStreamingTTSEngine

print('\nInitializing TTS engine...')
tts = RealStreamingTTSEngine()
tts.initialize()

test_phrases = [
    ('Short', 'hi there'),
    ('Medium', 'hello how are you doing today'),
    ('Long', 'thats really interesting i hadnt thought about it that way before'),
]

print('\n' + '-'*70)
print('  Measuring first chunk latency (5 runs per phrase)')
print('-'*70)

for name, phrase in test_phrases:
    print(f'\n=== {name}: "{phrase[:30]}..." ===')

    latencies = []
    for run in range(5):
        torch.cuda.synchronize()
        start = time.time()

        first_chunk_time = None
        for i, chunk in enumerate(tts.generate_stream(phrase, use_context=False)):
            if i == 0:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - start) * 1000
                chunk_ms = len(chunk) / 24000 * 1000
                break

        if first_chunk_time:
            latencies.append(first_chunk_time)
            print(f'  Run {run+1}: {first_chunk_time:.0f}ms (first chunk = {chunk_ms:.0f}ms audio)')

        torch.cuda.empty_cache()

    if latencies:
        avg = sum(latencies) / len(latencies)
        best = min(latencies)
        print(f'\n  Average: {avg:.0f}ms, Best: {best:.0f}ms')

print('\n' + '='*70)
print('  SUMMARY')
print('='*70)
print('\n  Target: < 150ms for first chunk (Sesame level)')
print('  If consistently under 150ms, latency is acceptable.')
print('='*70)
