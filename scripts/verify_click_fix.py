#!/usr/bin/env python3
"""
Verify click fix is consistent across multiple runs.
"""
import sys
import os

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

print('='*70)
print('  CLICK FIX VERIFICATION - Multiple Runs')
print('='*70)

from maya.engine.tts_streaming_real import RealStreamingTTSEngine

print('\nInitializing TTS engine...')
tts = RealStreamingTTSEngine()
tts.initialize()

phrases = [
    'hi there',
    'hello how are you',
    'thats a great question',
    'let me think about that',
    'oh thats really interesting'
]

NUM_RUNS = 20

print(f'\nRunning {NUM_RUNS} iterations per phrase...')
print('-'*70)

all_clean = True
total_runs = 0
clean_runs = 0

for phrase in phrases:
    phrase_results = []
    for run in range(NUM_RUNS):
        chunks = list(tts.generate_stream(phrase, use_context=False))
        if chunks:
            audio = torch.cat(chunks).cpu().numpy()
            diff = np.abs(np.diff(audio))
            severe = int(np.sum(diff > 0.3))
            max_disc = float(np.max(diff))
            phrase_results.append({'severe': severe, 'max_disc': max_disc})
            total_runs += 1
            if severe == 0:
                clean_runs += 1
        torch.cuda.empty_cache()

    # Report
    severe_values = [r['severe'] for r in phrase_results]
    max_disc_values = [r['max_disc'] for r in phrase_results]
    all_clean_phrase = all(s == 0 for s in severe_values)

    status = '✓ ALL CLEAN' if all_clean_phrase else f'✗ {sum(1 for s in severe_values if s > 0)} runs with clicks'
    print(f'\n  "{phrase}"')
    print(f'    Severe clicks per run: {severe_values}')
    print(f'    Max disc range: {min(max_disc_values):.4f} - {max(max_disc_values):.4f}')
    print(f'    Status: {status}')

    if not all_clean_phrase:
        all_clean = False

print('\n' + '='*70)
print('  FINAL RESULTS')
print('='*70)

clean_pct = (clean_runs / total_runs) * 100 if total_runs > 0 else 0

print(f'\n  Total runs: {total_runs}')
print(f'  Clean runs: {clean_runs} ({clean_pct:.1f}%)')

if clean_pct >= 95:
    print(f'\n  ✓✓✓ FIX VERIFIED: {clean_pct:.1f}% clean runs')
elif clean_pct >= 80:
    print(f'\n  ✓ MOSTLY FIXED: {clean_pct:.1f}% clean runs (acceptable)')
else:
    print(f'\n  ✗ NEEDS MORE WORK: Only {clean_pct:.1f}% clean runs')

print('='*70)
