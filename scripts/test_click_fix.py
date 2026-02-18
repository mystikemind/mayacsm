#!/usr/bin/env python3
"""
Test script to verify clicks are fixed after disabling audio enhancement.

Run with:
    cd /home/ec2-user/SageMaker/project_maya
    source venv/bin/activate
    python scripts/test_click_fix.py
"""
import sys
import os

# Add paths
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print('='*70)
print('  AUDIO CLICK TEST - Enhancement Disabled')
print('='*70)

import torch
import numpy as np

from maya.engine.tts_streaming_real import RealStreamingTTSEngine

# Initialize TTS
print('\nInitializing TTS engine...')
tts = RealStreamingTTSEngine()
tts.initialize()

# Test phrases with varying lengths
test_phrases = [
    'hi there',
    'hello how are you',
    'thats a really interesting question',
    'let me think about that for a moment okay',
    'oh wow i hadnt thought of it that way thats really interesting tell me more'
]

print('\n' + '-'*70)
print('Testing for clicks (severe discontinuities > 0.3)')
print('-'*70)

total_clicks = 0
all_results = []

for phrase in test_phrases:
    chunks = list(tts.generate_stream(phrase, use_context=False))
    if chunks:
        audio = torch.cat(chunks).cpu().numpy()
        diff = np.abs(np.diff(audio))
        severe_clicks = int(np.sum(diff > 0.3))
        moderate_clicks = int(np.sum(diff > 0.2))
        max_disc = float(np.max(diff))

        total_clicks += severe_clicks
        all_results.append({
            'phrase': phrase[:40],
            'severe': severe_clicks,
            'moderate': moderate_clicks,
            'max_disc': max_disc,
            'duration_ms': len(audio) / 24000 * 1000
        })

        status = '✓ CLEAN' if severe_clicks == 0 else f'✗ {severe_clicks} clicks'
        print(f'\n  "{phrase[:40]}..."')
        print(f'    Duration: {len(audio)/24000*1000:.0f}ms, Max discontinuity: {max_disc:.4f}')
        print(f'    Status: {status}')

    torch.cuda.empty_cache()

print('\n' + '='*70)
print('  SUMMARY')
print('='*70)

if total_clicks == 0:
    print('\n  ✓ SUCCESS: All audio samples are CLICK-FREE!')
    print('  The enhancement bypass is working correctly.')
else:
    print(f'\n  ✗ FAILED: Found {total_clicks} severe clicks across samples')
    print('  Further investigation needed.')

print('\n  Detailed Results:')
for r in all_results:
    status = 'CLEAN' if r['severe'] == 0 else f'{r["severe"]} severe'
    print(f'    - {r["phrase"]}: {r["duration_ms"]:.0f}ms, max_disc={r["max_disc"]:.4f}, {status}')

# Save a sample for manual listening
print('\n' + '-'*70)
print('Generating sample audio file for manual verification...')
print('-'*70)

audio_chunks = list(tts.generate_stream('hello this is maya speaking without audio enhancement', use_context=False))
if audio_chunks:
    audio = torch.cat(audio_chunks).cpu().numpy()

    # Save as WAV
    import scipy.io.wavfile as wav
    output_path = '/home/ec2-user/SageMaker/project_maya/test_no_enhancement.wav'
    wav.write(output_path, 24000, (audio * 32767).astype(np.int16))
    print(f'  Saved: {output_path}')
    print(f'  Duration: {len(audio)/24000:.2f}s')

    # Analyze
    diff = np.abs(np.diff(audio))
    print(f'  Max discontinuity: {np.max(diff):.4f}')
    print(f'  Severe clicks (>0.3): {np.sum(diff > 0.3)}')

print('\n' + '='*70)
print('  Test complete!')
print('='*70)
