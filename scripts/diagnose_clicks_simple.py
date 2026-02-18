#!/usr/bin/env python3
"""
Simple click diagnosis - just analyze the output of generate_stream.
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
import scipy.io.wavfile as wav

print('='*70)
print('  SIMPLE CLICK DIAGNOSIS')
print('='*70)

from maya.engine.tts_streaming_real import RealStreamingTTSEngine

# Initialize
print('\nInitializing TTS engine...')
tts = RealStreamingTTSEngine()
tts.initialize()

# Test multiple phrases
test_phrases = [
    ('short', 'hi there'),
    ('medium', 'hello how are you doing today'),
    ('long', 'oh thats really interesting i hadnt thought about it that way before'),
]

print('\n' + '-'*70)
print('  Analyzing chunks from generate_stream()')
print('-'*70)

for name, phrase in test_phrases:
    print(f'\n=== {name.upper()}: "{phrase[:40]}..." ===')

    # Collect chunks
    chunks = []
    for chunk in tts.generate_stream(phrase, use_context=False):
        chunks.append(chunk.clone())

    print(f'  Generated {len(chunks)} chunks')

    # Analyze individual chunks
    print('\n  Individual chunks:')
    for i, chunk in enumerate(chunks):
        audio = chunk.cpu().numpy()
        diff = np.abs(np.diff(audio))
        severe = int(np.sum(diff > 0.3))
        max_disc = float(np.max(diff))
        status = 'CLEAN' if severe == 0 else f'{severe} clicks'
        print(f'    Chunk {i}: {len(audio)/24000*1000:.0f}ms, max_disc={max_disc:.4f}, {status}')

    # Analyze boundaries between chunks
    print('\n  Chunk boundaries:')
    for i in range(len(chunks) - 1):
        c1 = chunks[i].cpu().numpy()
        c2 = chunks[i + 1].cpu().numpy()
        boundary_disc = abs(c1[-1] - c2[0])
        status = '*** SEVERE ***' if boundary_disc > 0.3 else ('MODERATE' if boundary_disc > 0.2 else 'ok')
        print(f'    {i} -> {i+1}: disc={boundary_disc:.4f} {status}')

    # Full concatenated analysis
    if chunks:
        full_audio = torch.cat(chunks).cpu().numpy()
        diff = np.abs(np.diff(full_audio))
        severe = int(np.sum(diff > 0.3))
        moderate = int(np.sum(diff > 0.2))
        max_disc = float(np.max(diff))
        print(f'\n  Full audio: {len(full_audio)/24000*1000:.0f}ms')
        print(f'    Max discontinuity: {max_disc:.4f}')
        print(f'    Severe clicks (>0.3): {severe}')
        print(f'    Moderate clicks (>0.2): {moderate}')

        # Save audio
        path = f'/home/ec2-user/SageMaker/project_maya/diag_{name}.wav'
        wav.write(path, 24000, (full_audio * 32767).astype(np.int16))
        print(f'    Saved: {path}')

    torch.cuda.empty_cache()

print('\n' + '='*70)
print('  ANALYSIS')
print('='*70)

# Look at the last saved audio files
print('\n  If clicks are at boundaries -> crossfade issue')
print('  If clicks are within chunks -> codec/normalization issue')
print('  If short phrases have more clicks -> warmup/stabilization issue')
print('\n  Check the diag_*.wav files to listen for clicks.')
print('='*70)
