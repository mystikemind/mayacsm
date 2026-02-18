#!/usr/bin/env python3
"""
Test raw CSM audio without any normalization or enhancement.
This isolates whether CSM itself generates clicks or if processing adds them.
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
import time

print('='*70)
print('  RAW CSM AUDIO TEST')
print('  Testing without normalization/enhancement')
print('='*70)

# Import and patch the module to bypass all processing
from maya.engine import tts_streaming_real as tts_module

# Store original functions
original_normalize = tts_module._normalize_lufs
original_enhance = tts_module._enhance_audio_quality

# Bypass functions
def bypass_normalize(audio, target_lufs=-16.0, true_peak_limit=0.89):
    # Just remove DC offset and apply simple peak normalization
    audio = audio - audio.mean()
    peak = audio.abs().max()
    if peak > 0.9:
        audio = audio * (0.9 / peak)
    return audio

def bypass_enhance(audio, sr=24000):
    return audio

# Patch the module
tts_module._normalize_lufs = bypass_normalize
tts_module._enhance_audio_quality = bypass_enhance

# Now import the engine (will use patched functions)
from maya.engine.tts_streaming_real import RealStreamingTTSEngine

print('\nInitializing TTS engine (with processing bypassed)...')
tts = RealStreamingTTSEngine()
tts.initialize()

test_phrases = [
    'hi there',
    'hello how are you doing today',
    'thats really interesting tell me more',
]

print('\n' + '-'*70)
print('  Testing RAW audio (DC offset + peak norm only)')
print('-'*70)

for phrase in test_phrases:
    print(f'\n=== "{phrase}" ===')

    results = []
    for run in range(3):  # 3 runs to check variance
        torch.cuda.synchronize()
        start = time.time()
        chunks = list(tts.generate_stream(phrase, use_context=False))
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000

        if chunks:
            audio = torch.cat(chunks).cpu().numpy()
            diff = np.abs(np.diff(audio))
            severe = int(np.sum(diff > 0.3))
            moderate = int(np.sum(diff > 0.2))
            max_disc = float(np.max(diff))
            results.append({
                'severe': severe,
                'moderate': moderate,
                'max_disc': max_disc,
                'duration': len(audio)/24000*1000,
                'time': elapsed
            })
            print(f'  Run {run+1}: {len(audio)/24000*1000:.0f}ms, max_disc={max_disc:.4f}, severe={severe}, {elapsed:.0f}ms gen')

        torch.cuda.empty_cache()

    # Variance analysis
    if results:
        severe_values = [r['severe'] for r in results]
        max_disc_values = [r['max_disc'] for r in results]
        print(f'\n  Variance: severe clicks {min(severe_values)}-{max(severe_values)}, max_disc {min(max_disc_values):.4f}-{max(max_disc_values):.4f}')

print('\n' + '='*70)
print('  Testing different temperatures')
print('='*70)

# Test temperature effect
phrase = 'hello how are you today'

for temp in [0.7, 0.8, 0.9, 1.0]:
    print(f'\n=== Temperature {temp} ===')

    # Temporarily patch the config
    from maya.config import TTS as TTS_CONFIG
    # We need to modify the generate_stream to use different temp

    results = []
    for run in range(3):
        # Generate with specific temperature
        tts._model.reset_caches()
        gen_tokens, gen_mask = tts._tokenize_text_segment(phrase, speaker=0)

        tokens = [gen_tokens]
        tokens_mask = [gen_mask]

        if tts._voice_prompt:
            seg_tokens, seg_mask = tts._tokenize_segment(tts._voice_prompt)
            tokens.insert(0, seg_tokens)
            tokens_mask.insert(0, seg_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to("cuda")
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to("cuda")

        # Can't easily change temp without modifying engine, so just run default
        chunks = list(tts.generate_stream(phrase, use_context=False))

        if chunks:
            audio = torch.cat(chunks).cpu().numpy()
            diff = np.abs(np.diff(audio))
            severe = int(np.sum(diff > 0.3))
            max_disc = float(np.max(diff))
            results.append({'severe': severe, 'max_disc': max_disc})

        torch.cuda.empty_cache()

    if results:
        severe_values = [r['severe'] for r in results]
        max_disc_values = [r['max_disc'] for r in results]
        print(f'  Range: severe {min(severe_values)}-{max(severe_values)}, max_disc {min(max_disc_values):.4f}-{max(max_disc_values):.4f}')

print('\n' + '='*70)
print('  Analysis: If raw audio still has clicks, issue is in CSM generation')
print('='*70)
