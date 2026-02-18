#!/usr/bin/env python3
"""Generate audio samples for listening test."""
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
print('  GENERATING AUDIO SAMPLES FOR LISTENING')
print('='*70)

from maya.engine.tts_streaming_real import RealStreamingTTSEngine

print('\nInitializing TTS engine...')
tts = RealStreamingTTSEngine()
tts.initialize()

test_phrases = [
    ('greeting', 'hi there how are you doing today'),
    ('question', 'what do you think about that'),
    ('statement', 'thats really interesting i hadnt thought of it that way'),
    ('excited', 'oh wow thats amazing i love it'),
    ('thinking', 'hmm let me think about that for a moment'),
]

output_dir = '/home/ec2-user/SageMaker/project_maya/audio_samples'
os.makedirs(output_dir, exist_ok=True)

print(f'\nSaving samples to: {output_dir}/')
print('-'*70)

for name, phrase in test_phrases:
    print(f'\nGenerating: "{phrase}"')

    # Generate multiple takes
    best_audio = None
    best_clicks = float('inf')

    for take in range(3):
        chunks = list(tts.generate_stream(phrase, use_context=False))
        if chunks:
            audio = torch.cat(chunks)
            diff = np.abs(np.diff(audio.cpu().numpy()))
            clicks = int(np.sum(diff > 0.3))
            max_disc = float(np.max(diff))

            print(f'  Take {take+1}: {clicks} clicks, max_disc={max_disc:.3f}')

            if clicks < best_clicks:
                best_clicks = clicks
                best_audio = audio

        torch.cuda.empty_cache()

    if best_audio is not None:
        # Save the best take
        audio_np = best_audio.cpu().numpy()
        path = f'{output_dir}/{name}.wav'
        wav.write(path, 24000, (audio_np * 32767).astype(np.int16))
        print(f'  Saved: {name}.wav (best: {best_clicks} clicks)')

        # Also analyze frequency content
        from scipy import fft
        spectrum = np.abs(fft.rfft(audio_np))
        freqs = fft.rfftfreq(len(audio_np), 1/24000)

        # Energy in bands
        low = np.sum(spectrum[(freqs >= 0) & (freqs < 500)]**2)
        mid = np.sum(spectrum[(freqs >= 500) & (freqs < 2000)]**2)
        high = np.sum(spectrum[(freqs >= 2000) & (freqs < 6000)]**2)
        air = np.sum(spectrum[(freqs >= 6000) & (freqs < 12000)]**2)
        total = low + mid + high + air

        if total > 0:
            print(f'  Spectrum: low={100*low/total:.1f}% mid={100*mid/total:.1f}% high={100*high/total:.1f}% air={100*air/total:.1f}%')

print('\n' + '='*70)
print('  SAMPLES GENERATED')
print('='*70)
print(f'\n  Files saved to: {output_dir}/')
print('  Listen to these files to judge audio quality.')
print('  Look for: clicks, muffled sound, naturalness')
print('='*70)
