#!/usr/bin/env python3
"""
Test TTS with the new Expresso voice prompt.
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
print('  TESTING WITH EXPRESSO VOICE PROMPT')
print('='*70)

from maya.engine.tts_streaming_real import RealStreamingTTSEngine

print('\nInitializing TTS engine...')
tts = RealStreamingTTSEngine()
tts.initialize()

# Check which voice prompt was loaded
if tts._voice_prompt:
    vp = tts._voice_prompt
    print(f'\nVoice prompt loaded:')
    print(f'  Duration: {len(vp.audio)/24000:.1f}s')
    print(f'  Text: {vp.text[:100]}...')
else:
    print('\nWARNING: No voice prompt loaded!')

# Test phrases - both plain and with style tags
test_cases = [
    # Plain text (should use default style)
    ('plain_greeting', 'hi there how are you doing today'),
    ('plain_question', 'what do you think about that'),
    ('plain_excited', 'oh wow thats amazing'),
    ('plain_thinking', 'hmm let me think about that'),

    # With style tags (model was trained on these)
    ('happy_greeting', '[happy] hi there how are you doing today'),
    ('sad_response', '[sad] oh im sorry to hear that'),
    ('confused_question', '[confused] wait what do you mean by that'),
    ('whisper_secret', '[whisper] can you keep a secret'),
]

output_dir = '/home/ec2-user/SageMaker/project_maya/audio_expresso_test'
os.makedirs(output_dir, exist_ok=True)

print(f'\nSaving samples to: {output_dir}/')
print('-'*70)

for name, text in test_cases:
    print(f'\nGenerating: "{text}"')

    chunks = list(tts.generate_stream(text, use_context=False))
    if chunks:
        audio = torch.cat(chunks)
        audio_np = audio.cpu().numpy()

        # Analyze
        diff = np.abs(np.diff(audio_np))
        clicks = int(np.sum(diff > 0.3))

        # Save
        path = f'{output_dir}/{name}.wav'
        wav.write(path, 24000, (audio_np * 32767).astype(np.int16))

        print(f'  Saved: {name}.wav ({len(audio_np)/24000:.1f}s, {clicks} clicks)')

    torch.cuda.empty_cache()

print('\n' + '='*70)
print('  SAMPLES GENERATED')
print('='*70)
print(f'\n  Listen to files in: {output_dir}/')
print('\n  Compare:')
print('  - plain_* vs happy_* (should sound different)')
print('  - Check if style tags change the voice emotion')
print('='*70)
