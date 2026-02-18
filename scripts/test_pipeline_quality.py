#!/usr/bin/env python3
"""
Compare pipeline TTS output with test script output.

This generates audio using BOTH approaches and saves them for comparison:
1. Official Generator (test script approach)
2. Our pipeline's generate_stream (production approach)

If they sound the same, the pipeline is working correctly.
If they sound different, there's still something to fix.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.io.wavfile as wav

OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/audio_comparison_test")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test phrases - same as production use cases
TEST_PHRASES = [
    ("greeting", "hey there how are you doing today"),
    ("answer", "i am maya nice to meet you"),  # Expanded form (what pipeline sends)
    ("answer_short", "im maya nice to meet you"),  # Short form (what test uses)
    ("happy", "oh wow that is amazing"),
    ("thinking", "hmm let me think about that"),
    ("question", "what do you mean by that"),
]


def test_official_generator():
    """Generate using official CSM Generator (test script approach)."""
    print("\n" + "=" * 60)
    print("Testing OFFICIAL GENERATOR (test script approach)")
    print("=" * 60)

    from models import Model
    from generator import Generator
    from moshi.models import loaders
    from huggingface_hub import hf_hub_download
    import torchaudio
    import json

    # Load model
    print("Loading model...")
    model = Model.from_pretrained("sesame/csm-1b")

    # Load fine-tuned weights
    FINETUNED = '/home/ec2-user/SageMaker/project_maya/training/checkpoints/csm_maya_correct/best_model/model_merged.pt'
    if os.path.exists(FINETUNED):
        print("Loading fine-tuned weights...")
        state_dict = torch.load(FINETUNED, map_location="cuda", weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)

    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    # Create generator
    generator = Generator(model)

    # Load context from training data (same as test script)
    from generator import Segment
    TRAINING_DATA = '/home/ec2-user/SageMaker/project_maya/training/data/csm_ready_ex04'
    train_json = os.path.join(TRAINING_DATA, 'train.json')

    context = []
    with open(train_json) as f:
        train_samples = json.load(f)

    default_samples = [s for s in train_samples if s.get("style") == "default"][:5]
    total_duration = 0
    for sample in default_samples:
        if total_duration >= 10:
            break
        audio_path = os.path.join(TRAINING_DATA, sample["path"])
        if not os.path.exists(audio_path):
            continue
        audio, sr = torchaudio.load(audio_path)
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, 24000)
        audio = audio.mean(dim=0) if audio.dim() > 1 else audio.squeeze(0)
        text = sample["text"]
        if text.startswith("["):
            text = text.split("]", 1)[-1].strip()
        context.append(Segment(speaker=0, text=text, audio=audio.cpu()))
        total_duration += sample.get("duration", len(audio) / 24000)

    print(f"  Context: {len(context)} segments, ~{total_duration:.1f}s")

    # Generate samples
    output_subdir = OUTPUT_DIR / "official_generator"
    output_subdir.mkdir(exist_ok=True)

    for name, phrase in TEST_PHRASES:
        print(f"\n  Generating: '{phrase}'")
        start = time.time()

        audio = generator.generate(
            text=phrase.lower(),
            speaker=0,
            context=context,
            max_audio_length_ms=5000,
            temperature=0.8,
            topk=50,
        )

        elapsed = (time.time() - start) * 1000
        audio_np = audio.cpu().float().numpy()

        # Simple peak normalization (like test script)
        peak = max(abs(audio_np.min()), abs(audio_np.max()))
        if peak > 0:
            audio_np = audio_np / peak * 0.9

        output_path = output_subdir / f"{name}.wav"
        wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))
        print(f"    {len(audio_np)/24000:.1f}s audio in {elapsed:.0f}ms -> {output_path}")

    # Cleanup
    del model, generator
    torch.cuda.empty_cache()


def test_pipeline_tts():
    """Generate using our pipeline's TTS."""
    print("\n" + "=" * 60)
    print("Testing PIPELINE TTS (production approach)")
    print("=" * 60)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Initialize TTS
    print("Initializing pipeline TTS...")
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Generate samples
    output_subdir = OUTPUT_DIR / "pipeline_tts"
    output_subdir.mkdir(exist_ok=True)

    for name, phrase in TEST_PHRASES:
        print(f"\n  Generating: '{phrase}'")
        start = time.time()

        # Collect all chunks
        chunks = []
        for chunk in tts.generate_stream(phrase.lower()):
            chunks.append(chunk.cpu())

        if chunks:
            audio = torch.cat(chunks)
            elapsed = (time.time() - start) * 1000
            audio_np = audio.float().numpy()

            output_path = output_subdir / f"{name}.wav"
            wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))
            print(f"    {len(audio_np)/24000:.1f}s audio in {elapsed:.0f}ms -> {output_path}")
        else:
            print(f"    No audio generated!")

    torch.cuda.empty_cache()


def main():
    print("\n" + "=" * 70)
    print("PIPELINE VS OFFICIAL GENERATOR COMPARISON")
    print("=" * 70)
    print(f"\nOutputs will be saved to: {OUTPUT_DIR}")
    print("\nCompare these directories to hear the difference:")
    print("  - official_generator/  (reference quality)")
    print("  - pipeline_tts/        (our production output)")
    print("\nIf they sound the same, the pipeline is working correctly!")

    # Run both tests
    test_official_generator()
    test_pipeline_tts()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nListen to files in: {OUTPUT_DIR}")
    print("  - official_generator/*.wav (reference)")
    print("  - pipeline_tts/*.wav (production)")


if __name__ == "__main__":
    main()
