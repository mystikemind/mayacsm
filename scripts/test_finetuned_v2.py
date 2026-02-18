#!/usr/bin/env python3
"""
Test fine-tuned CSM with proper voice prompt from training data.
"""

import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

# Install peft if needed
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft", "-q"])
    from peft import LoraConfig, get_peft_model

import torch
import numpy as np
import scipy.io.wavfile as wav
import torchaudio

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training/checkpoints/csm_maya_correct/best_model"
TRAINING_DATA = PROJECT_ROOT / "training/data/csm_ready_ex04"
OUTPUT_DIR = PROJECT_ROOT / "audio_finetuned_test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Test phrases
TEST_PHRASES = [
    ("greeting", "hi there how are you doing today"),
    ("happy", "oh wow thats amazing i love it"),
    ("sad", "oh no im so sorry to hear that"),
    ("thinking", "hmm let me think about that"),
    ("question", "wait what do you mean by that"),
    ("agree", "yeah that makes a lot of sense"),
    ("natural", "you know i was just thinking the same thing"),
]


def calculate_metrics(audio_np):
    """Calculate audio quality metrics."""
    return {
        "duration": len(audio_np) / 24000,
        "peak": float(np.abs(audio_np).max()),
        "rms": float(np.sqrt(np.mean(audio_np ** 2))),
        "clicks": int(np.sum(np.abs(np.diff(audio_np)) > 0.3)),
        "harsh_clicks": int(np.sum(np.abs(np.diff(audio_np)) > 0.5)),
    }


def load_voice_prompt(mimi, sample_idx=0):
    """Load a sample from training data as voice prompt."""
    train_json = TRAINING_DATA / "train.json"
    with open(train_json) as f:
        train_samples = json.load(f)

    # Get a few neutral/default style samples for context
    neutral_samples = [s for s in train_samples if s.get("style") == "default"][:5]
    if not neutral_samples:
        neutral_samples = train_samples[:5]

    contexts = []
    total_duration = 0

    from generator import Segment

    for sample in neutral_samples:
        if total_duration >= 10:  # Use ~10s of context
            break

        audio_path = TRAINING_DATA / sample["path"]
        if not audio_path.exists():
            continue

        # Load audio
        audio, sr = torchaudio.load(str(audio_path))
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, 24000)
        audio = audio.mean(dim=0) if audio.dim() > 1 and audio.size(0) > 1 else audio.squeeze(0)

        # Clean text (remove style tags)
        text = sample["text"]
        if text.startswith("["):
            text = text.split("]", 1)[-1].strip()

        # Segment only needs: speaker, text, audio
        contexts.append(Segment(
            speaker=0,
            text=text,
            audio=audio.cpu(),
        ))

        total_duration += sample.get("duration", len(audio) / 24000)
        print(f"    Context: '{text[:40]}...' ({len(audio)/24000:.1f}s)")

    return contexts


def load_finetuned_model():
    """Load the base CSM model with fine-tuned weights."""
    print("Loading base CSM-1B model...")

    from models import Model

    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device="cuda", dtype=torch.bfloat16)

    print("Applying fine-tuned weights...")

    # Configure LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "output_proj", "w1", "w2", "w3"],
        lora_dropout=0.0,
        bias="none",
    )

    model.backbone = get_peft_model(model.backbone, lora_config)

    # Load weights
    lora_state = torch.load(CHECKPOINT_DIR / "lora_weights.pt", weights_only=False)
    model.backbone.load_state_dict(lora_state, strict=False)

    decoder_state = torch.load(CHECKPOINT_DIR / "decoder.pt", weights_only=False)
    model.decoder.load_state_dict(decoder_state)

    head_state = torch.load(CHECKPOINT_DIR / "codebook0_head.pt", weights_only=False)
    model.codebook0_head.load_state_dict(head_state)

    # Merge LoRA for inference
    model.backbone = model.backbone.merge_and_unload()
    model.eval()

    print("  Fine-tuned weights loaded successfully")
    return model


def test_model(model_name, model, generator, context, output_subdir):
    """Test model with given context."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    output_subdir.mkdir(exist_ok=True)
    results = []

    for phrase_name, phrase_text in TEST_PHRASES:
        print(f"\n  '{phrase_text}'")

        try:
            start = time.time()

            audio = generator.generate(
                text=phrase_text,
                speaker=0,
                context=context,
                max_audio_length_ms=6000,  # Reduce max length
                temperature=0.8,
                topk=50,
            )

            gen_time = time.time() - start
            audio_np = audio.cpu().float().numpy()

            # Normalize
            if audio_np.max() > 0:
                peak = max(abs(audio_np.min()), abs(audio_np.max()))
                audio_np = audio_np / peak * 0.9

            metrics = calculate_metrics(audio_np)
            metrics["generation_time"] = gen_time
            metrics["phrase"] = phrase_name

            # Save
            output_path = output_subdir / f"{phrase_name}.wav"
            wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))

            print(f"    {metrics['duration']:.1f}s | {metrics['clicks']} clicks | {gen_time:.1f}s")
            results.append(metrics)

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"phrase": phrase_name, "error": str(e)})

        torch.cuda.empty_cache()

    return results


def main():
    print("\n" + "=" * 70)
    print("FINE-TUNED MODEL QUALITY TEST (v2)")
    print("=" * 70)

    # Load Mimi
    print("\nLoading Mimi codec...")
    from moshi.models import loaders
    from huggingface_hub import hf_hub_download
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cuda")
    mimi.set_num_codebooks(32)
    mimi.eval()

    # Load voice context
    print("\nLoading voice prompt from training data...")
    context = load_voice_prompt(mimi)
    print(f"  Total context segments: {len(context)}")

    # Test fine-tuned model
    print("\n" + "-" * 60)
    print("Loading fine-tuned model...")
    print("-" * 60)

    finetuned_model = load_finetuned_model()
    from generator import Generator
    finetuned_gen = Generator(finetuned_model)

    finetuned_results = test_model(
        "Fine-tuned CSM",
        finetuned_model,
        finetuned_gen,
        context,
        OUTPUT_DIR / "finetuned_with_context"
    )

    # Cleanup fine-tuned
    del finetuned_model, finetuned_gen
    torch.cuda.empty_cache()

    # Compare with base model
    print("\n" + "-" * 60)
    print("Loading BASE model for comparison...")
    print("-" * 60)

    from models import Model
    base_model = Model.from_pretrained("sesame/csm-1b")
    base_model.to(device="cuda", dtype=torch.bfloat16)
    base_gen = Generator(base_model)

    base_results = test_model(
        "Base CSM (no fine-tuning)",
        base_model,
        base_gen,
        context,
        OUTPUT_DIR / "base_with_context"
    )

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for name, results in [("Fine-tuned", finetuned_results), ("Base", base_results)]:
        valid = [r for r in results if "error" not in r]
        if valid:
            avg_dur = np.mean([r["duration"] for r in valid])
            avg_clicks = np.mean([r["clicks"] for r in valid])
            print(f"\n{name}:")
            print(f"  Avg duration: {avg_dur:.1f}s")
            print(f"  Avg clicks: {avg_clicks:.0f}")
            print(f"  Success: {len(valid)}/{len(results)}")

    print(f"\n\nAudio samples saved to: {OUTPUT_DIR}/")
    print("\nDirectories to compare:")
    print(f"  - finetuned_with_context/  (fine-tuned model)")
    print(f"  - base_with_context/       (base model)")
    print(f"  - Training data: {TRAINING_DATA}/audio/")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
