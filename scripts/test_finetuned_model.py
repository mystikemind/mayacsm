#!/usr/bin/env python3
"""
Test the fine-tuned CSM model for voice quality.
Compares fine-tuned model output with base model and training data.
"""

import sys
import os
import time
from pathlib import Path

# Add paths for all dependencies
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya/training')
os.chdir('/home/ec2-user/SageMaker/project_maya')

# Install peft if not available
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft", "-q"])
    from peft import LoraConfig, get_peft_model

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.io.wavfile as wav

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training/checkpoints/csm_maya_correct/best_model"
OUTPUT_DIR = PROJECT_ROOT / "audio_finetuned_test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Test phrases - same as used before
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


def load_finetuned_model():
    """Load the base CSM model with fine-tuned weights."""
    print("Loading base CSM-1B model...")

    from models import Model

    # Load base model
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device="cuda", dtype=torch.bfloat16)

    print("Applying LoRA weights...")

    # Configure LoRA (same config as training)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "output_proj", "w1", "w2", "w3"],
        lora_dropout=0.0,
        bias="none",
    )

    # Apply LoRA to the backbone
    model.backbone = get_peft_model(model.backbone, lora_config)

    # Load LoRA weights
    lora_path = CHECKPOINT_DIR / "lora_weights.pt"
    lora_state = torch.load(lora_path, weights_only=False)
    model.backbone.load_state_dict(lora_state, strict=False)
    print(f"  Loaded LoRA weights from {lora_path}")

    # Load decoder weights
    decoder_path = CHECKPOINT_DIR / "decoder.pt"
    decoder_state = torch.load(decoder_path, weights_only=False)
    model.decoder.load_state_dict(decoder_state)
    print(f"  Loaded decoder from {decoder_path}")

    # Load codebook0_head weights
    head_path = CHECKPOINT_DIR / "codebook0_head.pt"
    head_state = torch.load(head_path, weights_only=False)
    model.codebook0_head.load_state_dict(head_state)
    print(f"  Loaded codebook0_head from {head_path}")

    # Merge LoRA weights for inference
    model.backbone = model.backbone.merge_and_unload()

    model.eval()
    return model


def test_finetuned_model():
    """Test the fine-tuned model."""
    print("\n" + "=" * 70)
    print("TESTING FINE-TUNED CSM MODEL")
    print("=" * 70)

    # Load model
    model = load_finetuned_model()

    # Create generator
    from generator import Generator
    generator = Generator(model)

    # Load Mimi for voice prompt tokenization
    from moshi.models import loaders
    from huggingface_hub import hf_hub_download

    print("\nLoading Mimi codec...")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cuda")
    mimi.set_num_codebooks(32)
    mimi.eval()

    # Load voice prompt (use a short reference from training data)
    print("\nLoading voice reference...")

    # Use a training sample as context
    train_data_dir = PROJECT_ROOT / "training/data/csm_ready_ex04/train"
    sample_files = sorted(list(train_data_dir.glob("*.pt")))[:1]  # First sample

    if sample_files:
        ref_data = torch.load(sample_files[0], weights_only=False)
        ref_audio = ref_data["audio"]
        ref_text = ref_data.get("text", "")
        print(f"  Reference: '{ref_text[:50]}...' ({len(ref_audio)/24000:.1f}s)")

        # Tokenize reference
        ref_audio_tensor = ref_audio.float().unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            ref_tokens = mimi.encode(ref_audio_tensor).squeeze(0)

        from generator import Segment
        context = [Segment(
            text=ref_text,
            speaker=0,
            audio=ref_audio.cpu(),
            audio_tokens=ref_tokens.cpu(),
        )]
    else:
        print("  No reference found, generating without context")
        context = []

    # Generate test samples
    results = []
    output_subdir = OUTPUT_DIR / "finetuned"
    output_subdir.mkdir(exist_ok=True)

    print("\n" + "-" * 50)
    print("Generating test samples...")
    print("-" * 50)

    for phrase_name, phrase_text in TEST_PHRASES:
        print(f"\n  '{phrase_text}'")

        try:
            start = time.time()

            audio = generator.generate(
                text=phrase_text,
                speaker=0,
                context=context,
                max_audio_length_ms=8000,
                temperature=0.8,
                topk=50,
            )

            gen_time = time.time() - start
            audio_np = audio.cpu().float().numpy()

            # Normalize
            if audio_np.max() > 0:
                audio_np = audio_np / max(abs(audio_np.min()), abs(audio_np.max()))

            # Calculate metrics
            metrics = calculate_metrics(audio_np)
            metrics["generation_time"] = gen_time
            metrics["phrase"] = phrase_name
            metrics["text"] = phrase_text

            # Save audio
            output_path = output_subdir / f"{phrase_name}.wav"
            wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))

            print(f"    Duration: {metrics['duration']:.1f}s | Clicks: {metrics['clicks']} | Time: {gen_time:.1f}s")
            results.append(metrics)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"phrase": phrase_name, "error": str(e)})

        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    valid = [r for r in results if "error" not in r]
    if valid:
        avg_duration = np.mean([r["duration"] for r in valid])
        avg_clicks = np.mean([r["clicks"] for r in valid])
        avg_harsh = np.mean([r["harsh_clicks"] for r in valid])
        avg_time = np.mean([r["generation_time"] for r in valid])

        print(f"\nSuccessful generations: {len(valid)}/{len(results)}")
        print(f"Average duration: {avg_duration:.1f}s")
        print(f"Average clicks: {avg_clicks:.0f} (harsh: {avg_harsh:.0f})")
        print(f"Average generation time: {avg_time:.1f}s")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  - {r['phrase']}: {r.get('error', 'Unknown')}")

    print(f"\nAudio samples saved to: {output_subdir}/")
    print("\n" + "=" * 70)
    print("LISTEN TO THE SAMPLES TO EVALUATE QUALITY!")
    print("Compare with training data in: training/data/csm_ready_ex04/")
    print("=" * 70)

    # Cleanup
    del model, generator, mimi
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    test_finetuned_model()
