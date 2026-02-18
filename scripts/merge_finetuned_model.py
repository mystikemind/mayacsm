#!/usr/bin/env python3
"""
Merge fine-tuned CSM weights into a single model file for production use.
"""

import sys
import os
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

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training/checkpoints/csm_maya_correct/best_model"
OUTPUT_PATH = PROJECT_ROOT / "training/checkpoints/csm_maya_correct/best_model/model_merged.pt"

def main():
    print("=" * 60)
    print("MERGING FINE-TUNED CSM MODEL")
    print("=" * 60)

    # Load base model
    print("\nLoading base CSM-1B model...")
    from models import Model
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device="cuda", dtype=torch.bfloat16)

    print("Applying LoRA weights...")

    # Configure LoRA (same as training)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "output_proj", "w1", "w2", "w3"],
        lora_dropout=0.0,
        bias="none",
    )

    # Apply LoRA to backbone
    model.backbone = get_peft_model(model.backbone, lora_config)

    # Load LoRA weights
    lora_path = CHECKPOINT_DIR / "lora_weights.pt"
    lora_state = torch.load(lora_path, weights_only=False)
    model.backbone.load_state_dict(lora_state, strict=False)
    print(f"  Loaded LoRA weights from {lora_path}")

    # Merge LoRA into base weights (makes inference faster)
    print("Merging LoRA weights into backbone...")
    model.backbone = model.backbone.merge_and_unload()

    # Load fine-tuned decoder
    print("Loading fine-tuned decoder...")
    decoder_path = CHECKPOINT_DIR / "decoder.pt"
    decoder_state = torch.load(decoder_path, weights_only=False)
    model.decoder.load_state_dict(decoder_state)
    print(f"  Loaded decoder from {decoder_path}")

    # Load fine-tuned codebook0_head
    print("Loading fine-tuned codebook0_head...")
    head_path = CHECKPOINT_DIR / "codebook0_head.pt"
    head_state = torch.load(head_path, weights_only=False)
    model.codebook0_head.load_state_dict(head_state)
    print(f"  Loaded codebook0_head from {head_path}")

    # Save merged model
    print(f"\nSaving merged model to {OUTPUT_PATH}...")
    torch.save(model.state_dict(), OUTPUT_PATH)

    # Check file size
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved! Size: {size_mb:.1f} MB")

    # Verify by loading
    print("\nVerifying merged model...")
    model2 = Model.from_pretrained("sesame/csm-1b")
    model2.load_state_dict(torch.load(OUTPUT_PATH, weights_only=False), strict=True)
    print("  Verification successful!")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print(f"Merged model: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
