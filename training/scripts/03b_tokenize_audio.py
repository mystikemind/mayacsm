#!/usr/bin/env python3
"""
Step 3b: Pre-tokenize Audio with Mimi

This script pre-tokenizes all audio files using the Mimi neural audio codec.
This allows training without loading Mimi, saving ~2GB GPU memory.

Usage:
    python 03b_tokenize_audio.py --input data/csm_ready_ex04
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

# Setup CSM path
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def tokenize_dataset(data_dir: Path, split: str = "train"):
    """Tokenize all audio files in a dataset split."""
    from moshi.models import loaders
    from huggingface_hub import hf_hub_download

    metadata_path = data_dir / f"{split}.json"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return

    with open(metadata_path) as f:
        samples = json.load(f)

    logger.info(f"Tokenizing {len(samples)} {split} samples...")

    # Load Mimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading Mimi on {device}...")

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
    audio_tokenizer.set_num_codebooks(32)

    # Create tokens directory
    tokens_dir = data_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)

    updated_samples = []

    for i, sample in enumerate(tqdm(samples)):
        audio_path = data_dir / sample["path"]

        try:
            # Load audio
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            audio = audio.astype(np.float32)

            # Convert to tensor
            audio = torch.tensor(audio, dtype=torch.float32, device=device)

            # Tokenize
            with torch.no_grad():
                tokens = audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]  # (32, num_frames)

            # Save tokens
            token_filename = f"{i:05d}_tokens.pt"
            token_path = tokens_dir / token_filename
            torch.save(tokens.cpu(), token_path)

            # Update sample metadata
            sample["tokens_path"] = f"tokens/{token_filename}"
            sample["num_frames"] = tokens.size(1)
            updated_samples.append(sample)

        except Exception as e:
            logger.warning(f"Failed to tokenize {audio_path}: {e}")
            continue

    # Save updated metadata
    output_path = data_dir / f"{split}_tokenized.json"
    with open(output_path, "w") as f:
        json.dump(updated_samples, f, indent=2)

    logger.info(f"Saved {len(updated_samples)} tokenized samples to {output_path}")

    # Cleanup
    del audio_tokenizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize audio with Mimi")
    parser.add_argument("--input", type=str, required=True, help="Data directory")

    args = parser.parse_args()
    data_dir = Path(args.input)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PRE-TOKENIZING AUDIO WITH MIMI")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")

    # Tokenize train and val splits
    tokenize_dataset(data_dir, "train")
    tokenize_dataset(data_dir, "val")

    logger.info("")
    logger.info("=" * 60)
    logger.info("TOKENIZATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
