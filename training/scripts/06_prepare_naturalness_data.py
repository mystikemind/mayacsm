#!/usr/bin/env python3
"""
Prepare Downloaded Datasets for CSM Decoder-Only Training
==========================================================

Processes DisfluencySpeech, NonverbalTTS, and Expresso into
CSM-compatible tokenized format for decoder-only training.

Uses direct WAV byte decoding to avoid torchcodec dependency.

Usage:
    python 06_prepare_naturalness_data.py [--dataset all|disfluency|nonverbal|expresso]
"""

import os
import sys
import json
import logging
import argparse
import time
import io
from pathlib import Path

import torch
import torchaudio
import numpy as np
import soundfile as sf

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_CACHE"] = "/home/ec2-user/SageMaker/.cache/huggingface/datasets"
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/.cache/huggingface"

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_BASE = Path("/home/ec2-user/SageMaker/project_maya/training/data/datasets")
TARGET_SR = 24000


def load_mimi(device='cuda'):
    """Load Mimi codec for audio tokenization."""
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(32)
    return mimi


def decode_audio_bytes(audio_bytes: bytes) -> tuple:
    """Decode raw WAV/audio bytes to numpy array and sample rate."""
    audio_io = io.BytesIO(audio_bytes)
    audio_array, sr = sf.read(audio_io, dtype='float32')
    return audio_array, sr


@torch.no_grad()
def tokenize_audio(mimi, audio_np: np.ndarray, sr: int, device='cuda') -> torch.Tensor:
    """Tokenize audio to 32 codebooks using Mimi codec."""
    audio = torch.tensor(audio_np, dtype=torch.float32)

    # Ensure mono
    if audio.dim() > 1:
        audio = audio.mean(dim=-1) if audio.size(-1) <= 2 else audio.mean(dim=0)
    audio = audio.squeeze()

    # Resample to 24kHz if needed
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)

    # Normalize
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak * 0.95

    # Skip very short audio
    if len(audio) < TARGET_SR * 0.3:
        return None

    # Tokenize
    audio = audio.to(device).unsqueeze(0).unsqueeze(0)
    tokens = mimi.encode(audio)
    return tokens[0]  # [32, num_frames]


def process_dataset(ds_name, hf_id, hf_config, text_key, mimi, device):
    """Generic dataset processor."""
    from datasets import load_dataset

    output_dir = OUTPUT_BASE / f"{ds_name}_prepared"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)

    logger.info(f"Loading {ds_name} from {hf_id}...")

    # Load without audio decoding (raw arrow data)
    if hf_config:
        ds = load_dataset(hf_id, hf_config,
                         cache_dir=os.environ["HF_DATASETS_CACHE"])
    else:
        ds = load_dataset(hf_id, cache_dir=os.environ["HF_DATASETS_CACHE"])

    samples = []
    skipped = 0
    errors = 0

    for split_name in ds.keys():
        split = ds[split_name]
        logger.info(f"Processing {split_name}: {len(split)} samples")

        # Access raw arrow table to get audio bytes directly
        table = split.data

        for i in range(len(split)):
            try:
                # Get audio bytes from arrow table directly
                audio_struct = table.column('audio')[i].as_py()
                audio_bytes = audio_struct['bytes']

                if audio_bytes is None or len(audio_bytes) < 100:
                    skipped += 1
                    continue

                # Decode audio bytes
                audio_np, sr = decode_audio_bytes(audio_bytes)
                duration = len(audio_np) / sr

                if duration < 0.5 or duration > 30:
                    skipped += 1
                    continue

                # Tokenize
                tokens = tokenize_audio(mimi, audio_np, sr, device)
                if tokens is None or tokens.size(1) < 5:
                    skipped += 1
                    continue

                num_frames = tokens.size(1)

                # Get text
                if isinstance(text_key, list):
                    # Try multiple text keys
                    text = ""
                    for tk in text_key:
                        col = table.column(tk)
                        val = col[i].as_py()
                        if val:
                            text = val
                            break
                else:
                    text = table.column(text_key)[i].as_py() or ""

                # Get style/speaker info if available
                style = "default"
                speaker = "unknown"
                for col_name in ['style', 'emotion', 'Emotion']:
                    if col_name in table.column_names:
                        val = table.column(col_name)[i].as_py()
                        if val:
                            style = val.lower().strip()

                for col_name in ['speaker_id', 'speaker']:
                    if col_name in table.column_names:
                        val = table.column(col_name)[i].as_py()
                        if val:
                            speaker = str(val)

                # Tag text with style
                if style != "default":
                    tagged_text = f"[{style}] {text}"
                else:
                    tagged_text = text

                # Save tokens
                idx = len(samples)
                token_path = f"tokens/{idx:05d}_tokens.pt"
                torch.save(tokens.cpu(), tokens_dir / f"{idx:05d}_tokens.pt")

                samples.append({
                    "text": tagged_text,
                    "tokens_path": token_path,
                    "num_frames": num_frames,
                    "duration": duration,
                    "style": style,
                    "speaker": speaker,
                    "source": ds_name,
                    "split": split_name,
                })

                if (i + 1) % 200 == 0:
                    logger.info(f"  {split_name}: {i+1}/{len(split)} ({len(samples)} saved, {skipped} skipped)")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"Error at {split_name}[{i}]: {e}")
                elif errors == 6:
                    logger.warning("Suppressing further error messages...")
                continue

    # Save metadata
    with open(output_dir / "train_tokenized.json", "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"{ds_name}: {len(samples)} saved, {skipped} skipped, {errors} errors")

    # Style distribution
    style_counts = {}
    for s in samples:
        style_counts[s["style"]] = style_counts.get(s["style"], 0) + 1
    if style_counts:
        logger.info(f"Styles: {json.dumps(style_counts)}")

    return len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "disfluency", "nonverbal", "expresso"],
                        default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Mimi codec...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    mimi = load_mimi(device)
    logger.info(f"Mimi loaded on {device}")

    total = 0
    start = time.time()

    if args.dataset in ["all", "disfluency"]:
        total += process_dataset(
            ds_name="disfluency_speech",
            hf_id="amaai-lab/DisfluencySpeech",
            hf_config=None,
            text_key=["transcript_annotated", "transcript_a"],
            mimi=mimi, device=device,
        )

    if args.dataset in ["all", "nonverbal"]:
        total += process_dataset(
            ds_name="nonverbal_tts",
            hf_id="deepvk/NonverbalTTS",
            hf_config=None,
            text_key=["Result", "Initial text"],
            mimi=mimi, device=device,
        )

    if args.dataset in ["all", "expresso"]:
        total += process_dataset(
            ds_name="expresso_full",
            hf_id="ylacombe/expresso",
            hf_config="read",
            text_key="text",
            mimi=mimi, device=device,
        )

    elapsed = time.time() - start
    logger.info(f"\nTotal: {total} samples prepared in {elapsed/60:.1f} minutes")
    logger.info(f"Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
