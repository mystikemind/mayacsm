#!/usr/bin/env python3
"""
Step 3: Preprocess Audio for CSM Training

This script converts the extracted speaker data to CSM-compatible format:
- Resample from 48kHz to 24kHz (CSM's native rate)
- Normalize audio levels for consistent training
- Add emotion/style tags to text where available
- Create train/val splits
- Generate forced alignment timestamps (optional)

Key preprocessing steps:
1. Resample 48kHz → 24kHz using high-quality resampling
2. Peak normalize to -6dB with soft limiting
3. Remove DC offset
4. Validate no clipping or artifacts

Usage:
    python 03_preprocess_audio.py --input data/expresso_talia

Output:
    data/csm_ready_talia/
    ├── audio/        (24kHz normalized WAV files)
    ├── train.json    (90% of data)
    ├── val.json      (10% of data)
    └── config.json   (preprocessing config used)
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
DATA_DIR = PROJECT_ROOT / "training" / "data"

# Audio processing parameters
TARGET_SAMPLE_RATE = 24000    # CSM native rate
TARGET_PEAK_DB = -6.0         # Target peak level in dB
TARGET_PEAK_LINEAR = 0.5      # 10^(-6/20) ≈ 0.5
SOFT_CLIP_THRESHOLD = 0.95    # Soft clip above this

# Train/val split
VAL_RATIO = 0.1  # 10% validation
RANDOM_SEED = 42

# Style to emotion tag mapping
STYLE_TO_TAG = {
    "default": "",
    "happy": "[happy]",
    "sad": "[sad]",
    "confused": "[confused]",
    "laughing": "[laughing]",
    "whisper": "[whisper]",
    "enunciated": "[clear]",
    "angry": "[angry]",
    "excited": "[excited]",
    "neutral": "",
}


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality resampling using scipy."""
    if orig_sr == target_sr:
        return audio

    try:
        from scipy import signal
        # Use polyphase resampling for quality
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        resampled = signal.resample_poly(audio, up, down)
        return resampled
    except ImportError:
        # Fallback to torchaudio
        import torchaudio
        import torch
        audio_tensor = torch.tensor(audio).unsqueeze(0).float()
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(audio_tensor).squeeze(0).numpy()
        return resampled


def normalize_audio(audio: np.ndarray, target_peak: float = TARGET_PEAK_LINEAR) -> np.ndarray:
    """Normalize audio with peak normalization and soft limiting."""
    # Remove DC offset
    audio = audio - np.mean(audio)

    # Find peak
    peak = np.abs(audio).max()

    if peak < 1e-6:
        return audio  # Silence

    # Normalize to target peak
    audio = audio * (target_peak / peak)

    # Soft clip to prevent harsh clipping
    # Using tanh-based soft limiter
    audio = np.tanh(audio * 1.5) / 1.5

    return audio


def validate_audio(audio: np.ndarray, sr: int) -> Tuple[bool, str]:
    """Validate processed audio quality."""
    # Check for NaN or Inf
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        return False, "contains_nan_or_inf"

    # Check for clipping
    if np.abs(audio).max() > 1.0:
        return False, "clipping_detected"

    # Check duration
    duration = len(audio) / sr
    if duration < 0.5:
        return False, "too_short_after_processing"

    # Check for silence (entire audio near zero)
    if np.abs(audio).max() < 0.01:
        return False, "nearly_silent"

    return True, "valid"


def process_sample(args: Tuple) -> Optional[Dict]:
    """Process a single audio sample (for parallel processing)."""
    sample, input_dir, output_audio_dir, idx = args

    try:
        # Load audio
        input_path = input_dir / sample["path"]

        if not input_path.exists():
            return None

        try:
            import soundfile as sf
            audio, sr = sf.read(str(input_path))
        except:
            import torchaudio
            audio_tensor, sr = torchaudio.load(str(input_path))
            audio = audio_tensor.mean(dim=0).numpy()

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 24kHz
        if sr != TARGET_SAMPLE_RATE:
            audio = resample_audio(audio, sr, TARGET_SAMPLE_RATE)

        # Normalize
        audio = normalize_audio(audio)

        # Validate
        is_valid, reason = validate_audio(audio, TARGET_SAMPLE_RATE)
        if not is_valid:
            return {"error": reason, "original_id": sample.get("original_id", "")}

        # Save processed audio
        output_filename = f"{idx:05d}.wav"
        output_path = output_audio_dir / output_filename

        try:
            import soundfile as sf
            sf.write(str(output_path), audio, TARGET_SAMPLE_RATE)
        except:
            import torchaudio
            import torch
            torchaudio.save(str(output_path), torch.tensor(audio).unsqueeze(0), TARGET_SAMPLE_RATE)

        # Process text - add style tags if available
        text = sample.get("text", "").strip()
        style = sample.get("style", "default")

        # Add emotion tag if style is not default
        emotion_tag = STYLE_TO_TAG.get(style, "")
        if emotion_tag:
            text = f"{emotion_tag} {text}"

        # Clean text for TTS
        text = text.lower()
        # Keep prosody-affecting punctuation
        import re
        text = re.sub(r"[^\w\s.,?!'\-\[\]]", " ", text)
        text = re.sub(r'\s+', ' ', text).strip()

        return {
            "path": f"audio/{output_filename}",
            "text": text,
            "speaker": sample.get("speaker", ""),
            "style": style,
            "duration": len(audio) / TARGET_SAMPLE_RATE,
            "original_id": sample.get("original_id", ""),
        }

    except Exception as e:
        return {"error": str(e), "original_id": sample.get("original_id", "")}


def preprocess_dataset(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    num_workers: int = 4
) -> Dict:
    """Preprocess entire dataset for CSM training."""

    logger.info("=" * 60)
    logger.info("PREPROCESSING FOR CSM TRAINING")
    logger.info("=" * 60)

    # Load metadata
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return {"error": "metadata_not_found"}

    with open(metadata_path) as f:
        metadata = json.load(f)

    speaker = metadata.get("speaker", "unknown")
    samples = metadata.get("samples", [])

    logger.info(f"Input: {input_dir}")
    logger.info(f"Speaker: {speaker}")
    logger.info(f"Samples: {len(samples)}")

    # Setup output directory
    if output_dir is None:
        output_dir = DATA_DIR / f"csm_ready_{speaker}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_audio_dir = output_dir / "audio"
    output_audio_dir.mkdir(exist_ok=True)

    logger.info(f"Output: {output_dir}")
    logger.info(f"Target sample rate: {TARGET_SAMPLE_RATE}Hz")
    logger.info(f"Target peak: {TARGET_PEAK_DB}dB")
    logger.info("")

    # Process samples in parallel
    logger.info(f"Processing with {num_workers} workers...")

    processed_samples = []
    errors = {}

    # Prepare args for parallel processing
    process_args = [
        (sample, input_dir, output_audio_dir, i)
        for i, sample in enumerate(samples)
    ]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, args): args[3] for args in process_args}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result is None:
                    errors["file_not_found"] = errors.get("file_not_found", 0) + 1
                elif "error" in result:
                    error_key = result["error"]
                    errors[error_key] = errors.get(error_key, 0) + 1
                else:
                    processed_samples.append(result)

                if (len(processed_samples) + sum(errors.values())) % 100 == 0:
                    logger.info(f"  Processed: {len(processed_samples) + sum(errors.values())}/{len(samples)}")

            except Exception as e:
                errors[str(e)] = errors.get(str(e), 0) + 1

    logger.info(f"Processing complete: {len(processed_samples)} valid samples")

    # Create train/val split
    random.seed(RANDOM_SEED)
    random.shuffle(processed_samples)

    val_size = int(len(processed_samples) * VAL_RATIO)
    train_samples = processed_samples[val_size:]
    val_samples = processed_samples[:val_size]

    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")

    # Calculate total duration
    train_duration = sum(s["duration"] for s in train_samples)
    val_duration = sum(s["duration"] for s in val_samples)

    logger.info(f"Train duration: {train_duration/3600:.2f} hours")
    logger.info(f"Val duration: {val_duration/3600:.2f} hours")

    # Save train.json
    train_path = output_dir / "train.json"
    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)

    # Save val.json
    val_path = output_dir / "val.json"
    with open(val_path, "w") as f:
        json.dump(val_samples, f, indent=2)

    # Save config
    config = {
        "speaker": speaker,
        "source_dir": str(input_dir),
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_peak_db": TARGET_PEAK_DB,
        "val_ratio": VAL_RATIO,
        "random_seed": RANDOM_SEED,
        "total_samples": len(processed_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_duration_hours": train_duration / 3600,
        "val_duration_hours": val_duration / 3600,
        "processing_errors": errors,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Train samples: {len(train_samples)} ({train_duration/3600:.2f}h)")
    logger.info(f"Val samples: {len(val_samples)} ({val_duration/3600:.2f}h)")
    logger.info(f"Total duration: {(train_duration + val_duration)/3600:.2f} hours")
    logger.info("")
    if errors:
        logger.info("Processing errors:")
        for error, count in sorted(errors.items(), key=lambda x: -x[1]):
            logger.info(f"  {error}: {count}")
    logger.info("=" * 60)

    return config


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for CSM training")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory (e.g., data/expresso_talia)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/csm_ready_{speaker})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, multiprocessing.cpu_count()),
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else None
    result = preprocess_dataset(input_dir, output_dir, args.workers)

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
