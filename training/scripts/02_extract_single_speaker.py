#!/usr/bin/env python3
"""
Step 2: Extract Single Speaker from Expresso Dataset

This script extracts all audio segments for a single speaker (default: Talia/ex04)
and applies quality filters to ensure only the best data goes into training.

Uses direct audio file access to avoid torchcodec dependencies.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

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
EXPRESSO_DIR = DATA_DIR / "expresso"

# Speaker mapping
SPEAKER_NAMES = {
    "ex01": "jerry",
    "ex02": "elisabeth",
    "ex03": "thomas",
    "ex04": "talia",
}

# Quality thresholds
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0
MIN_WORDS = 3
MAX_SILENCE_RATIO = 0.5
MIN_SNR_DB = 15


def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode audio from bytes using soundfile."""
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    return audio, sr


def calculate_snr(audio: np.ndarray, frame_length: int = 2048) -> float:
    """Estimate Signal-to-Noise Ratio in dB."""
    frame_energies = []
    for i in range(0, len(audio) - frame_length, frame_length):
        frame = audio[i:i+frame_length]
        frame_energies.append(np.mean(frame ** 2))

    if not frame_energies:
        return 0.0

    frame_energies = np.array(frame_energies)
    noise_floor = np.percentile(frame_energies, 10)
    signal_power = np.mean(frame_energies)

    if noise_floor <= 0:
        return 100.0

    snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
    return snr


def calculate_silence_ratio(audio: np.ndarray, threshold: float = 0.01) -> float:
    """Calculate ratio of silent frames in audio."""
    abs_audio = np.abs(audio)
    silent_samples = np.sum(abs_audio < threshold)
    return silent_samples / len(audio)


def passes_quality_filter(audio: np.ndarray, sr: int, text: str) -> Tuple[bool, str]:
    """Check if audio sample passes all quality filters."""
    duration = len(audio) / sr

    if duration < MIN_DURATION_SEC:
        return False, "too_short"
    if duration > MAX_DURATION_SEC:
        return False, "too_long"

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return False, "text_too_short"

    silence_ratio = calculate_silence_ratio(audio)
    if silence_ratio > MAX_SILENCE_RATIO:
        return False, "too_much_silence"

    snr = calculate_snr(audio)
    if snr < MIN_SNR_DB:
        return False, "low_snr"

    return True, "passed"


def extract_speaker(
    speaker_id: str,
    output_dir: Optional[Path] = None,
    max_samples: Optional[int] = None
) -> Dict:
    """Extract and filter samples for a single speaker using direct parquet access."""
    import soundfile as sf

    speaker_name = SPEAKER_NAMES.get(speaker_id, speaker_id)

    logger.info("=" * 60)
    logger.info(f"EXTRACTING SPEAKER: {speaker_id} ({speaker_name})")
    logger.info("=" * 60)

    if output_dir is None:
        output_dir = DATA_DIR / f"expresso_{speaker_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Find arrow files directly
    hf_data_dir = EXPRESSO_DIR / "hf_dataset" / "train"
    arrow_files = list(hf_data_dir.glob("*.arrow"))

    if not arrow_files:
        logger.error(f"No arrow files found in {hf_data_dir}")
        return {"error": "no_arrow_files"}

    logger.info(f"Found {len(arrow_files)} arrow shards")

    # Apply quality filters
    logger.info("")
    logger.info("Applying quality filters...")
    logger.info(f"  Min duration: {MIN_DURATION_SEC}s")
    logger.info(f"  Max duration: {MAX_DURATION_SEC}s")
    logger.info(f"  Min words: {MIN_WORDS}")
    logger.info(f"  Max silence ratio: {MAX_SILENCE_RATIO:.0%}")
    logger.info(f"  Min SNR: {MIN_SNR_DB}dB")
    logger.info("")

    filtered_samples = []
    rejection_reasons = {}
    total_duration = 0.0
    total_processed = 0

    for arrow_file in sorted(arrow_files):
        logger.info(f"Processing: {arrow_file.name}")

        # Read arrow file using IPC streaming format (HuggingFace format)
        with open(str(arrow_file), 'rb') as f:
            reader = ipc.open_stream(f)
            table = reader.read_all()
        df = table.to_pandas()

        # Filter for speaker
        speaker_df = df[df["speaker_id"] == speaker_id]

        for idx, row in speaker_df.iterrows():
            if max_samples and len(filtered_samples) >= max_samples:
                break

            total_processed += 1

            try:
                # Extract audio bytes from the nested structure
                audio_data = row["audio"]

                # Handle different formats
                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get("bytes", audio_data.get("array"))
                    path = audio_data.get("path", "")
                elif hasattr(audio_data, "as_py"):
                    audio_data = audio_data.as_py()
                    audio_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else audio_data
                    path = audio_data.get("path", "") if isinstance(audio_data, dict) else ""
                else:
                    audio_bytes = audio_data
                    path = ""

                if audio_bytes is None:
                    rejection_reasons["no_audio_bytes"] = rejection_reasons.get("no_audio_bytes", 0) + 1
                    continue

                # Decode audio
                try:
                    audio, sr = decode_audio_bytes(audio_bytes)
                except Exception as e:
                    rejection_reasons["decode_error"] = rejection_reasons.get("decode_error", 0) + 1
                    continue

                text = row.get("text", "")
                style = row.get("style", "default")
                sample_id = row.get("id", f"sample_{total_processed}")

                if not text:
                    rejection_reasons["no_text"] = rejection_reasons.get("no_text", 0) + 1
                    continue

                # Convert to mono if needed
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)

                # Apply quality filter
                passed, reason = passes_quality_filter(audio, sr, text)

                if not passed:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    continue

                # Save audio file
                duration = len(audio) / sr
                output_filename = f"{len(filtered_samples):05d}.wav"
                output_path = audio_dir / output_filename

                sf.write(str(output_path), audio, sr)

                filtered_samples.append({
                    "path": f"audio/{output_filename}",
                    "text": text,
                    "speaker": speaker_id,
                    "speaker_name": speaker_name,
                    "style": style,
                    "duration": duration,
                    "sample_rate": sr,
                    "original_id": sample_id,
                })

                total_duration += duration

            except Exception as e:
                rejection_reasons["error"] = rejection_reasons.get("error", 0) + 1
                continue

        if max_samples and len(filtered_samples) >= max_samples:
            break

        logger.info(f"  -> Kept {len(filtered_samples)} samples so far ({total_duration/60:.1f} min)")

    # Save metadata
    metadata = {
        "speaker": speaker_id,
        "speaker_name": speaker_name,
        "total_samples": len(filtered_samples),
        "total_duration_hours": total_duration / 3600,
        "total_duration_minutes": total_duration / 60,
        "original_sample_rate": 48000,
        "samples": filtered_samples,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Style distribution
    styles = {}
    for s in filtered_samples:
        style = s["style"]
        styles[style] = styles.get(style, 0) + 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Speaker: {speaker_id} ({speaker_name})")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Total samples: {len(filtered_samples)}")
    logger.info(f"Total duration: {total_duration/3600:.2f} hours ({total_duration/60:.0f} minutes)")
    logger.info("")
    logger.info("Style distribution:")
    for style, count in sorted(styles.items(), key=lambda x: -x[1]):
        logger.info(f"  {style}: {count}")
    logger.info("")
    logger.info("Rejection reasons:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count}")
    logger.info("=" * 60)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Extract single speaker from Expresso dataset")
    parser.add_argument(
        "--speaker",
        type=str,
        default="ex04",
        help="Speaker to extract: ex01 (jerry), ex02 (elisabeth), ex03 (thomas), ex04 (talia)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to extract"
    )

    args = parser.parse_args()
    output_dir = Path(args.output) if args.output else None
    result = extract_speaker(args.speaker, output_dir, args.max_samples)

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
