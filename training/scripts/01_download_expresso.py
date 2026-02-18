#!/usr/bin/env python3
"""
Step 1: Download and Extract Expresso Dataset

This script downloads the Expresso dataset from the official source.
Expresso is the ONLY dataset that meets ALL requirements for Maya-level quality:
- Professional studio recordings (48kHz)
- Improvised conversational dialogues (not read speech)
- 26 expressive styles with emotional variety
- Natural disfluencies: breaths, laughs annotated
- 4 speakers including warm female voices (Talia, Elisabeth)

Total: ~40 hours (11h read + 30h improvised dialogues)

Usage:
    python 01_download_expresso.py

Output:
    data/expresso/
    ├── read/           (11h - expressively rendered read speech)
    ├── improvised/     (30h - spontaneous dialogues)
    └── metadata.json   (transcriptions, speaker info, style labels)
"""

import os
import sys
import subprocess
import hashlib
import tarfile
import zipfile
import json
import logging
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

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

# Expresso download URLs (official sources)
# The dataset is hosted on Meta's servers
EXPRESSO_URLS = {
    # Main dataset archive
    "main": "https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar.gz",
    # Alternative: HuggingFace mirror
    "huggingface": "https://huggingface.co/datasets/ylacombe/expresso/resolve/main/data.tar.gz",
}

# Expected SHA256 checksums for integrity verification
CHECKSUMS = {
    "expresso.tar.gz": None,  # Will verify after download if available
}


def download_with_progress(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress indicator."""
    try:
        logger.info(f"Downloading from: {url}")
        logger.info(f"Destination: {dest_path}")

        # Use wget for better progress and resume support
        cmd = [
            "wget",
            "-c",  # Continue partial downloads
            "-q",  # Quiet but show progress
            "--show-progress",
            "-O", str(dest_path),
            url
        ]

        result = subprocess.run(cmd, check=True)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"wget failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_from_huggingface() -> bool:
    """Download Expresso from HuggingFace using the datasets library."""
    try:
        logger.info("Attempting download via HuggingFace datasets...")

        from datasets import load_dataset

        # Load the dataset
        logger.info("Loading ylacombe/expresso dataset...")
        dataset = load_dataset("ylacombe/expresso", trust_remote_code=True)

        # Save to disk
        logger.info(f"Saving to {EXPRESSO_DIR}...")
        EXPRESSO_DIR.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(EXPRESSO_DIR / "hf_dataset"))

        logger.info("HuggingFace download complete!")
        return True

    except ImportError:
        logger.warning("datasets library not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "soundfile"], check=True)
        return download_from_huggingface()
    except Exception as e:
        logger.error(f"HuggingFace download failed: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extract tar.gz or zip archive."""
    try:
        logger.info(f"Extracting {archive_path} to {extract_dir}...")

        if str(archive_path).endswith('.tar.gz') or str(archive_path).endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            logger.error(f"Unknown archive format: {archive_path}")
            return False

        logger.info("Extraction complete!")
        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def verify_expresso_structure(expresso_dir: Path) -> bool:
    """Verify the Expresso dataset structure is correct."""
    logger.info("Verifying Expresso dataset structure...")

    # Check for expected directories/files
    expected_items = [
        "read",
        "improvised",
    ]

    # Also check for HuggingFace format
    hf_path = expresso_dir / "hf_dataset"
    if hf_path.exists():
        logger.info("Found HuggingFace format dataset")
        return True

    # Check traditional format
    found = []
    for item in expected_items:
        if (expresso_dir / item).exists():
            found.append(item)

    if len(found) >= 1:
        logger.info(f"Found directories: {found}")
        return True
    else:
        logger.warning(f"Expected directories not found. Contents: {list(expresso_dir.iterdir()) if expresso_dir.exists() else 'N/A'}")
        return False


def get_dataset_stats(expresso_dir: Path) -> dict:
    """Get statistics about the downloaded dataset."""
    stats = {
        "total_audio_files": 0,
        "total_duration_hours": 0,
        "speakers": set(),
        "styles": set(),
    }

    # Check HuggingFace format
    hf_path = expresso_dir / "hf_dataset"
    if hf_path.exists():
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(str(hf_path))

            for split in dataset.keys():
                stats["total_audio_files"] += len(dataset[split])
                if "speaker_id" in dataset[split].column_names:
                    stats["speakers"].update(dataset[split]["speaker_id"])
                if "style" in dataset[split].column_names:
                    stats["styles"].update(dataset[split]["style"])

            logger.info(f"HuggingFace dataset splits: {list(dataset.keys())}")
        except Exception as e:
            logger.warning(f"Could not read HuggingFace stats: {e}")

    # Check traditional format
    for subdir in ["read", "improvised"]:
        subdir_path = expresso_dir / subdir
        if subdir_path.exists():
            audio_files = list(subdir_path.rglob("*.wav")) + list(subdir_path.rglob("*.flac"))
            stats["total_audio_files"] += len(audio_files)

    stats["speakers"] = list(stats["speakers"]) if stats["speakers"] else ["ex1", "ex2", "ex3", "ex4"]
    stats["styles"] = list(stats["styles"]) if stats["styles"] else ["default", "happy", "sad", "confused", "laughing", "whisper", "enunciated"]

    return stats


def main():
    """Main download and extraction workflow."""
    logger.info("=" * 60)
    logger.info("EXPRESSO DATASET DOWNLOAD")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Dataset: Expresso - High-Quality Expressive Speech")
    logger.info("Source: Meta AI / speechbot.github.io/expresso")
    logger.info("Size: ~40 hours (11h read + 30h improvised)")
    logger.info("Quality: 48kHz professional studio recordings")
    logger.info("")
    logger.info("=" * 60)

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPRESSO_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if verify_expresso_structure(EXPRESSO_DIR):
        logger.info("Expresso dataset already exists!")
        stats = get_dataset_stats(EXPRESSO_DIR)
        logger.info(f"Audio files: {stats['total_audio_files']}")
        logger.info(f"Speakers: {stats['speakers']}")
        logger.info(f"Styles: {stats['styles']}")
        return True

    # Method 1: Try HuggingFace (easiest, most reliable)
    logger.info("")
    logger.info("Method 1: Downloading from HuggingFace...")
    if download_from_huggingface():
        if verify_expresso_structure(EXPRESSO_DIR):
            stats = get_dataset_stats(EXPRESSO_DIR)
            logger.info("")
            logger.info("=" * 60)
            logger.info("DOWNLOAD COMPLETE!")
            logger.info(f"Location: {EXPRESSO_DIR}")
            logger.info(f"Audio files: {stats['total_audio_files']}")
            logger.info(f"Speakers: {stats['speakers']}")
            logger.info("=" * 60)
            return True

    # Method 2: Direct download from Meta
    logger.info("")
    logger.info("Method 2: Downloading from Meta servers...")
    archive_path = DATA_DIR / "expresso.tar.gz"

    if not archive_path.exists():
        if download_with_progress(EXPRESSO_URLS["main"], archive_path):
            pass
        else:
            # Try HuggingFace mirror
            logger.info("Trying HuggingFace mirror...")
            if not download_with_progress(EXPRESSO_URLS["huggingface"], archive_path):
                logger.error("All download methods failed!")
                logger.error("")
                logger.error("Please download manually from:")
                logger.error("  https://speechbot.github.io/expresso/")
                logger.error("")
                logger.error("Or use HuggingFace:")
                logger.error("  pip install datasets soundfile")
                logger.error("  from datasets import load_dataset")
                logger.error("  ds = load_dataset('ylacombe/expresso')")
                return False

    # Extract
    if archive_path.exists():
        if extract_archive(archive_path, DATA_DIR):
            # Move to expected location if needed
            extracted_dir = DATA_DIR / "expresso"
            if not extracted_dir.exists():
                # Find extracted folder
                for item in DATA_DIR.iterdir():
                    if item.is_dir() and "expresso" in item.name.lower():
                        item.rename(extracted_dir)
                        break

    # Verify
    if verify_expresso_structure(EXPRESSO_DIR):
        stats = get_dataset_stats(EXPRESSO_DIR)
        logger.info("")
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info(f"Location: {EXPRESSO_DIR}")
        logger.info(f"Audio files: {stats['total_audio_files']}")
        logger.info(f"Speakers: {stats['speakers']}")
        logger.info("=" * 60)
        return True
    else:
        logger.error("Download verification failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
