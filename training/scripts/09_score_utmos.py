#!/usr/bin/env python3
"""
UTMOS Perceptual Quality Scoring
=================================
Score generated audio samples using UTMOS (Universal TTS MOS prediction).
Gives MOS-like scores (1-5 scale) correlated with human perception.

Usage:
    python 09_score_utmos.py                    # Score all eval_samples
    python 09_score_utmos.py --dir /path/to/wavs  # Score specific directory
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
EVAL_DIR = PROJECT_ROOT / "training" / "eval_samples"


def load_utmos(device='cuda'):
    """Load UTMOS model via torch.hub."""
    logger.info("Loading UTMOS model...")
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0",
        "utmos22_strong",
        trust_repo=True,
    )
    predictor = predictor.to(device)
    predictor.eval()
    logger.info("UTMOS ready")
    return predictor


def score_wav(predictor, wav_path, device='cuda'):
    """Score a single WAV file. Returns MOS score (1-5)."""
    wav, sr = torchaudio.load(str(wav_path))

    # UTMOS expects 16kHz mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav.to(device)

    with torch.no_grad():
        score = predictor(wav, sr=16000)

    return score.item()


def score_directory(predictor, audio_dir, device='cuda'):
    """Score all WAV files in a directory."""
    audio_dir = Path(audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        logger.warning(f"No WAV files in {audio_dir}")
        return {}

    results = {}
    for i, wav_file in enumerate(wav_files):
        try:
            score = score_wav(predictor, wav_file, device)
            results[wav_file.name] = {
                "score": round(score, 3),
                "path": str(wav_file),
            }
            logger.info(f"  [{i+1}/{len(wav_files)}] {wav_file.name}: {score:.3f}")
        except Exception as e:
            logger.error(f"  FAILED: {wav_file.name}: {e}")
            results[wav_file.name] = {"score": None, "error": str(e)}

    # Summary
    valid_scores = [r["score"] for r in results.values() if r["score"] is not None]
    if valid_scores:
        avg = np.mean(valid_scores)
        std = np.std(valid_scores)
        mn = np.min(valid_scores)
        mx = np.max(valid_scores)
        logger.info(f"  → Mean: {avg:.3f} ± {std:.3f} | Range: [{mn:.3f}, {mx:.3f}] | N={len(valid_scores)}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None,
                       help="Specific directory of WAV files to score")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    predictor = load_utmos(args.device)

    all_results = {}

    if args.dir:
        # Score specific directory
        name = Path(args.dir).name
        logger.info(f"\nScoring: {name}")
        results = score_directory(predictor, args.dir, args.device)
        all_results[name] = results
    else:
        # Score all subdirectories in eval_samples
        if not EVAL_DIR.exists():
            logger.error(f"Eval directory not found: {EVAL_DIR}")
            return

        subdirs = sorted([d for d in EVAL_DIR.iterdir() if d.is_dir()])
        if not subdirs:
            logger.error(f"No subdirectories in {EVAL_DIR}")
            return

        for subdir in subdirs:
            name = subdir.name
            logger.info(f"\n{'='*60}")
            logger.info(f"Scoring: {name}")
            logger.info(f"{'='*60}")
            results = score_directory(predictor, subdir, args.device)
            all_results[name] = results

    # Print comparison table
    logger.info(f"\n{'='*80}")
    logger.info("UTMOS COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<30} {'Mean MOS':<12} {'Std':<8} {'Min':<8} {'Max':<8} {'N':<5}")
    logger.info("-" * 80)

    summary = {}
    for model_name, results in sorted(all_results.items()):
        valid = [r["score"] for r in results.values() if r["score"] is not None]
        if valid:
            avg = np.mean(valid)
            std = np.std(valid)
            mn = np.min(valid)
            mx = np.max(valid)
            logger.info(f"{model_name:<30} {avg:<12.3f} {std:<8.3f} {mn:<8.3f} {mx:<8.3f} {len(valid):<5}")
            summary[model_name] = {
                "mean_mos": round(float(avg), 3),
                "std": round(float(std), 3),
                "min": round(float(mn), 3),
                "max": round(float(mx), 3),
                "n": len(valid),
            }

    # Per-sentence comparison if multiple models
    if len(all_results) > 1:
        logger.info(f"\n{'='*80}")
        logger.info("PER-SENTENCE BREAKDOWN")
        logger.info(f"{'='*80}")

        # Get all unique sentence indices
        all_files = set()
        for results in all_results.values():
            all_files.update(results.keys())

        # Group by sentence index (first 2 chars of filename: 00_, 01_, etc.)
        sentence_indices = sorted(set(f[:2] for f in all_files if f[:2].isdigit()))

        for idx in sentence_indices:
            scores_by_model = {}
            for model_name, results in sorted(all_results.items()):
                matching = [(k, v) for k, v in results.items() if k.startswith(idx)]
                if matching:
                    fname, result = matching[0]
                    if result["score"] is not None:
                        scores_by_model[model_name] = result["score"]

            if scores_by_model:
                parts = " | ".join(f"{m}: {s:.3f}" for m, s in sorted(scores_by_model.items()))
                best = max(scores_by_model, key=scores_by_model.get)
                logger.info(f"  Sentence {idx}: {parts} → Best: {best}")

    # Save results
    output_file = EVAL_DIR / "utmos_scores.json"
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "details": all_results}, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
