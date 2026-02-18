#!/usr/bin/env python3
"""
Test Post-Processing Impact on UTMOS Scores
=============================================
Apply audio humanizer (exciter, warmth, spectral cleanup) to generated
samples and re-score with UTMOS to measure perceptual quality impact.

Tests multiple configurations:
1. Raw (no processing)
2. Warmth only
3. Presence (exciter) only
4. Warmth + Presence
5. Full humanize (jitter + shimmer + warmth + presence)
6. Spectral cleanup only
7. Full pipeline (humanize + spectral cleanup)
"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
EVAL_DIR = PROJECT_ROOT / "training" / "eval_samples"


def load_utmos(device='cuda'):
    """Load UTMOS model."""
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0",
        "utmos22_strong",
        trust_repo=True,
    )
    predictor = predictor.to(device)
    predictor.eval()
    return predictor


def score_audio(predictor, audio_tensor, sr=24000, device='cuda'):
    """Score audio tensor with UTMOS."""
    if isinstance(audio_tensor, np.ndarray):
        audio_tensor = torch.from_numpy(audio_tensor)

    audio_tensor = audio_tensor.float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Resample to 16kHz for UTMOS
    audio_16k = torchaudio.functional.resample(audio_tensor, sr, 16000)
    audio_16k = audio_16k.to(device)

    with torch.no_grad():
        score = predictor(audio_16k, sr=16000)

    return score.item()


def main():
    device = 'cuda:0'

    # Load UTMOS
    logger.info("Loading UTMOS...")
    predictor = load_utmos(device)

    # Import humanizer functions
    from maya.engine.audio_humanizer import (
        add_warmth, add_presence, add_jitter, add_shimmer,
        humanize_audio, spectral_cleanup
    )

    # Define processing configurations
    configs = {
        "raw": lambda audio, sr: audio,
        "warmth_008": lambda audio, sr: torch.from_numpy(
            add_warmth(audio.cpu().numpy(), sr, amount=0.08)),
        "presence_012": lambda audio, sr: torch.from_numpy(
            add_presence(audio.cpu().numpy(), sr, amount=0.12)),
        "warmth+presence": lambda audio, sr: torch.from_numpy(
            add_presence(add_warmth(audio.cpu().numpy(), sr, amount=0.08), sr, amount=0.12)),
        "full_humanize": lambda audio, sr: humanize_audio(
            audio, sr, jitter=0.3, shimmer=1.0, breaths=False, warmth=0.08, presence=0.12),
        "spectral_cleanup": lambda audio, sr: spectral_cleanup(audio.to(device), sr, strength=0.25),
        "humanize+cleanup": lambda audio, sr: spectral_cleanup(
            humanize_audio(audio, sr, jitter=0.3, shimmer=1.0, breaths=False, warmth=0.08, presence=0.12).to(device),
            sr, strength=0.25),
    }

    # Test on combined_naturalness samples (our best model)
    test_dirs = ["combined_naturalness", "base_csm"]

    all_results = {}

    for test_dir in test_dirs:
        audio_dir = EVAL_DIR / test_dir
        if not audio_dir.exists():
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Testing post-processing on: {test_dir}")
        logger.info(f"{'='*80}")

        wav_files = sorted(audio_dir.glob("*.wav"))
        if not wav_files:
            continue

        dir_results = {}

        for config_name, processor in configs.items():
            scores = []
            for wav_file in wav_files:
                wav, sr = torchaudio.load(str(wav_file))
                wav = wav.squeeze(0)  # (samples,)

                # Apply processing
                try:
                    processed = processor(wav, sr)
                    if isinstance(processed, torch.Tensor):
                        processed = processed.float().cpu()
                        if processed.dim() > 1:
                            processed = processed.squeeze(0)
                    else:
                        processed = torch.from_numpy(processed).float()
                except Exception as e:
                    logger.warning(f"  {config_name} failed on {wav_file.name}: {e}")
                    processed = wav

                # Score
                score = score_audio(predictor, processed, sr, device)
                scores.append(score)

            avg = np.mean(scores)
            std = np.std(scores)
            dir_results[config_name] = {
                "mean": round(float(avg), 3),
                "std": round(float(std), 3),
                "scores": [round(s, 3) for s in scores],
            }
            logger.info(f"  {config_name:<25} MOS: {avg:.3f} ± {std:.3f}")

        all_results[test_dir] = dir_results

    # Print comparison table
    logger.info(f"\n{'='*80}")
    logger.info("POST-PROCESSING IMPACT ON UTMOS")
    logger.info(f"{'='*80}")

    for test_dir, dir_results in all_results.items():
        logger.info(f"\n{test_dir}:")
        raw_score = dir_results.get("raw", {}).get("mean", 0)
        logger.info(f"  {'Config':<25} {'MOS':<10} {'Delta':<10}")
        logger.info(f"  {'-'*50}")

        for config_name, result in sorted(dir_results.items(), key=lambda x: -x[1]["mean"]):
            delta = result["mean"] - raw_score
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            logger.info(f"  {config_name:<25} {result['mean']:<10.3f} {delta_str:<10}")

    # Save
    output_file = EVAL_DIR / "postprocessing_utmos.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
