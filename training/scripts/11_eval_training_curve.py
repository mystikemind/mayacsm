#!/usr/bin/env python3
"""
Evaluate Training Curve - Score All Checkpoints
=================================================
Evaluates all available checkpoints with UTMOS + basic audio metrics
to track quality improvement over training steps.

Usage:
    python 11_eval_training_curve.py                     # Eval all checkpoints
    python 11_eval_training_curve.py --checkpoint 3500   # Eval specific step
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path

import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints" / "csm_maya_combined_naturalness"
OUTPUT_FILE = PROJECT_ROOT / "training" / "eval_samples" / "training_curve.json"

# Quick eval set (5 diverse sentences for speed)
EVAL_SENTENCES = [
    "Oh wow, I didn't expect that at all!",
    "Hmm, let me think about that for a second.",
    "That's absolutely amazing, I'm so happy for you!",
    "So, like, the thing is... it's complicated.",
    "Definitely!",
]


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


def load_generator(checkpoint_path=None, device='cuda'):
    """Load CSM Generator with optional checkpoint."""
    from models import Model
    from generator import Generator

    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)

    if checkpoint_path:
        cp = Path(checkpoint_path)
        merged = cp / "model_merged.pt"
        if merged.exists():
            state = torch.load(merged, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
        else:
            for fname, attr in [("decoder.pt", "decoder"), ("projection.pt", "projection")]:
                fpath = cp / fname
                if fpath.exists():
                    getattr(model, attr).load_state_dict(
                        torch.load(fpath, map_location=device, weights_only=True))
            head_path = cp / "audio_head.pt"
            if head_path.exists():
                head = torch.load(head_path, map_location=device, weights_only=True)
                model.audio_head.data.copy_(head.data)

    gen = Generator(model)
    return gen


def load_voice_context(generator, device='cuda'):
    """Load Maya voice prompt."""
    from generator import Segment

    vp_path = PROJECT_ROOT / "assets" / "voice_prompt" / "maya_voice_prompt.pt"
    if vp_path.exists():
        vp = torch.load(vp_path, map_location=device, weights_only=False)
        if isinstance(vp, dict) and "audio" in vp:
            audio = vp["audio"]
            if audio.dim() > 1:
                audio = audio.squeeze(0)
            text = vp.get("text", "Hey, how's it going?")
            return [Segment(speaker=0, text=text, audio=audio.to(device))]
    return []


def score_audio_utmos(predictor, audio_tensor, sr=24000, device='cuda'):
    """Score with UTMOS."""
    audio = audio_tensor.float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio_16k = torchaudio.functional.resample(audio, sr, 16000).to(device)
    with torch.no_grad():
        score = predictor(audio_16k, sr=16000)
    return score.item()


def compute_metrics(audio_tensor, sr=24000):
    """Compute basic audio metrics."""
    audio_np = audio_tensor.cpu().float().numpy()

    peak = np.abs(audio_np).max()
    rms = np.sqrt(np.mean(audio_np ** 2))
    dr_db = 20 * np.log10(peak / (rms + 1e-8))

    # Silence ratio
    frame_size = int(0.025 * sr)
    hop = int(0.010 * sr)
    frames = []
    for i in range(0, len(audio_np) - frame_size, hop):
        frames.append(np.sqrt(np.mean(audio_np[i:i+frame_size] ** 2)))
    frames = np.array(frames)
    threshold = 10 ** (-40 / 20)
    silence = (frames < threshold).sum() / len(frames) if len(frames) > 0 else 0

    duration = len(audio_np) / sr

    return {
        "duration": round(float(duration), 2),
        "dr_db": round(float(dr_db), 1),
        "silence_pct": round(float(silence * 100), 1),
    }


def eval_checkpoint(step, checkpoint_path, utmos_predictor, device):
    """Evaluate a single checkpoint."""
    logger.info(f"  Loading checkpoint step {step}...")
    gen = load_generator(checkpoint_path, device)
    context = load_voice_context(gen, device)

    utmos_scores = []
    dr_values = []
    silence_values = []
    durations = []
    gen_times = []

    for i, text in enumerate(EVAL_SENTENCES):
        start = time.time()
        try:
            audio = gen.generate(
                text=text, speaker=0, context=context,
                max_audio_length_ms=8000, temperature=0.8, topk=50,
            )
            gen_time = time.time() - start

            mos = score_audio_utmos(utmos_predictor, audio, device=device)
            metrics = compute_metrics(audio)

            utmos_scores.append(mos)
            dr_values.append(metrics["dr_db"])
            silence_values.append(metrics["silence_pct"])
            durations.append(metrics["duration"])
            gen_times.append(gen_time)

        except Exception as e:
            logger.warning(f"    Failed on '{text[:30]}': {e}")

    # Free model
    del gen
    torch.cuda.empty_cache()

    if utmos_scores:
        result = {
            "step": step,
            "checkpoint": str(checkpoint_path),
            "utmos_mean": round(float(np.mean(utmos_scores)), 3),
            "utmos_std": round(float(np.std(utmos_scores)), 3),
            "utmos_min": round(float(np.min(utmos_scores)), 3),
            "utmos_max": round(float(np.max(utmos_scores)), 3),
            "dr_db_mean": round(float(np.mean(dr_values)), 1),
            "silence_pct_mean": round(float(np.mean(silence_values)), 1),
            "duration_mean": round(float(np.mean(durations)), 2),
            "gen_time_mean": round(float(np.mean(gen_times)), 1),
            "n_samples": len(utmos_scores),
            "utmos_per_sentence": [round(s, 3) for s in utmos_scores],
        }
        logger.info(f"  Step {step}: UTMOS={result['utmos_mean']:.3f} | DR={result['dr_db_mean']}dB | Silence={result['silence_pct_mean']}%")
        return result

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, default=None,
                       help="Specific checkpoint step to evaluate")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device

    # Load UTMOS once
    logger.info("Loading UTMOS model...")
    utmos_predictor = load_utmos(device)

    # Load existing results
    existing_results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            data = json.load(f)
            for r in data.get("checkpoints", []):
                existing_results[r["step"]] = r

    # Find checkpoints to evaluate
    if args.checkpoint:
        steps_to_eval = [args.checkpoint]
    else:
        # Discover all checkpoints
        steps_to_eval = []
        for d in sorted(CHECKPOINT_DIR.iterdir()):
            if d.name.startswith("checkpoint-"):
                step = int(d.name.split("-")[1])
                if step not in existing_results:
                    steps_to_eval.append(step)

        # Also eval base model (step 0) if not done
        if 0 not in existing_results:
            steps_to_eval.insert(0, 0)

        # And best_model if not done
        if -1 not in existing_results and (CHECKPOINT_DIR / "best_model").exists():
            steps_to_eval.append(-1)

    if not steps_to_eval:
        logger.info("All checkpoints already evaluated! Showing results...")
    else:
        logger.info(f"Checkpoints to evaluate: {steps_to_eval}")

    # Evaluate each
    for step in steps_to_eval:
        logger.info(f"\n{'='*60}")

        if step == 0:
            logger.info(f"Evaluating: Base CSM-1B (no training)")
            result = eval_checkpoint(0, None, utmos_predictor, device)
        elif step == -1:
            logger.info(f"Evaluating: Best Model")
            result = eval_checkpoint(-1, str(CHECKPOINT_DIR / "best_model"), utmos_predictor, device)
        else:
            cp_path = CHECKPOINT_DIR / f"checkpoint-{step}"
            if not cp_path.exists():
                logger.warning(f"Checkpoint {step} not found")
                continue
            logger.info(f"Evaluating: Checkpoint {step}")
            result = eval_checkpoint(step, str(cp_path), utmos_predictor, device)

        if result:
            existing_results[step] = result

    # Sort results by step
    sorted_results = sorted(existing_results.values(), key=lambda x: x["step"])

    # Print training curve
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING QUALITY CURVE")
    logger.info(f"{'='*80}")
    logger.info(f"{'Step':<10} {'UTMOS':<10} {'DR(dB)':<10} {'Silence%':<10} {'GenTime':<10}")
    logger.info("-" * 60)

    for r in sorted_results:
        step_label = "base" if r["step"] == 0 else ("best" if r["step"] == -1 else str(r["step"]))
        logger.info(f"{step_label:<10} {r['utmos_mean']:<10.3f} {r['dr_db_mean']:<10.1f} "
                    f"{r['silence_pct_mean']:<10.1f} {r['gen_time_mean']:<10.1f}")

    # Best UTMOS checkpoint
    best = max(sorted_results, key=lambda x: x["utmos_mean"])
    step_label = "base" if best["step"] == 0 else ("best" if best["step"] == -1 else str(best["step"]))
    logger.info(f"\nBest UTMOS: {best['utmos_mean']:.3f} at step {step_label}")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_checkpoints": len(sorted_results),
            "best_utmos_step": best["step"],
            "best_utmos": best["utmos_mean"],
            "checkpoints": sorted_results,
        }, f, indent=2)
    logger.info(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
