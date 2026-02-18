#!/usr/bin/env python3
"""
Evaluate CSM Model Quality
============================

Generate audio samples from different model checkpoints and compare quality.
Uses the official CSM Generator class for proper audio generation.

Usage:
    python 08_evaluate_model_quality.py --checkpoint /path/to/checkpoint
    python 08_evaluate_model_quality.py --all
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
OUTPUT_DIR = PROJECT_ROOT / "training" / "eval_samples"

# Test utterances spanning different speech styles
TEST_UTTERANCES = [
    # Natural conversation
    "Oh wow, I didn't expect that at all!",
    "Hmm, let me think about that for a second.",
    "Yeah, I totally get what you mean.",
    "Wait, are you serious right now?",

    # Emotional range
    "That's absolutely amazing, I'm so happy for you!",
    "I'm really sorry to hear that happened.",
    "Honestly, that makes me a little nervous.",

    # Disfluencies / natural speech
    "So, like, the thing is... it's complicated.",
    "Well, um, I guess we could try that approach.",
    "Right, right, okay so basically what happened was...",

    # Short responses (common in conversation)
    "Definitely!",
    "Oh no, really?",
    "That makes sense.",
    "I see what you mean.",
]


def load_generator(checkpoint_path=None, device='cuda'):
    """Load CSM Generator, optionally with checkpoint weights injected."""
    from models import Model
    from generator import Generator

    logger.info("Loading base CSM-1B model...")
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)

    if checkpoint_path:
        cp = Path(checkpoint_path)

        # Try merged model
        merged = cp / "model_merged.pt"
        if merged.exists():
            state = torch.load(merged, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded merged model from {merged}")
        else:
            # Try component files
            decoder_path = cp / "decoder.pt"
            proj_path = cp / "projection.pt"
            head_path = cp / "audio_head.pt"

            if decoder_path.exists():
                model.decoder.load_state_dict(
                    torch.load(decoder_path, map_location=device, weights_only=True))
            if proj_path.exists():
                model.projection.load_state_dict(
                    torch.load(proj_path, map_location=device, weights_only=True))
            if head_path.exists():
                head = torch.load(head_path, map_location=device, weights_only=True)
                model.audio_head.data.copy_(head.data)

            logger.info(f"Loaded checkpoint components from {cp}")

    logger.info("Creating Generator (loading mimi codec + tokenizer)...")
    gen = Generator(model)
    logger.info("Generator ready")
    return gen


def load_voice_context(generator, device='cuda'):
    """Load Maya voice prompt as a Segment for context."""
    from generator import Segment

    vp_path = PROJECT_ROOT / "assets" / "voice_prompt" / "maya_voice_prompt.pt"
    if vp_path.exists():
        vp = torch.load(vp_path, map_location=device, weights_only=False)
        if isinstance(vp, dict) and "audio" in vp:
            audio = vp["audio"]
            if audio.dim() > 1:
                audio = audio.squeeze(0)
            text = vp.get("text", "Hey, how's it going?")
            segment = Segment(speaker=0, text=text, audio=audio.to(device))
            logger.info(f"Voice context loaded: {audio.shape[0]/24000:.1f}s")
            return [segment]

    logger.warning("No voice prompt found - generating without context")
    return []


def compute_audio_metrics(audio: torch.Tensor, sr=24000) -> dict:
    """Compute basic audio quality metrics."""
    audio_np = audio.cpu().float().numpy()

    # Dynamic range
    peak = np.abs(audio_np).max()
    rms = np.sqrt(np.mean(audio_np ** 2))
    dynamic_range_db = 20 * np.log10(peak / (rms + 1e-8))

    # Silence ratio (frames below -40dB)
    frame_size = int(0.025 * sr)
    hop_size = int(0.010 * sr)
    frames = []
    for i in range(0, len(audio_np) - frame_size, hop_size):
        frame_energy = np.sqrt(np.mean(audio_np[i:i+frame_size] ** 2))
        frames.append(frame_energy)
    frames = np.array(frames)
    threshold = 10 ** (-40 / 20)
    silence_ratio = (frames < threshold).sum() / len(frames) if len(frames) > 0 else 0

    # Zero crossing rate (higher = more noise/fricatives)
    zcr = np.sum(np.abs(np.diff(np.sign(audio_np)))) / (2 * len(audio_np))

    # Duration
    duration = len(audio_np) / sr

    return {
        "duration_s": round(duration, 2),
        "peak": round(float(peak), 4),
        "rms": round(float(rms), 4),
        "dynamic_range_db": round(float(dynamic_range_db), 1),
        "silence_ratio": round(float(silence_ratio), 3),
        "zcr": round(float(zcr), 4),
    }


def evaluate_model(model_name, checkpoint_path, device, output_subdir):
    """Run full evaluation for a model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"Checkpoint: {checkpoint_path or 'base model'}")
    logger.info(f"{'='*60}")

    gen = load_generator(checkpoint_path, device)
    context = load_voice_context(gen, device)

    out_dir = OUTPUT_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_time = 0

    for i, text in enumerate(TEST_UTTERANCES):
        start = time.time()
        try:
            audio = gen.generate(
                text=text,
                speaker=0,
                context=context,
                max_audio_length_ms=8000,
                temperature=0.8,
                topk=50,
            )
            gen_time = time.time() - start
            total_time += gen_time

            # Save audio
            safe_name = text[:30].replace(' ', '_').replace(',', '').replace("'", '').replace('?', '').replace('!', '')
            audio_path = out_dir / f"{i:02d}_{safe_name}.wav"
            torchaudio.save(str(audio_path), audio.unsqueeze(0).cpu().float(), 24000)

            # Compute metrics
            metrics = compute_audio_metrics(audio)
            metrics["text"] = text
            metrics["gen_time_s"] = round(gen_time, 2)
            results.append(metrics)

            logger.info(f"  [{i+1}/{len(TEST_UTTERANCES)}] {gen_time:.1f}s | {metrics['duration_s']}s | DR={metrics['dynamic_range_db']}dB | Silence={metrics['silence_ratio']:.1%} | \"{text[:40]}\"")

        except Exception as e:
            logger.error(f"  FAILED: {text[:40]}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"text": text, "error": str(e)})

    # Summary
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_dr = np.mean([r["dynamic_range_db"] for r in valid])
        avg_silence = np.mean([r["silence_ratio"] for r in valid])
        avg_gen = np.mean([r["gen_time_s"] for r in valid])
        avg_dur = np.mean([r["duration_s"] for r in valid])

        summary = {
            "model": model_name,
            "checkpoint": str(checkpoint_path) if checkpoint_path else "base",
            "num_samples": len(valid),
            "avg_dynamic_range_db": round(float(avg_dr), 1),
            "avg_silence_ratio": round(float(avg_silence), 3),
            "avg_gen_time_s": round(float(avg_gen), 2),
            "avg_duration_s": round(float(avg_dur), 2),
            "total_time_s": round(total_time, 1),
        }

        logger.info(f"\nSummary for {model_name}:")
        logger.info(f"  Avg Dynamic Range: {avg_dr:.1f} dB")
        logger.info(f"  Avg Silence Ratio: {avg_silence:.1%}")
        logger.info(f"  Avg Gen Time: {avg_gen:.1f}s")
        logger.info(f"  Avg Duration: {avg_dur:.1f}s")

        with open(out_dir / "results.json", "w") as f:
            json.dump({"summary": summary, "samples": results}, f, indent=2)
    else:
        summary = None

    # Free model
    del gen
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Specific checkpoint to evaluate")
    parser.add_argument("--all", action="store_true",
                       help="Evaluate all available checkpoints")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        evaluate_model("custom", args.checkpoint, args.device, "custom")
        return

    # Evaluate available models
    summaries = []

    # 1. Base CSM (no fine-tuning)
    s = evaluate_model("Base CSM-1B", None, args.device, "base_csm")
    if s: summaries.append(s)

    # 2. Decoder-only (current training)
    dec_best = PROJECT_ROOT / "training/checkpoints/csm_maya_decoder_only/best_model"
    if dec_best.exists():
        s = evaluate_model("Decoder-Only (ex04)", str(dec_best), args.device, "decoder_only_ex04")
        if s: summaries.append(s)

    # 3. Combined naturalness
    comb_best = PROJECT_ROOT / "training/checkpoints/csm_maya_combined_naturalness/best_model"
    if comb_best.exists():
        s = evaluate_model("Combined Naturalness", str(comb_best), args.device, "combined_naturalness")
        if s: summaries.append(s)

    # Comparison table
    if len(summaries) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)
        logger.info(f"{'Model':<30} {'DR(dB)':<10} {'Silence%':<10} {'GenTime':<10}")
        logger.info("-" * 80)
        for s in summaries:
            logger.info(f"{s['model']:<30} {s['avg_dynamic_range_db']:<10.1f} "
                       f"{s['avg_silence_ratio']*100:<10.1f} {s['avg_gen_time_s']:<10.1f}")

    # Save comparison
    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
