#!/usr/bin/env python3
"""
GRPO Checkpoint Evaluator for CSM-1B
======================================
Comprehensive evaluation comparing GRPO checkpoints against base and SFT models.

Metrics:
- UTMOS (perceptual quality, 1-5)
- CER (character error rate)
- Speaker similarity (cosine sim)
- Pitch variance (prosody health - checks for monotone collapse)
- Dynamic range (dB)

Usage:
    python 15_eval_grpo.py --gpu 0                    # Eval all GRPO checkpoints
    python 15_eval_grpo.py --gpu 0 --checkpoint step-200  # Eval specific checkpoint
    python 15_eval_grpo.py --gpu 0 --compare-all      # Full comparison table
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

EVAL_SENTENCES = [
    # Verified NOT in grpo_prompts.txt or DEFAULT_PROMPTS
    "My neighbor's cat keeps showing up at my door every morning.",
    "I just found out they're closing the coffee shop on Elm Street.",
    "Sometimes I wonder what life would be like in a different city.",
    "The sunset yesterday was absolutely breathtaking, you should have seen it.",
    "I accidentally sent that text to the wrong person, how embarrassing.",
    "I don't think I've ever been this tired in my entire life.",
    "We should probably talk about what happened at dinner last night.",
    "I've been meaning to pick up a new hobby, maybe painting or something.",
    "The way she explained it actually made a lot more sense than I expected.",
    "I keep forgetting to water my plants, I'm terrible at this.",
    # Additional variety
    "You would not believe the traffic I dealt with this morning.",
    "I'm starting to think we should just go for it, what's the worst that could happen?",
    "She told me the funniest joke yesterday, I can't stop thinking about it.",
    "I've been sleeping terribly lately, I think it's the stress.",
]


def load_generator(device: str):
    """Load CSM generator."""
    from generator import Generator, load_csm_1b
    gen = load_csm_1b(device=device)
    return gen


def load_voice_context(device: str):
    """Load voice prompt."""
    from generator import Segment

    vp_path = PROJECT_ROOT / "assets/voice_prompt/maya_voice_prompt.pt"
    vp = torch.load(vp_path, map_location=device, weights_only=True)
    if isinstance(vp, dict):
        ctx_audio = vp["audio"].to(device)
        ctx_text = vp.get("text", "Hey, how's it going? I'm Maya, nice to meet you!")
    else:
        ctx_audio = vp.to(device)
        ctx_text = "Hey, how's it going? I'm Maya, nice to meet you!"

    if ctx_audio.dim() == 2:
        ctx_audio = ctx_audio[0]

    return [Segment(text=ctx_text, speaker=0, audio=ctx_audio)]


def load_eval_models(device: str):
    """Load UTMOS, Whisper, speaker encoder."""
    models = {}

    # UTMOS
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos = utmos.to(device).eval()
    models["utmos"] = utmos

    # Whisper
    import whisper
    models["whisper"] = whisper.load_model("base.en", device=device)

    # Speaker encoder
    from resemblyzer import VoiceEncoder, preprocess_wav
    models["speaker_encoder"] = VoiceEncoder(device=device)

    return models


def compute_pitch_variance(audio: torch.Tensor, sr: int = 24000) -> float:
    """Compute pitch variance to detect monotone collapse."""
    try:
        import torchaudio.functional as F
        # Detect pitch using autocorrelation
        audio_np = audio.cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np[0]

        # Simple pitch detection via autocorrelation
        frame_len = int(sr * 0.03)  # 30ms frames
        hop_len = int(sr * 0.01)    # 10ms hop
        pitches = []

        for i in range(0, len(audio_np) - frame_len, hop_len):
            frame = audio_np[i:i+frame_len]
            if np.max(np.abs(frame)) < 0.01:  # Skip silence
                continue

            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            corr = corr / corr[0]

            # Find first peak after 2ms (500Hz max)
            min_lag = int(sr * 0.002)
            max_lag = int(sr * 0.02)  # 50Hz min

            if max_lag > len(corr):
                continue

            peak_region = corr[min_lag:max_lag]
            if len(peak_region) == 0:
                continue

            peak_idx = np.argmax(peak_region) + min_lag
            if corr[peak_idx] > 0.3:  # Voiced threshold
                f0 = sr / peak_idx
                if 50 < f0 < 500:
                    pitches.append(f0)

        if len(pitches) < 3:
            return 0.0

        return float(np.std(pitches))
    except Exception:
        return 0.0


def compute_dynamic_range(audio: torch.Tensor) -> float:
    """Compute dynamic range in dB."""
    audio_np = audio.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np[0]
    audio_abs = np.abs(audio_np)
    audio_abs = audio_abs[audio_abs > 0.001]  # Remove near-silence
    if len(audio_abs) < 100:
        return 0.0
    p95 = np.percentile(audio_abs, 95)
    p5 = np.percentile(audio_abs, 5)
    if p5 < 1e-6:
        p5 = 1e-6
    return float(20 * np.log10(p95 / p5))


def evaluate_model(
    generator, context, eval_models, device,
    sentences=None, label="model"
):
    """Evaluate a model configuration on all sentences."""
    if sentences is None:
        sentences = EVAL_SENTENCES

    results = {
        "utmos_scores": [],
        "cer_scores": [],
        "sim_scores": [],
        "pitch_vars": [],
        "dr_scores": [],
        "durations": [],
        "gen_times": [],
    }

    # Compute reference speaker embedding
    ref_audio = context[0].audio.cpu().numpy()
    ref_16k = torchaudio.functional.resample(
        torch.from_numpy(ref_audio).unsqueeze(0), 24000, 16000
    ).squeeze(0).numpy()
    from resemblyzer import preprocess_wav
    ref_wav = preprocess_wav(ref_16k, source_sr=16000)
    ref_embed = eval_models["speaker_encoder"].embed_utterance(ref_wav)

    for i, text in enumerate(sentences):
        t0 = time.time()
        audio = generator.generate(
            text=text, speaker=0, context=context,
            max_audio_length_ms=5000, temperature=0.8, topk=50
        )
        gen_time = time.time() - t0
        results["gen_times"].append(gen_time)
        results["durations"].append(len(audio) / 24000)

        # UTMOS
        audio_16k = torchaudio.functional.resample(audio.unsqueeze(0), 24000, 16000).to(device)
        with torch.no_grad():
            utmos_score = eval_models["utmos"](audio_16k, sr=16000).item()
        results["utmos_scores"].append(utmos_score)

        # CER
        audio_np_16k = torchaudio.functional.resample(
            audio.unsqueeze(0).cpu(), 24000, 16000
        ).squeeze(0).numpy()
        whisper_result = eval_models["whisper"].transcribe(audio_np_16k, language="en")
        transcript = whisper_result["text"].strip()
        ref_text = text.lower().strip()
        hyp_text = transcript.lower().strip()
        import difflib
        cer = 1.0 - difflib.SequenceMatcher(None, ref_text, hyp_text).ratio()
        results["cer_scores"].append(cer)

        # Speaker similarity
        gen_wav = preprocess_wav(audio_np_16k, source_sr=16000)
        if len(gen_wav) > 1600:
            gen_embed = eval_models["speaker_encoder"].embed_utterance(gen_wav)
            sim = float(np.dot(gen_embed, ref_embed) / (
                np.linalg.norm(gen_embed) * np.linalg.norm(ref_embed) + 1e-8
            ))
        else:
            sim = 0.0
        results["sim_scores"].append(sim)

        # Pitch variance
        pv = compute_pitch_variance(audio)
        results["pitch_vars"].append(pv)

        # Dynamic range
        dr = compute_dynamic_range(audio)
        results["dr_scores"].append(dr)

    # Compute averages
    summary = {
        "label": label,
        "utmos_mean": np.mean(results["utmos_scores"]),
        "utmos_std": np.std(results["utmos_scores"]),
        "cer_mean": np.mean(results["cer_scores"]),
        "sim_mean": np.mean(results["sim_scores"]),
        "pitch_var_mean": np.mean(results["pitch_vars"]),
        "dr_mean": np.mean(results["dr_scores"]),
        "duration_mean": np.mean(results["durations"]),
        "gen_time_mean": np.mean(results["gen_times"]),
        "n_sentences": len(sentences),
    }

    return summary, results


def load_checkpoint_into_generator(generator, checkpoint_path, device):
    """Load a checkpoint into the generator's model."""
    ckpt_path = Path(checkpoint_path)

    if (ckpt_path / "model_merged.pt").exists():
        state = torch.load(ckpt_path / "model_merged.pt", map_location=device, weights_only=True)
        generator._model.load_state_dict(state, strict=False)
    elif (ckpt_path / "decoder.pt").exists():
        from models import Model
        # Reload base first
        base_state = Model.from_pretrained("sesame/csm-1b").state_dict()
        generator._model.load_state_dict(base_state, strict=False)
        generator._model.to(device=device, dtype=torch.bfloat16)

        decoder_state = torch.load(ckpt_path / "decoder.pt", map_location=device, weights_only=True)
        generator._model.decoder.load_state_dict(decoder_state)
        proj_state = torch.load(ckpt_path / "projection.pt", map_location=device, weights_only=True)
        generator._model.projection.load_state_dict(proj_state)
        head_data = torch.load(ckpt_path / "audio_head.pt", map_location=device, weights_only=True)
        generator._model.audio_head.data = head_data.to(device)
    else:
        raise FileNotFoundError(f"No recognized checkpoint files in {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="",
                       help="Specific checkpoint name (e.g., step-200, best_model, final)")
    parser.add_argument("--compare-all", action="store_true",
                       help="Compare all available checkpoints")
    parser.add_argument("--grpo-dir", type=str,
                       default=str(PROJECT_ROOT / "training/checkpoints/csm_maya_grpo"))
    parser.add_argument("--output", type=str, default="",
                       help="Output JSON path for results")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    logger.info("Loading generator...")
    generator = load_generator(device)
    context = load_voice_context(device)

    logger.info("Loading eval models...")
    eval_models = load_eval_models(device)

    all_results = []

    if args.compare_all:
        # Evaluate base CSM, SFT-1500, and all GRPO checkpoints
        models_to_eval = []

        # Base CSM
        models_to_eval.append(("Base CSM-1B", None))

        # SFT checkpoint-1500
        sft_path = PROJECT_ROOT / "training/checkpoints/csm_maya_combined_naturalness/checkpoint-1500-merged"
        if sft_path.exists():
            models_to_eval.append(("SFT ckpt-1500", str(sft_path)))

        # GRPO checkpoints
        grpo_dir = Path(args.grpo_dir)
        if grpo_dir.exists():
            for ckpt in sorted(grpo_dir.iterdir()):
                if ckpt.is_dir() and (ckpt / "model_merged.pt").exists():
                    models_to_eval.append((f"GRPO {ckpt.name}", str(ckpt)))

        for label, ckpt_path in models_to_eval:
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {label}")

            if ckpt_path:
                load_checkpoint_into_generator(generator, ckpt_path, device)
            else:
                # Reload base model
                from models import Model
                base_state = Model.from_pretrained("sesame/csm-1b").state_dict()
                generator._model.load_state_dict(base_state, strict=False)
                generator._model.to(device=device, dtype=torch.bfloat16)

            summary, details = evaluate_model(generator, context, eval_models, device, label=label)
            all_results.append(summary)

            logger.info(
                f"  UTMOS: {summary['utmos_mean']:.3f}±{summary['utmos_std']:.3f} | "
                f"CER: {summary['cer_mean']:.3f} | "
                f"SIM: {summary['sim_mean']:.3f} | "
                f"Pitch: {summary['pitch_var_mean']:.1f}Hz | "
                f"DR: {summary['dr_mean']:.1f}dB"
            )

    elif args.checkpoint:
        ckpt_path = Path(args.grpo_dir) / args.checkpoint
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            return

        load_checkpoint_into_generator(generator, str(ckpt_path), device)
        summary, details = evaluate_model(
            generator, context, eval_models, device,
            label=f"GRPO {args.checkpoint}"
        )
        all_results.append(summary)
        logger.info(f"Results: {json.dumps(summary, indent=2)}")

    else:
        # Evaluate latest GRPO checkpoint
        grpo_dir = Path(args.grpo_dir)
        if grpo_dir.exists():
            latest = None
            for ckpt in sorted(grpo_dir.iterdir()):
                if ckpt.is_dir():
                    latest = ckpt
            if latest:
                load_checkpoint_into_generator(generator, str(latest), device)
                summary, details = evaluate_model(
                    generator, context, eval_models, device,
                    label=f"GRPO {latest.name}"
                )
                all_results.append(summary)
                logger.info(f"Results: {json.dumps(summary, indent=2)}")

    # Print comparison table
    if len(all_results) > 1:
        logger.info(f"\n{'='*80}")
        logger.info("COMPARISON TABLE")
        logger.info(f"{'='*80}")
        logger.info(f"{'Model':<25} {'UTMOS':>8} {'CER':>8} {'SIM':>8} {'Pitch':>8} {'DR(dB)':>8}")
        logger.info("-" * 80)
        for r in all_results:
            logger.info(
                f"{r['label']:<25} {r['utmos_mean']:>8.3f} {r['cer_mean']:>8.3f} "
                f"{r['sim_mean']:>8.3f} {r['pitch_var_mean']:>8.1f} {r['dr_mean']:>8.1f}"
            )

    # Save results
    output_path = args.output or str(PROJECT_ROOT / "training/eval_samples/grpo_eval_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
