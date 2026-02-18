#!/usr/bin/env python3
"""
Post-GRPO Evaluation & Integration Script
==========================================
Runs after GRPO training completes to:
1. Find the best checkpoint across all GRPO steps
2. Compare against base CSM and production Orpheus
3. Generate head-to-head audio samples
4. Produce a decision report on which model to deploy

Usage:
    CUDA_VISIBLE_DEVICES=3 python 16_post_grpo_evaluation.py
"""

import os
import sys
import json
import logging
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
CHECKPOINT_DIR = PROJECT_ROOT / "training/checkpoints/csm_maya_grpo_v3"
OUTPUT_DIR = PROJECT_ROOT / "training/evaluation/post_grpo"

# Evaluation sentences NOT in any training data
EVAL_SENTENCES = [
    "I just realized I left my keys in the car again.",
    "That new restaurant downtown has amazing pasta, you should try it.",
    "Sometimes the best thing you can do is just take a deep breath.",
    "I'm not sure what to think about that, it's kind of complicated.",
    "My sister called me yesterday and told me the most hilarious story.",
    "You know what, I think we should just go for a walk instead.",
    "The weather has been so unpredictable lately, I never know what to wear.",
    "I've been trying to learn guitar but my fingers keep hurting.",
    "Honestly, I think that was one of the best movies I've seen all year.",
    "Can you believe it's already February? Time really flies.",
]


def load_eval_models(device: str):
    """Load evaluation models (UTMOS, Whisper, speaker encoder)."""
    models = {}

    # UTMOS
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos = utmos.to(device).eval()
    models["utmos"] = utmos

    # Whisper for CER
    import whisper
    models["whisper"] = whisper.load_model("base.en", device=device)

    # Speaker encoder for similarity
    from resemblyzer import VoiceEncoder
    models["speaker_encoder"] = VoiceEncoder(device=device)

    return models


def score_utmos(audio: torch.Tensor, model, device: str) -> float:
    """Score audio with UTMOS."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    # UTMOS expects 16kHz
    if audio.shape[-1] > 16000 * 0.5:  # At least 0.5s
        audio_16k = torchaudio.functional.resample(audio, 24000, 16000)
        with torch.inference_mode():
            score = model(audio_16k.to(device), 16000)
        return score.item()
    return 0.0


def compute_cer(audio: torch.Tensor, text: str, whisper_model) -> float:
    """Compute Character Error Rate."""
    import jiwer
    audio_np = audio.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np[0]

    # Resample to 16kHz for Whisper
    audio_16k = torchaudio.functional.resample(
        torch.from_numpy(audio_np).unsqueeze(0), 24000, 16000
    ).squeeze(0).numpy()

    result = whisper_model.transcribe(audio_16k, language="en", fp16=True, verbose=False)
    hypothesis = result["text"].strip().lower()
    reference = text.strip().lower()

    if not reference:
        return 0.0

    return jiwer.cer(reference, hypothesis)


def compute_speaker_sim(audio: torch.Tensor, ref_embed, encoder) -> float:
    """Compute speaker similarity."""
    from resemblyzer import preprocess_wav
    audio_np = audio.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np[0]

    # Resample to 16kHz
    audio_16k = torchaudio.functional.resample(
        torch.from_numpy(audio_np).unsqueeze(0), 24000, 16000
    ).squeeze(0).numpy()

    wav = preprocess_wav(audio_16k, source_sr=16000)
    if len(wav) < 1600:  # Too short
        return 0.0

    embed = encoder.embed_utterance(wav)
    sim = np.dot(embed, ref_embed) / (np.linalg.norm(embed) * np.linalg.norm(ref_embed))
    return float(sim)


def load_csm_generator(device: str, checkpoint_path=None):
    """Load CSM-1B generator, optionally with fine-tuned checkpoint."""
    from generator import Generator, load_csm_1b

    gen = load_csm_1b(device=device)

    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt_dir = Path(checkpoint_path)

        # Load decoder, projection, audio_head
        if (ckpt_dir / "decoder.pt").exists():
            decoder_state = torch.load(ckpt_dir / "decoder.pt", map_location=device, weights_only=True)
            gen._model.decoder.load_state_dict(decoder_state)
            logger.info("  Loaded decoder weights")

        if (ckpt_dir / "projection.pt").exists():
            proj_state = torch.load(ckpt_dir / "projection.pt", map_location=device, weights_only=True)
            gen._model.projection.load_state_dict(proj_state)
            logger.info("  Loaded projection weights")

        if (ckpt_dir / "audio_head.pt").exists():
            audio_head = torch.load(ckpt_dir / "audio_head.pt", map_location=device, weights_only=True)
            gen._model.audio_head.data.copy_(audio_head)
            logger.info("  Loaded audio_head weights")

    return gen


def load_voice_context(device: str):
    """Load voice prompt for CSM generation."""
    from generator import Segment

    vp_path = PROJECT_ROOT / "assets/voice_prompt/maya_voice_prompt.pt"
    vp = torch.load(vp_path, map_location=device, weights_only=True)
    if isinstance(vp, dict):
        ctx_audio = vp["audio"].to(device)
        ctx_text = vp.get("text", "Hey, how\'s it going? I\'m Maya, nice to meet you!")
    else:
        ctx_audio = vp.to(device)
        ctx_text = "Hey, how\'s it going? I\'m Maya, nice to meet you!"

    if ctx_audio.dim() == 2:
        ctx_audio = ctx_audio[0]

    return [Segment(text=ctx_text, speaker=0, audio=ctx_audio)]


def generate_csm_audio(gen, text: str, context, device: str) -> torch.Tensor:
    """Generate audio with CSM generator."""
    from generator import Segment

    with torch.inference_mode():
        audio = gen.generate(
            text=text,
            speaker=0,
            context=context,
            max_audio_length_ms=10000,
            temperature=0.8,
            topk=50,
        )
    return audio


def evaluate_model(name: str, gen, context, device: str, eval_models: dict, ref_embed):
    """Run full evaluation on a model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {name}")
    logger.info(f"{'='*60}")

    results = {"name": name, "samples": []}
    all_utmos = []
    all_cer = []
    all_sim = []
    total_gen_time = 0

    for i, text in enumerate(EVAL_SENTENCES):
        logger.info(f"  [{i+1}/{len(EVAL_SENTENCES)}] '{text[:50]}...'")

        t0 = time.time()
        audio = generate_csm_audio(gen, text, context, device)
        gen_time = time.time() - t0

        if audio is None or audio.numel() < 2400:  # < 0.1s
            logger.warning(f"    Failed to generate audio for sample {i}")
            continue

        total_gen_time += gen_time
        duration = audio.shape[-1] / 24000
        rtf = gen_time / duration

        # Score
        utmos = score_utmos(audio, eval_models["utmos"], device)
        cer = compute_cer(audio, text, eval_models["whisper"])
        sim = compute_speaker_sim(audio, ref_embed, eval_models["speaker_encoder"])

        all_utmos.append(utmos)
        all_cer.append(cer)
        all_sim.append(sim)

        logger.info(f"    UTMOS={utmos:.3f} CER={cer:.3f} SIM={sim:.3f} "
                     f"RTF={rtf:.2f} ({duration:.1f}s in {gen_time:.1f}s)")

        results["samples"].append({
            "text": text,
            "utmos": utmos,
            "cer": cer,
            "sim": sim,
            "rtf": rtf,
            "duration": duration,
            "gen_time": gen_time,
        })

        # Save audio
        out_dir = OUTPUT_DIR / name.replace(" ", "_").lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = out_dir / f"sample_{i:02d}.wav"
        torchaudio.save(str(wav_path), audio.unsqueeze(0).cpu(), 24000)

    # Summary
    if all_utmos:
        results["avg_utmos"] = float(np.mean(all_utmos))
        results["std_utmos"] = float(np.std(all_utmos))
        results["avg_cer"] = float(np.mean(all_cer))
        results["avg_sim"] = float(np.mean(all_sim))
        results["avg_rtf"] = total_gen_time / sum(s["duration"] for s in results["samples"])

        logger.info(f"\n  SUMMARY: UTMOS={results['avg_utmos']:.3f}+-{results['std_utmos']:.3f} "
                     f"CER={results['avg_cer']:.4f} SIM={results['avg_sim']:.3f} "
                     f"RTF={results['avg_rtf']:.2f}")

    return results


def find_best_checkpoint():
    """Find the best GRPO checkpoint based on training_state.json."""
    best_dir = CHECKPOINT_DIR / "best_model"
    if best_dir.exists():
        state_file = best_dir / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            logger.info(f"Best model: step {state['step']}, reward {state['best_reward']:.4f}")
            return str(best_dir), state

    # Fallback: find all step checkpoints
    checkpoints = sorted(CHECKPOINT_DIR.glob("step-*"))
    if checkpoints:
        return str(checkpoints[-1]), None

    return None, None


def main():
    device = "cuda"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading evaluation models...")
    eval_models = load_eval_models(device)

    logger.info("Loading voice context...")
    context = load_voice_context(device)

    # Get reference speaker embedding
    from resemblyzer import preprocess_wav
    vp_path = PROJECT_ROOT / "assets/voice_prompt/maya_voice_prompt.pt"
    vp = torch.load(vp_path, map_location="cpu", weights_only=True)
    ref_audio = vp["audio"] if isinstance(vp, dict) else vp
    if ref_audio.dim() == 2:
        ref_audio = ref_audio[0]
    ref_16k = torchaudio.functional.resample(ref_audio.unsqueeze(0), 24000, 16000).squeeze(0).numpy()
    ref_wav = preprocess_wav(ref_16k, source_sr=16000)
    ref_embed = eval_models["speaker_encoder"].embed_utterance(ref_wav)

    all_results = []

    # 1. Evaluate base CSM-1B
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Base CSM-1B (no fine-tuning)")
    gen_base = load_csm_generator(device, checkpoint_path=None)
    results_base = evaluate_model("Base CSM-1B", gen_base, context, device, eval_models, ref_embed)
    all_results.append(results_base)
    del gen_base
    torch.cuda.empty_cache()

    # 2. Evaluate SFT checkpoint (combined naturalness)
    sft_ckpt = PROJECT_ROOT / "training/checkpoints/csm_maya_combined_naturalness/checkpoint-5000"
    if sft_ckpt.exists():
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: SFT Model (combined naturalness checkpoint-5000)")
        gen_sft = load_csm_generator(device, checkpoint_path=str(sft_ckpt))
        results_sft = evaluate_model("SFT Combined", gen_sft, context, device, eval_models, ref_embed)
        all_results.append(results_sft)
        del gen_sft
        torch.cuda.empty_cache()

    # 3. Evaluate best GRPO checkpoint
    best_ckpt_path, best_state = find_best_checkpoint()
    if best_ckpt_path:
        logger.info("\n" + "="*60)
        logger.info(f"PHASE 3: GRPO Best Model ({best_ckpt_path})")
        gen_grpo_best = load_csm_generator(device, checkpoint_path=best_ckpt_path)
        results_grpo = evaluate_model("GRPO Best", gen_grpo_best, context, device, eval_models, ref_embed)
        all_results.append(results_grpo)
        del gen_grpo_best
        torch.cuda.empty_cache()

    # 4. Evaluate latest GRPO checkpoint (might be different from best)
    checkpoints = sorted(CHECKPOINT_DIR.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if checkpoints:
        latest_ckpt = str(checkpoints[-1])
        if latest_ckpt != best_ckpt_path:
            logger.info("\n" + "="*60)
            logger.info(f"PHASE 4: GRPO Latest ({latest_ckpt})")
            gen_grpo_latest = load_csm_generator(device, checkpoint_path=latest_ckpt)
            results_latest = evaluate_model("GRPO Latest", gen_grpo_latest, context, device, eval_models, ref_embed)
            all_results.append(results_latest)
            del gen_grpo_latest
            torch.cuda.empty_cache()

    # Summary comparison table
    logger.info("\n" + "="*60)
    logger.info("COMPARISON TABLE")
    logger.info("="*60)
    logger.info(f"{'Model':<25} {'UTMOS':>8} {'CER':>8} {'SIM':>8} {'RTF':>8}")
    logger.info("-"*60)
    for r in all_results:
        if "avg_utmos" in r:
            logger.info(f"{r['name']:<25} {r['avg_utmos']:>8.3f} {r['avg_cer']:>8.4f} "
                         f"{r['avg_sim']:>8.3f} {r['avg_rtf']:>8.2f}")

    # Save results
    results_file = OUTPUT_DIR / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Decision report
    logger.info("\n" + "="*60)
    logger.info("DECISION REPORT")
    logger.info("="*60)

    if len(all_results) >= 2:
        best_model = max(all_results, key=lambda r: r.get("avg_utmos", 0))
        logger.info(f"Best quality model: {best_model['name']} (UTMOS={best_model.get('avg_utmos', 0):.3f})")
        logger.info(f"")
        logger.info(f"Note: Orpheus 3B benchmarks: UTMOS=4.345, RTF=0.64 on A10G")
        logger.info(f"If CSM UTMOS > 4.2 with RTF < 1.0, consider deploying CSM for quality.")
        logger.info(f"If CSM RTF > 1.0, Orpheus remains better for production (fast + good quality).")


if __name__ == "__main__":
    main()
