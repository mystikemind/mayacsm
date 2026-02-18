#!/usr/bin/env python3
"""
Fair Baseline Evaluation: Base Orpheus 3B via HF Transformers generate
======================================================================

The production UTMOS baseline (4.391) was measured via llama-server GGUF.
The training eval UTMOS (3.942) was measured via HF Transformers generate.
These are DIFFERENT inference paths and may produce different quality.

This script tests the BASE model (no LoRA) through the SAME HF Transformers
generate path used during training eval, giving us a fair comparison.

Fair comparison:
    Base (HF generate)  vs  Fine-tuned (HF generate)  = training eval gap
    Base (llama-server) vs  Fine-tuned (llama-server)  = production gap
"""

import os
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/.cache/huggingface"

import sys
_gpu = 3
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = int(sys.argv[i + 1])
        break
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torchaudio

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Token IDs (official Orpheus)
PAD_TOKEN = 128263
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MAX = 128266 + (7 * 4096) - 1

# Production inference params (Orpheus-FastAPI)
PRODUCTION_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# Same test prompts as training eval
TEST_PROMPTS = [
    "Yeah I'm doing pretty good, how's everything with you?",
    "Oh that's so cool, tell me more about that.",
    "Aww that sounds really tough, I'm sorry you're dealing with that.",
    "Oh my gosh that's hilarious, I can't believe that happened!",
    "Hmm yeah that makes sense, I've been thinking about that too.",
    "Honestly I think that's a great idea, you should definitely go for it.",
    "Yeah I know what you mean, sometimes things just don't work out.",
    "Oh really?",
    "Yeah for sure.",
    "Hmm that's interesting.",
    "Well you know what I think, let's just go for it!",
    "That reminds me of something that happened last week.",
    "Mhm.",
    "Wow that's amazing!",
    "Okay cool, thanks for letting me know.",
]


def load_snac(device):
    from snac import SNAC
    return SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)


def load_utmos(device):
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    return utmos.to(device).eval()


def decode_tokens(token_ids: list, snac, device) -> Optional[torch.Tensor]:
    n = (len(token_ids) // 7) * 7
    if n < 7:
        return None
    token_ids = token_ids[:n]

    l0, l1, l2 = [], [], []
    for i in range(n // 7):
        b = 7 * i
        l0.append(max(0, min(4095, token_ids[b] - AUDIO_TOKEN_BASE)))
        l1.append(max(0, min(4095, token_ids[b+1] - AUDIO_TOKEN_BASE - 4096)))
        l2.append(max(0, min(4095, token_ids[b+2] - AUDIO_TOKEN_BASE - 2*4096)))
        l2.append(max(0, min(4095, token_ids[b+3] - AUDIO_TOKEN_BASE - 3*4096)))
        l1.append(max(0, min(4095, token_ids[b+4] - AUDIO_TOKEN_BASE - 4*4096)))
        l2.append(max(0, min(4095, token_ids[b+5] - AUDIO_TOKEN_BASE - 5*4096)))
        l2.append(max(0, min(4095, token_ids[b+6] - AUDIO_TOKEN_BASE - 6*4096)))

    codes = [
        torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(device),
    ]
    with torch.inference_mode():
        audio = snac.decode(codes)
    return audio.squeeze().cpu()


def score_audio(audio: torch.Tensor, utmos_model, device) -> Optional[float]:
    if audio is None:
        return None
    try:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        resampled = torchaudio.functional.resample(audio, 24000, 16000).to(device)
        with torch.inference_mode():
            score = utmos_model(resampled, 16000)
        return float(score.item())
    except Exception as e:
        logger.warning(f"UTMOS scoring failed: {e}")
        return None


def evaluate_model(
    model, tokenizer, test_prompts: List[str],
    snac, utmos_model, device: str,
    voice_name: str = "tara",
    max_new_tokens: int = 500,
    n_runs: int = 1,
    save_audio_dir: Optional[str] = None,
) -> Dict:
    """Evaluate model with production parameters."""

    all_scores = []
    per_prompt = {}

    for run in range(n_runs):
        for i, prompt_text in enumerate(test_prompts):
            text = f"{voice_name}: {prompt_text}"
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            input_ids = [128000, START_OF_HUMAN] + text_ids + [END_OF_HUMAN, START_OF_AI]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.device)

            t0 = time.time()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=PRODUCTION_PARAMS["temperature"],
                    top_p=PRODUCTION_PARAMS["top_p"],
                    repetition_penalty=PRODUCTION_PARAMS["repetition_penalty"],
                    do_sample=True,
                    pad_token_id=PAD_TOKEN,
                    eos_token_id=END_OF_SPEECH,
                )
            gen_time = time.time() - t0

            generated = output[0][len(input_ids):].tolist()
            audio_tokens = []
            for t in generated:
                if t == END_OF_SPEECH:
                    break
                if AUDIO_TOKEN_BASE <= t <= AUDIO_TOKEN_MAX:
                    audio_tokens.append(t)

            score = None
            if len(audio_tokens) >= 7:
                audio = decode_tokens(audio_tokens, snac, device)
                if audio is not None and audio.numel() > 0:
                    score = score_audio(audio, utmos_model, device)

                    if save_audio_dir and score is not None:
                        os.makedirs(save_audio_dir, exist_ok=True)
                        fname = f"{voice_name}_r{run}_p{i:02d}_{score:.3f}.wav"
                        torchaudio.save(
                            os.path.join(save_audio_dir, fname),
                            audio.unsqueeze(0), 24000
                        )

            status = f"UTMOS={score:.3f}" if score else "FAILED"
            logger.info(
                f"  [{run+1}/{n_runs}] [{i+1:2d}/{len(test_prompts)}] "
                f"{prompt_text[:40]:40s} → {len(audio_tokens):4d} tokens, "
                f"{gen_time:.1f}s, {status}"
            )

            if score is not None:
                all_scores.append(score)
                key = f"p{i:02d}"
                per_prompt.setdefault(key, []).append(score)

    if not all_scores:
        return {"utmos_mean": 0.0, "n_evaluated": 0, "n_total": len(test_prompts) * n_runs}

    return {
        "utmos_mean": float(np.mean(all_scores)),
        "utmos_std": float(np.std(all_scores)),
        "utmos_min": float(min(all_scores)),
        "utmos_max": float(max(all_scores)),
        "utmos_median": float(np.median(all_scores)),
        "n_evaluated": len(all_scores),
        "n_total": len(test_prompts) * n_runs,
        "per_prompt_means": {k: float(np.mean(v)) for k, v in per_prompt.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Fair baseline evaluation")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per prompt (for variance estimation)")
    parser.add_argument("--voice", default="tara", help="Voice name for base model (tara=Orpheus default)")
    parser.add_argument("--finetuned-voice", default="maya", help="Voice name for fine-tuned model")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned LoRA checkpoint to compare")
    parser.add_argument("--save-audio", action="store_true", help="Save generated audio files")
    args = parser.parse_args()

    device = "cuda:0"
    output_dir = Path("/home/ec2-user/SageMaker/project_maya/training/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = str(output_dir / "audio") if args.save_audio else None

    logger.info("=" * 80)
    logger.info("  FAIR BASELINE EVALUATION")
    logger.info("  Testing BASE model via HF Transformers generate")
    logger.info("  Same pipeline as training eval for apples-to-apples comparison")
    logger.info(f"  Production params: temp={PRODUCTION_PARAMS['temperature']}, "
                f"top_p={PRODUCTION_PARAMS['top_p']}, rep={PRODUCTION_PARAMS['repetition_penalty']}")
    logger.info(f"  Runs per prompt: {args.n_runs}")
    logger.info("=" * 80)

    # Load evaluation tools
    logger.info("\nLoading SNAC + UTMOS...")
    snac = load_snac(device)
    utmos_model = load_utmos(device)

    # Load base model
    logger.info("\nLoading base Orpheus 3B (BF16)...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = PAD_TOKEN
    logger.info(f"Base model loaded in {time.time()-t0:.1f}s ({torch.cuda.memory_allocated()/1e9:.1f}GB)")

    results = {}

    # Test 1: Base model with default voice (tara)
    logger.info(f"\n{'='*80}")
    logger.info(f"  TEST 1: Base model, voice='{args.voice}' (Orpheus default)")
    logger.info(f"{'='*80}")
    base_result = evaluate_model(
        base_model, tokenizer, TEST_PROMPTS,
        snac, utmos_model, device,
        voice_name=args.voice, n_runs=args.n_runs,
        save_audio_dir=os.path.join(audio_dir, f"base_{args.voice}") if audio_dir else None,
    )
    results[f"base_{args.voice}"] = base_result
    logger.info(f"\n  BASE ({args.voice}): UTMOS = {base_result['utmos_mean']:.3f} +/- {base_result['utmos_std']:.3f}")
    logger.info(f"  Range: {base_result.get('utmos_min', 0):.3f} - {base_result.get('utmos_max', 0):.3f}")
    logger.info(f"  Median: {base_result.get('utmos_median', 0):.3f}")
    logger.info(f"  Evaluated: {base_result['n_evaluated']}/{base_result['n_total']}")

    # Test 2: Base model with "maya" voice (to see how unknown voice performs)
    if args.voice != args.finetuned_voice:
        logger.info(f"\n{'='*80}")
        logger.info(f"  TEST 2: Base model, voice='{args.finetuned_voice}' (unknown to base)")
        logger.info(f"{'='*80}")
        base_maya_result = evaluate_model(
            base_model, tokenizer, TEST_PROMPTS,
            snac, utmos_model, device,
            voice_name=args.finetuned_voice, n_runs=args.n_runs,
            save_audio_dir=os.path.join(audio_dir, f"base_{args.finetuned_voice}") if audio_dir else None,
        )
        results[f"base_{args.finetuned_voice}"] = base_maya_result
        logger.info(f"\n  BASE ({args.finetuned_voice}): UTMOS = {base_maya_result['utmos_mean']:.3f} +/- {base_maya_result['utmos_std']:.3f}")

    # Test 3: Fine-tuned model (if checkpoint provided)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            logger.info(f"\n{'='*80}")
            logger.info(f"  TEST 3: Fine-tuned model ({checkpoint_path.name}), voice='{args.finetuned_voice}'")
            logger.info(f"{'='*80}")

            logger.info("Loading LoRA adapter...")
            ft_model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
            ft_model.eval()

            ft_result = evaluate_model(
                ft_model, tokenizer, TEST_PROMPTS,
                snac, utmos_model, device,
                voice_name=args.finetuned_voice, n_runs=args.n_runs,
                save_audio_dir=os.path.join(audio_dir, f"finetuned_{args.finetuned_voice}") if audio_dir else None,
            )
            results[f"finetuned_{args.finetuned_voice}"] = ft_result
            logger.info(f"\n  FINETUNED ({args.finetuned_voice}): UTMOS = {ft_result['utmos_mean']:.3f} +/- {ft_result['utmos_std']:.3f}")

            # Also test fine-tuned model with base voice name
            logger.info(f"\n{'='*80}")
            logger.info(f"  TEST 4: Fine-tuned model, voice='{args.voice}' (base voice on ft model)")
            logger.info(f"{'='*80}")
            ft_base_voice = evaluate_model(
                ft_model, tokenizer, TEST_PROMPTS,
                snac, utmos_model, device,
                voice_name=args.voice, n_runs=args.n_runs,
                save_audio_dir=os.path.join(audio_dir, f"finetuned_{args.voice}") if audio_dir else None,
            )
            results[f"finetuned_{args.voice}"] = ft_base_voice
            logger.info(f"\n  FINETUNED ({args.voice}): UTMOS = {ft_base_voice['utmos_mean']:.3f} +/- {ft_base_voice['utmos_std']:.3f}")

            del ft_model
            torch.cuda.empty_cache()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("  FAIR COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Inference: HF Transformers generate (BF16)")
    logger.info(f"  Params: temp={PRODUCTION_PARAMS['temperature']}, top_p={PRODUCTION_PARAMS['top_p']}, rep={PRODUCTION_PARAMS['repetition_penalty']}")
    logger.info(f"  Runs per prompt: {args.n_runs} ({args.n_runs * len(TEST_PROMPTS)} total samples)")
    logger.info("-" * 80)

    for name, result in results.items():
        utmos = result["utmos_mean"]
        std = result.get("utmos_std", 0)
        n = result["n_evaluated"]
        logger.info(f"  {name:30s}: UTMOS = {utmos:.3f} +/- {std:.3f}  (n={n})")

    logger.info("-" * 80)
    logger.info(f"  Production baseline (llama-server GGUF): 4.391")
    logger.info(f"  Training v3 eval (step 50):              3.942")
    logger.info("=" * 80)

    # Save results
    output_path = output_dir / "fair_baseline_results.json"
    results["metadata"] = {
        "inference": "HF Transformers generate (BF16)",
        "params": PRODUCTION_PARAMS,
        "n_runs": args.n_runs,
        "n_prompts": len(TEST_PROMPTS),
        "production_baseline_gguf": 4.391,
        "training_v3_step50": 3.942,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    del base_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
