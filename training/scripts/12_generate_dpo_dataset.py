#!/usr/bin/env python3
"""
DPO Preference Dataset Generator for CSM-1B
=============================================
Generates N candidates per prompt, scores them with:
- UTMOS (perceptual quality, 1-5)
- ASR CER (Character Error Rate via faster-whisper)
- Speaker Similarity (cosine sim via resemblyzer)

Then constructs (chosen, rejected) pairs via Pareto ranking.

Usage:
    python 12_generate_dpo_dataset.py --prompts training/data/dpo_prompts.txt
    python 12_generate_dpo_dataset.py --num-candidates 6 --gpu 0
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
OUTPUT_DIR = PROJECT_ROOT / "training" / "dpo_dataset"


# ============================================================
# Default prompts for DPO training (conversational speech)
# ============================================================
DEFAULT_PROMPTS = [
    # Natural conversation
    "Oh wow, I didn't expect that at all!",
    "Hmm, let me think about that for a second.",
    "Yeah, I totally get what you mean.",
    "Wait, are you serious right now?",
    "That's absolutely amazing, I'm so happy for you!",
    "I'm really sorry to hear that happened.",
    "Honestly, that makes me a little nervous.",

    # Disfluencies / natural speech
    "So, like, the thing is... it's complicated.",
    "Well, um, I guess we could try that approach.",
    "Right, right, okay so basically what happened was...",

    # Short responses
    "Definitely!",
    "Oh no, really?",
    "That makes sense.",
    "I see what you mean.",

    # Longer conversational
    "Hey there! So I was thinking about what you said yesterday.",
    "You know what, let's just go for it and see what happens.",
    "I mean, it's not the worst idea I've ever heard, but...",
    "Oh my god, that's hilarious! Tell me more.",
    "I appreciate you sharing that with me, it means a lot.",
    "Okay, so here's the thing I keep going back and forth on.",

    # Questions
    "Wait, what did you just say?",
    "Do you really think that's going to work?",
    "How long has this been going on?",
    "Have you talked to anyone else about this?",

    # Emotional range
    "I'm so excited, I can barely contain myself!",
    "That's actually really disappointing to hear.",
    "I'm not gonna lie, that kind of hurt my feelings.",
    "This is the best news I've gotten all week!",

    # Maya-style responses (short, conversational)
    "Oh, interesting! Tell me more about that.",
    "Yeah, that's a really good point actually.",
    "Hmm, I'm not sure about that one.",
    "Absolutely, I completely agree.",
    "That's a tough situation, I'm sorry.",
    "Ha! That's actually pretty funny.",
    "Oh wow, I had no idea.",
    "Sure, that sounds like a plan!",
    "Well, it depends on how you look at it.",
    "I think you're onto something there.",
]


@dataclass
class ScoringConfig:
    num_candidates: int = 6
    temperature: float = 0.7
    topk: int = 80
    max_audio_ms: int = 10000
    min_cer_gap: float = 0.05    # Minimum CER gap for valid pair
    min_ssim_gap: float = 0.02   # Minimum SSIM gap for valid pair


class DPODataGenerator:
    """Generate and score preference pairs for DPO training."""

    def __init__(self, device='cuda:0', config=ScoringConfig()):
        self.device = device
        self.config = config
        self._generator = None
        self._utmos = None
        self._asr = None
        self._spk_encoder = None
        self._ref_embedding = None

    def load_models(self):
        """Load all scoring models."""
        # CSM Generator
        logger.info("Loading CSM-1B Generator...")
        from models import Model
        from generator import Generator

        model = Model.from_pretrained("sesame/csm-1b")
        model.to(device=self.device, dtype=torch.bfloat16)

        # Load best checkpoint
        ckpt_path = PROJECT_ROOT / "training/checkpoints/csm_maya_combined_naturalness/checkpoint-1500-merged/model_merged.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded checkpoint: {ckpt_path.name}")

        self._generator = Generator(model)
        logger.info("CSM Generator ready")

        # Voice context
        from generator import Segment
        vp_path = PROJECT_ROOT / "assets/voice_prompt/maya_voice_prompt.pt"
        if vp_path.exists():
            vp = torch.load(vp_path, map_location=self.device, weights_only=False)
            if isinstance(vp, dict) and "audio" in vp:
                audio = vp["audio"]
                if audio.dim() > 1:
                    audio = audio.squeeze(0)
                text = vp.get("text", "Hey, how's it going?")
                self._context = [Segment(speaker=0, text=text, audio=audio.to(self.device))]
                self._ref_audio = audio.cpu()
                logger.info(f"Voice context: {audio.shape[0]/24000:.1f}s")
        else:
            self._context = []
            self._ref_audio = None

        # UTMOS
        logger.info("Loading UTMOS...")
        self._utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        ).to(self.device).eval()
        logger.info("UTMOS ready")

        # ASR (openai whisper - more compatible with CUDA setup)
        logger.info("Loading ASR (whisper)...")
        import whisper
        self._asr = whisper.load_model("base.en", device=self.device)
        logger.info("ASR ready (whisper base.en for fast CER scoring)")

        # Speaker encoder (resemblyzer)
        logger.info("Loading speaker encoder...")
        from resemblyzer import VoiceEncoder
        self._spk_encoder = VoiceEncoder(device=self.device)

        # Compute reference speaker embedding
        if self._ref_audio is not None:
            ref_np = self._ref_audio.numpy()
            # Resample to 16kHz for resemblyzer
            ref_16k = torchaudio.functional.resample(
                self._ref_audio.unsqueeze(0), 24000, 16000
            ).squeeze(0).numpy()
            self._ref_embedding = self._spk_encoder.embed_utterance(ref_16k)
            logger.info("Reference speaker embedding computed")

        logger.info("All models loaded!")

    def generate_candidates(self, text: str) -> List[Dict]:
        """Generate N diverse candidates for a prompt."""
        candidates = []
        for i in range(self.config.num_candidates):
            start = time.time()
            audio = self._generator.generate(
                text=text,
                speaker=0,
                context=self._context,
                max_audio_length_ms=self.config.max_audio_ms,
                temperature=self.config.temperature,
                topk=self.config.topk,
            )
            gen_time = time.time() - start

            # Encode audio tokens (for DPO training later)
            # Must use inference_mode to avoid mimi VQ torch.cdist issue
            with torch.inference_mode():
                audio_tokens = self._generator._audio_tokenizer.encode(
                    audio.unsqueeze(0).unsqueeze(0).to(self.device)
                )  # (1, num_codebooks, T)

            candidates.append({
                "audio": audio.cpu(),
                "audio_tokens": audio_tokens.squeeze(0).cpu(),  # (32, T)
                "gen_time": gen_time,
                "duration": len(audio) / 24000,
            })

        return candidates

    def score_utmos(self, audio: torch.Tensor) -> float:
        """Score audio with UTMOS."""
        audio_16k = torchaudio.functional.resample(
            audio.unsqueeze(0), 24000, 16000
        ).to(self.device)
        with torch.no_grad():
            score = self._utmos(audio_16k, sr=16000)
        return score.item()

    def score_cer(self, audio: torch.Tensor, reference_text: str) -> float:
        """Score with ASR CER (Character Error Rate)."""
        import tempfile

        # Save to temp file for whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            torchaudio.save(f.name, audio.unsqueeze(0), 24000)
            temp_path = f.name

        try:
            result = self._asr.transcribe(temp_path, language="en",
                                          fp16=True)
            transcript = result["text"].strip()
        finally:
            os.unlink(temp_path)

        # Compute CER
        ref = reference_text.lower().strip()
        hyp = transcript.lower().strip()

        if not ref:
            return 0.0 if not hyp else 1.0

        # Simple edit distance based CER
        cer = _edit_distance(ref, hyp) / len(ref)
        return min(cer, 1.0)

    def score_speaker_similarity(self, audio: torch.Tensor) -> float:
        """Score speaker similarity to reference voice."""
        if self._ref_embedding is None:
            return 0.0

        audio_16k = torchaudio.functional.resample(
            audio.unsqueeze(0), 24000, 16000
        ).squeeze(0).numpy()

        gen_embedding = self._spk_encoder.embed_utterance(audio_16k)
        similarity = np.dot(self._ref_embedding, gen_embedding) / (
            np.linalg.norm(self._ref_embedding) * np.linalg.norm(gen_embedding) + 1e-8
        )
        return float(similarity)

    def score_candidates(self, candidates: List[Dict], text: str) -> List[Dict]:
        """Score all candidates."""
        for i, cand in enumerate(candidates):
            cand["utmos"] = self.score_utmos(cand["audio"])
            cand["cer"] = self.score_cer(cand["audio"], text)
            cand["ssim"] = self.score_speaker_similarity(cand["audio"])
            cand["text"] = text
            logger.debug(f"  Cand {i}: UTMOS={cand['utmos']:.3f} CER={cand['cer']:.3f} "
                        f"SSIM={cand['ssim']:.3f}")
        return candidates

    def select_pair(self, candidates: List[Dict]) -> Optional[Tuple[int, int]]:
        """
        Select (chosen_idx, rejected_idx) via Pareto ranking.

        Ranks on: lower CER is better, higher SSIM is better, higher UTMOS is better.
        """
        n = len(candidates)
        if n < 2:
            return None

        # Composite score for ranking
        scores = []
        for i, c in enumerate(candidates):
            # Weighted composite: intelligibility + similarity + quality
            composite = (
                (1.0 - c["cer"]) * 0.4 +    # Intelligibility (40%)
                c["ssim"] * 0.3 +             # Speaker similarity (30%)
                (c["utmos"] / 5.0) * 0.3      # Quality (30%)
            )
            scores.append((i, composite))

        scores.sort(key=lambda x: x[1], reverse=True)
        chosen_idx = scores[0][0]
        rejected_idx = scores[-1][0]

        # Validate: check minimum quality gap
        c = candidates[chosen_idx]
        r = candidates[rejected_idx]

        gap = abs(scores[0][1] - scores[-1][1])
        if gap < 0.02:
            return None  # Too similar, skip

        return (chosen_idx, rejected_idx)

    def generate_dataset(self, prompts: List[str]) -> List[Dict]:
        """Generate full preference dataset."""
        pairs = []
        skipped = 0

        for i, text in enumerate(prompts):
            logger.info(f"\n[{i+1}/{len(prompts)}] \"{text[:50]}...\"")

            # Generate candidates
            candidates = self.generate_candidates(text)

            # Score candidates
            candidates = self.score_candidates(candidates, text)

            # Select pair
            pair_indices = self.select_pair(candidates)

            if pair_indices is None:
                logger.info(f"  SKIPPED: candidates too similar")
                skipped += 1
                continue

            chosen_idx, rejected_idx = pair_indices
            c = candidates[chosen_idx]
            r = candidates[rejected_idx]

            pair = {
                "text": text,
                "chosen": {
                    "audio_tokens": c["audio_tokens"].tolist(),
                    "utmos": c["utmos"],
                    "cer": c["cer"],
                    "ssim": c["ssim"],
                    "duration": c["duration"],
                },
                "rejected": {
                    "audio_tokens": r["audio_tokens"].tolist(),
                    "utmos": r["utmos"],
                    "cer": r["cer"],
                    "ssim": r["ssim"],
                    "duration": r["duration"],
                },
                "num_candidates": len(candidates),
                "score_gap": abs(
                    (1-c["cer"])*0.4 + c["ssim"]*0.3 + c["utmos"]/5*0.3 -
                    ((1-r["cer"])*0.4 + r["ssim"]*0.3 + r["utmos"]/5*0.3)
                ),
            }
            pairs.append(pair)

            logger.info(f"  CHOSEN: UTMOS={c['utmos']:.3f} CER={c['cer']:.3f} SSIM={c['ssim']:.3f}")
            logger.info(f"  REJECTED: UTMOS={r['utmos']:.3f} CER={r['cer']:.3f} SSIM={r['ssim']:.3f}")
            logger.info(f"  Gap: {pair['score_gap']:.3f}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Generated {len(pairs)} pairs from {len(prompts)} prompts (skipped {skipped})")

        return pairs


def _edit_distance(ref: str, hyp: str) -> int:
    """Compute edit distance between two strings."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default=None,
                       help="File with prompts (one per line)")
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=None,
                       help="Limit number of prompts (for testing)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    config = ScoringConfig(num_candidates=args.num_candidates)

    # Load prompts
    if args.prompts and Path(args.prompts).exists():
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Candidates per prompt: {config.num_candidates}")
    logger.info(f"Device: {device}")

    # Generate dataset
    generator = DPODataGenerator(device=device, config=config)
    generator.load_models()

    pairs = generator.generate_dataset(prompts)

    # Save dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "preference_pairs.json"

    with open(output_file, "w") as f:
        json.dump({
            "num_pairs": len(pairs),
            "num_prompts": len(prompts),
            "config": {
                "num_candidates": config.num_candidates,
                "temperature": config.temperature,
                "topk": config.topk,
            },
            "pairs": pairs,
        }, f, indent=2)

    logger.info(f"\nDataset saved to: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1e6:.1f} MB")

    # Summary statistics
    if pairs:
        utmos_chosen = [p["chosen"]["utmos"] for p in pairs]
        utmos_rejected = [p["rejected"]["utmos"] for p in pairs]
        cer_chosen = [p["chosen"]["cer"] for p in pairs]
        cer_rejected = [p["rejected"]["cer"] for p in pairs]
        ssim_chosen = [p["chosen"]["ssim"] for p in pairs]
        ssim_rejected = [p["rejected"]["ssim"] for p in pairs]

        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Chosen  UTMOS: {np.mean(utmos_chosen):.3f} ± {np.std(utmos_chosen):.3f}")
        logger.info(f"  Rejected UTMOS: {np.mean(utmos_rejected):.3f} ± {np.std(utmos_rejected):.3f}")
        logger.info(f"  Chosen  CER: {np.mean(cer_chosen):.3f}")
        logger.info(f"  Rejected CER: {np.mean(cer_rejected):.3f}")
        logger.info(f"  Chosen  SSIM: {np.mean(ssim_chosen):.3f}")
        logger.info(f"  Rejected SSIM: {np.mean(ssim_rejected):.3f}")
        logger.info(f"  Avg Score Gap: {np.mean([p['score_gap'] for p in pairs]):.3f}")


if __name__ == "__main__":
    main()
