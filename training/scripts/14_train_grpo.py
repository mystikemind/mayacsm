#!/usr/bin/env python3
"""
GRPO Training for CSM-1B
==========================
Group Relative Policy Optimization for improving speech naturalness.

Based on:
- DeepSeek GRPO (arXiv 2402.03300): Core algorithm
- Align2Speak (arXiv 2509.21718): Simplified GRPO for TTS, GRPO > DPO
- Multi-Reward GRPO (arXiv 2511.21270): Multi-reward with entropy regularization
- "No Verifiable Reward for Prosody" (arXiv 2509.18531): UTMOS needed to prevent prosody collapse

Algorithm:
  1. For each prompt, generate G candidates from current policy
  2. Score each candidate: UTMOS (quality) + CER (intelligibility) + Speaker Sim
  3. Compute group-relative advantages: A_i = (R_i - mean) / (std + eps)
  4. Update policy: L = -(1/G) * sum_i A_i * log_pi(y_i|x)

Usage:
    python 14_train_grpo.py --gpu 0 --group-size 8 --steps 1000
    python 14_train_grpo.py --gpu 0 --mode both --steps 1500  # Train backbone + decoder
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")


@dataclass
class GRPOConfig:
    # Model
    base_model: str = "sesame/csm-1b"
    sft_checkpoint: str = str(
        PROJECT_ROOT / "training/checkpoints/csm_maya_combined_naturalness/checkpoint-5000"
    )  # ckpt-5000 is optimal: UTMOS 4.031±0.202 (evaluated all ckpts 1500-13500)
    voice_prompt: str = str(PROJECT_ROOT / "assets/voice_prompt/maya_voice_prompt.pt")

    # GRPO
    group_size: int = 8          # G: candidates per prompt
    epsilon: float = 0.2         # Clip range (for multi-update)
    beta: float = 0.0            # KL penalty (0 = simplified GRPO, Align2Speak style)
    advantage_eps: float = 1e-8  # Numerical stability for std normalization

    # Reward weights (additive combination)
    # "No Verifiable Reward for Prosody" warns: CER alone collapses prosody
    # So we weight UTMOS highly to preserve naturalness
    w_utmos: float = 0.4         # Quality + prosody + naturalness
    w_cer: float = 0.4           # Intelligibility (prevents gibberish)
    w_sim: float = 0.2           # Speaker consistency
    cer_alpha: float = 3.0       # Sensitivity: R_cer = 1 - tanh(alpha * CER)

    # Training mode: 'decoder', 'backbone', 'both'
    mode: str = 'decoder'

    # Training
    lr: float = 1e-6
    min_lr: float = 2e-7
    weight_decay: float = 0.01
    max_steps: int = 1500
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    grad_accum: int = 4          # Accumulate over 4 prompts before update (more stable)

    # Generation
    temperature: float = 0.8
    topk: int = 50
    max_audio_length_ms: int = 3000  # Short for training efficiency

    # Monitoring
    log_every: int = 10
    save_every: int = 100         # More frequent saves for evaluation
    eval_every: int = 100
    max_frames_logprob: int = 50  # Limit frames for log prob computation

    # Output
    output_dir: str = str(PROJECT_ROOT / "training/checkpoints/csm_maya_grpo")

    # Prompts
    prompts_file: str = ""  # Optional file with text prompts (one per line)


# ============================================================
# Default prompts for GRPO training (conversational speech)
# ============================================================
DEFAULT_PROMPTS = [
    # Natural conversation
    "Oh wow, I didn't expect that at all!",
    "Hmm, let me think about that for a second.",
    "Yeah, I totally get what you mean.",
    "Wait, are you serious right now?",
    "That's absolutely amazing, I'm so happy for you!",
    "I don't know, it just feels different somehow.",
    "Okay so here's what I was thinking.",
    "Right, that makes a lot of sense actually.",
    "No way, you're kidding me!",
    "So basically what happened was really unexpected.",

    # Emotional range
    "I'm sorry to hear that, that must be really tough.",
    "Oh my gosh, this is the best news ever!",
    "Hmm, I'm not entirely sure about that.",
    "That's really frustrating, I totally understand.",
    "Wow, I never thought of it that way before.",
    "To be honest, I was a little nervous about it.",
    "Ha, yeah that's pretty funny when you think about it.",
    "I really appreciate you sharing that with me.",
    "Okay wait, let me start over from the beginning.",
    "You know what, I think you might be right.",

    # Conversational fillers and flow
    "Well, the thing is, it's kind of complicated.",
    "I mean, at the end of the day, it worked out.",
    "So yeah, that's basically the whole story.",
    "Actually, now that I think about it more carefully.",
    "But honestly, what really matters is how you feel.",
    "Listen, I know this might sound a bit weird.",
    "Alright, let me try to explain this differently.",
    "Oh interesting, I hadn't considered that angle.",
    "You know, sometimes these things just happen.",
    "Look, I just want to make sure we're on the same page.",

    # Questions and engagement
    "What do you think about all of this?",
    "Does that make sense to you?",
    "How are you feeling about everything?",
    "Can you tell me more about what happened?",
    "What would you do in that situation?",
    "Really? And then what happened next?",
    "So what are you planning to do about it?",
    "Do you want my honest opinion on this?",
]

# Held-out evaluation sentences (verified NOT in grpo_prompts.txt or DEFAULT_PROMPTS)
EVAL_SENTENCES = [
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
]


def _load_checkpoint_into_model(model, checkpoint_path, device):
    """Load checkpoint (supports both merged .pt files and directory with separate components)."""
    ckpt = Path(checkpoint_path)

    if ckpt.is_file() and ckpt.suffix == '.pt':
        # Single merged file
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded merged checkpoint: {ckpt}")
    elif ckpt.is_dir():
        if (ckpt / 'model_merged.pt').exists():
            state = torch.load(str(ckpt / 'model_merged.pt'), map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded merged checkpoint from dir: {ckpt}")
        elif (ckpt / 'decoder.pt').exists():
            # Separate component files (decoder-only SFT) - preserves base backbone exactly
            decoder_state = torch.load(str(ckpt / 'decoder.pt'), map_location=device, weights_only=True)
            model.decoder.load_state_dict(decoder_state)
            proj_state = torch.load(str(ckpt / 'projection.pt'), map_location=device, weights_only=True)
            model.projection.load_state_dict(proj_state)
            head_data = torch.load(str(ckpt / 'audio_head.pt'), map_location=device, weights_only=True)
            model.audio_head.data = head_data.to(device)
            logger.info(f"Loaded component checkpoint (decoder+proj+head): {ckpt}")
        else:
            raise FileNotFoundError(f"No recognized checkpoint files in {ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")


def load_csm_model(config: GRPOConfig, device: str):
    """Load CSM model with SFT checkpoint."""
    from models import Model

    model = Model.from_pretrained(config.base_model)
    model.to(device=device, dtype=torch.bfloat16)

    if config.sft_checkpoint and Path(config.sft_checkpoint).exists():
        _load_checkpoint_into_model(model, config.sft_checkpoint, device)

    return model


def load_generator(config: GRPOConfig, device: str):
    """Load CSM Generator for audio generation."""
    from generator import Generator, load_csm_1b
    import torchaudio

    generator = load_csm_1b(device=device)

    # Load SFT weights into generator's model
    if config.sft_checkpoint and Path(config.sft_checkpoint).exists():
        _load_checkpoint_into_model(generator._model, config.sft_checkpoint, device)
        logger.info("Loaded SFT weights into Generator")

    return generator


def load_voice_context(config: GRPOConfig, device: str):
    """Load voice prompt for speaker context."""
    from generator import Segment
    import torchaudio

    voice_data = torch.load(config.voice_prompt, map_location=device, weights_only=True)
    if isinstance(voice_data, dict):
        context_audio = voice_data["audio"].to(device)
        context_text = voice_data.get("text", "Hey, how's it going? I'm Maya, nice to meet you!")
    else:
        context_audio = voice_data.to(device)
        context_text = "Hey, how's it going? I'm Maya, nice to meet you!"

    if context_audio.dim() == 2:
        context_audio = context_audio[0]

    context = [Segment(
        text=context_text,
        speaker=0,
        audio=context_audio
    )]

    duration = len(context_audio) / 24000
    logger.info(f"Voice context: {duration:.1f}s")
    return context


def load_reward_models(device: str):
    """Load UTMOS, Whisper (CER), and speaker encoder."""
    rewards = {}

    # UTMOS
    logger.info("Loading UTMOS...")
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos = utmos.to(device).eval()
    rewards["utmos"] = utmos
    logger.info("UTMOS ready")

    # Whisper for CER
    logger.info("Loading Whisper (base.en for fast CER)...")
    import whisper
    whisper_model = whisper.load_model("base.en", device=device)
    rewards["whisper"] = whisper_model
    logger.info("Whisper ready")

    # Speaker encoder
    logger.info("Loading speaker encoder...")
    from resemblyzer import VoiceEncoder
    encoder = VoiceEncoder(device=device)
    rewards["speaker_encoder"] = encoder
    logger.info("Speaker encoder ready")

    return rewards


def score_utmos(audio: torch.Tensor, utmos_model, device: str) -> float:
    """Score audio quality with UTMOS (1-5 scale)."""
    with torch.no_grad():
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(device).float()
        # UTMOS expects 16kHz
        if audio.shape[-1] > 16000:  # Assume 24kHz, resample
            import torchaudio
            audio = torchaudio.functional.resample(audio, 24000, 16000)
        score = utmos_model(audio, sr=16000)
        return score.item()


def score_cer(audio: torch.Tensor, text: str, whisper_model, device: str) -> float:
    """Compute Character Error Rate using Whisper."""
    import tempfile, torchaudio

    audio_np = audio.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np[0]

    # Whisper expects float32 numpy at 16kHz
    audio_16k = torchaudio.functional.resample(
        torch.from_numpy(audio_np).unsqueeze(0), 24000, 16000
    ).squeeze(0).numpy()

    result = whisper_model.transcribe(audio_16k, language="en")
    transcript = result["text"].strip()

    # Compute CER
    ref = text.lower().strip()
    hyp = transcript.lower().strip()
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0

    # Simple edit distance CER
    import difflib
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    cer = 1.0 - matcher.ratio()
    return cer


def score_speaker_similarity(
    audio: torch.Tensor,
    ref_embedding: np.ndarray,
    speaker_encoder,
) -> float:
    """Compute speaker similarity via cosine similarity."""
    audio_np = audio.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np[0]

    # Resemblyzer expects 16kHz
    import torchaudio
    audio_16k = torchaudio.functional.resample(
        torch.from_numpy(audio_np).unsqueeze(0), 24000, 16000
    ).squeeze(0).numpy()

    from resemblyzer import preprocess_wav
    wav = preprocess_wav(audio_16k, source_sr=16000)
    if len(wav) < 1600:  # Too short
        return 0.0

    embedding = speaker_encoder.embed_utterance(wav)
    similarity = np.dot(embedding, ref_embedding) / (
        np.linalg.norm(embedding) * np.linalg.norm(ref_embedding) + 1e-8
    )
    return float(similarity)


def compute_composite_reward(
    utmos_score: float,
    cer: float,
    sim: float,
    config: GRPOConfig,
) -> float:
    """Compute composite reward from individual scores."""
    # R_cer = 1 - tanh(alpha * CER)  [1=perfect, 0=terrible]
    r_cer = 1.0 - np.tanh(config.cer_alpha * cer)

    # R_utmos = score / 5.0  [0-1 range]
    r_utmos = min(utmos_score / 5.0, 1.0)

    # R_sim = cosine similarity  [already 0-1]
    r_sim = max(sim, 0.0)

    # Weighted sum
    reward = config.w_utmos * r_utmos + config.w_cer * r_cer + config.w_sim * r_sim
    return reward


def compute_sequence_logprob_efficient(
    model: nn.Module,
    text_tokens: torch.Tensor,    # (T_text, 33)
    text_mask: torch.Tensor,      # (T_text, 33)
    audio_tokens: torch.Tensor,   # (T_frames, 32)
    device: str = 'cuda',
    max_frames: int = 50,
    mode: str = 'decoder',        # 'decoder', 'backbone', 'both'
) -> torch.Tensor:
    """
    Compute log P(audio | text) efficiently without KV caches.

    For backbone: computes log P(cb0) at each frame
    For decoder: computes log P(cb1..31 | backbone_h, cb0) at each frame
    For both: sum of backbone + decoder log probs

    Uses full-sequence forward passes (no KV cache inplace issues).
    Returns scalar tensor with grad.
    """
    from models import _index_causal_mask

    dtype = next(model.parameters()).dtype
    num_codebooks = model.config.audio_num_codebooks  # 32
    T = min(audio_tokens.size(0), max_frames)

    # ========================================
    # Step 1: Full backbone forward pass
    # ========================================
    # Build full input: [text_tokens, audio_frame_0, ..., audio_frame_{T-1}]
    all_tokens_list = [text_tokens.unsqueeze(0)]  # (1, T_text, 33)
    all_masks_list = [text_mask.unsqueeze(0)]

    for t in range(T):
        frame = audio_tokens[t]
        frame_token = torch.zeros(1, 1, num_codebooks + 1, device=device, dtype=frame.dtype)
        frame_token[0, 0, :num_codebooks] = frame
        frame_mask = torch.zeros(1, 1, num_codebooks + 1, device=device, dtype=torch.bool)
        frame_mask[0, 0, :num_codebooks] = True
        all_tokens_list.append(frame_token)
        all_masks_list.append(frame_mask)

    all_tokens = torch.cat(all_tokens_list, dim=1)  # (1, T_text + T, 33)
    all_masks = torch.cat(all_masks_list, dim=1)
    S = all_tokens.shape[1]

    # Backbone forward
    embeds = model._embed_tokens(all_tokens)
    masked_embeds = embeds * all_masks.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)

    # Causal mask for backbone
    causal_mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool)).unsqueeze(0)

    if mode in ('backbone', 'both'):
        # Need gradients through backbone
        backbone_out = model.backbone(h, mask=causal_mask).to(dtype=dtype)
    else:
        # Decoder-only: backbone is frozen
        with torch.no_grad():
            backbone_out = model.backbone(h, mask=causal_mask).to(dtype=dtype)

    # backbone_out: (1, S, hidden_dim)
    T_text = text_tokens.shape[0]

    total_logprob = torch.tensor(0.0, device=device, dtype=torch.float32)

    # ========================================
    # Step 2: Backbone log probs (codebook 0)
    # ========================================
    if mode in ('backbone', 'both'):
        for t in range(T):
            # Hidden state at position before frame t
            pos = T_text + t - 1 if t > 0 else T_text - 1
            h_t = backbone_out[:, pos, :]  # (1, hidden_dim)

            # Codebook 0 logits via codebook0_head
            cb0_logits = F.linear(h_t, model.codebook0_head)  # (1, vocab_size)
            cb0_logprob = F.log_softmax(cb0_logits.float(), dim=-1)
            cb0_target = audio_tokens[t, 0].long()
            total_logprob = total_logprob + cb0_logprob[0, cb0_target]

    # ========================================
    # Step 3: Decoder log probs (codebooks 1-31)
    # ========================================
    if mode in ('decoder', 'both'):
        for t in range(T):
            # Backbone hidden for this frame
            pos = T_text + t - 1 if t > 0 else T_text - 1
            h_t = backbone_out[:, pos:pos+1, :]  # (1, 1, hidden_dim)

            frame = audio_tokens[t]  # (32,)

            # Build decoder input: [backbone_h, embed(cb0), ..., embed(cb30)]
            cb_embeds = []
            for cb in range(num_codebooks - 1):  # cb0 to cb30
                cb_embed = model._embed_audio(cb, frame[cb].long().unsqueeze(0).unsqueeze(0))
                cb_embeds.append(cb_embed)

            cb_embeds_cat = torch.cat(cb_embeds, dim=1)  # (1, 31, dim)
            decoder_input = torch.cat([h_t, cb_embeds_cat], dim=1)  # (1, 32, dim)

            # Project and forward decoder
            projected = model.projection(decoder_input)
            dec_mask = torch.tril(
                torch.ones(32, 32, device=device, dtype=torch.bool)
            ).unsqueeze(0)

            decoder_out = model.decoder(projected, mask=dec_mask).to(dtype=dtype)

            # Log probs for each codebook 1-31
            for cb in range(1, num_codebooks):
                ci_logits = torch.mm(
                    decoder_out[:, cb, :],
                    model.audio_head[cb - 1].to(dtype)
                )
                ci_logprob = F.log_softmax(ci_logits.float(), dim=-1)
                ci_target = frame[cb].long()
                total_logprob = total_logprob + ci_logprob[0, ci_target]

    return total_logprob / T  # Average per frame


class GRPOTrainer:
    """GRPO Trainer for CSM-1B TTS."""

    def __init__(self, config: GRPOConfig, device: str):
        self.config = config
        self.device = device

        # Load generator for candidate generation (has KV caches for inference)
        logger.info("Loading CSM Generator...")
        self.generator = load_generator(config, device)
        logger.info("Generator ready")

        # Load SEPARATE training model WITHOUT KV caches (for gradient computation)
        logger.info("Loading training model (no KV caches)...")
        self.policy = load_csm_model(config, device)
        # Do NOT call setup_caches on self.policy - this is the key!
        logger.info("Training model ready")

        # Load voice context
        self.context = load_voice_context(config, device)

        # Setup trainable parameters based on mode
        self._setup_trainable_params()

        # Load reward models
        logger.info("Loading reward models...")
        self.rewards = load_reward_models(device)

        # Compute reference speaker embedding
        ref_audio = self.context[0].audio.cpu().numpy()
        import torchaudio
        ref_16k = torchaudio.functional.resample(
            torch.from_numpy(ref_audio).unsqueeze(0), 24000, 16000
        ).squeeze(0).numpy()
        from resemblyzer import preprocess_wav
        ref_wav = preprocess_wav(ref_16k, source_sr=16000)
        self.ref_speaker_embedding = self.rewards["speaker_encoder"].embed_utterance(ref_wav)
        logger.info("Reference speaker embedding computed")

        # Setup tokenizer for text encoding
        self._setup_tokenizer()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # LR scheduler: warmup + cosine decay to min_lr
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
            min_ratio = config.min_lr / config.lr
            return min_ratio + (1 - min_ratio) * (1 + np.cos(np.pi * progress)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Metrics
        self.step = 0
        self.metrics_history = []
        self.best_reward = -float('inf')

        # Prompts
        if config.prompts_file and Path(config.prompts_file).exists():
            with open(config.prompts_file) as f:
                self.prompts = [
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith('#')
                ]
            logger.info(f"Loaded {len(self.prompts)} prompts from {config.prompts_file}")
        else:
            self.prompts = DEFAULT_PROMPTS
            logger.info(f"Using {len(self.prompts)} default prompts")

        logger.info("All models loaded!")

    def _setup_trainable_params(self):
        """Configure which parameters are trainable based on mode."""
        mode = self.config.mode

        # Freeze everything first
        for p in self.policy.parameters():
            p.requires_grad = False

        self.trainable_params = []

        if mode in ('decoder', 'both'):
            for p in self.policy.decoder.parameters():
                p.requires_grad = True
                self.trainable_params.append(p)
            for p in self.policy.projection.parameters():
                p.requires_grad = True
                self.trainable_params.append(p)
            self.policy.audio_head.requires_grad = True
            self.trainable_params.append(self.policy.audio_head)

        if mode in ('backbone', 'both'):
            for p in self.policy.backbone.parameters():
                p.requires_grad = True
                self.trainable_params.append(p)
            for p in self.policy.text_embeddings.parameters():
                p.requires_grad = True
                self.trainable_params.append(p)
            for p in self.policy.audio_embeddings.parameters():
                p.requires_grad = True
                self.trainable_params.append(p)
            self.policy.codebook0_head.requires_grad = True
            self.trainable_params.append(self.policy.codebook0_head)

        n_trainable = sum(p.numel() for p in self.trainable_params)
        n_total = sum(p.numel() for p in self.policy.parameters())
        logger.info(f"Mode: {mode}")
        logger.info(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    def _setup_tokenizer(self):
        """Setup text tokenizer for log prob computation."""
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing

        self._tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        bos = self._tokenizer.bos_token
        eos = self._tokenizer.eos_token
        self._tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[
                (f"{bos}", self._tokenizer.bos_token_id),
                (f"{eos}", self._tokenizer.eos_token_id),
            ],
        )

    def _tokenize_text(self, text: str):
        """Tokenize text for backbone input."""
        text_ids = self._tokenizer.encode(text)
        num_codebooks = self.policy.config.audio_num_codebooks

        T = len(text_ids)
        tokens = torch.zeros(T, num_codebooks + 1, device=self.device, dtype=torch.long)
        mask = torch.zeros(T, num_codebooks + 1, device=self.device, dtype=torch.bool)

        for i, tid in enumerate(text_ids):
            tokens[i, -1] = tid
            mask[i, -1] = True

        return tokens, mask

    def _sync_weights_to_generator(self):
        """Copy training model weights to generator model."""
        # strict=False because generator model has extra cache buffers from setup_caches
        self.generator._model.load_state_dict(self.policy.state_dict(), strict=False)

    @torch.no_grad()
    def generate_candidates(self, text: str) -> List[Dict]:
        """Generate G candidates for a text prompt using the generator."""
        from generator import Segment

        candidates = []

        for g in range(self.config.group_size):
            t0 = time.time()

            # Generate audio
            audio = self.generator.generate(
                text=text,
                speaker=0,
                context=self.context,
                max_audio_length_ms=self.config.max_audio_length_ms,
                temperature=self.config.temperature,
                topk=self.config.topk,
            )

            gen_time = time.time() - t0

            # Encode audio to tokens using mimi codec
            with torch.inference_mode():
                audio_tokens = self.generator._audio_tokenizer.encode(
                    audio.unsqueeze(0).unsqueeze(0).to(self.device)
                )
            # audio_tokens: (1, 32, T) -> squeeze to (32, T)
            audio_tokens = audio_tokens.squeeze(0)

            candidates.append({
                "audio": audio.cpu(),
                "audio_tokens": audio_tokens.cpu(),
                "gen_time": gen_time,
                "duration": len(audio) / 24000,
            })

        return candidates

    def score_candidates(self, candidates: List[Dict], text: str) -> List[Dict]:
        """Score all candidates with reward models."""
        for cand in candidates:
            audio = cand["audio"]

            # UTMOS
            utmos = score_utmos(audio, self.rewards["utmos"], self.device)

            # CER
            cer = score_cer(audio, text, self.rewards["whisper"], self.device)

            # Speaker similarity
            sim = score_speaker_similarity(
                audio, self.ref_speaker_embedding, self.rewards["speaker_encoder"]
            )

            # Composite reward
            reward = compute_composite_reward(utmos, cer, sim, self.config)

            cand["utmos"] = utmos
            cand["cer"] = cer
            cand["sim"] = sim
            cand["reward"] = reward

        return candidates

    def compute_advantages(self, candidates: List[Dict]) -> List[float]:
        """Compute group-relative advantages."""
        rewards = [c["reward"] for c in candidates]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + self.config.advantage_eps

        advantages = [(r - mean_r) / std_r for r in rewards]
        return advantages

    def grpo_step(self, text: str, candidates: List[Dict], advantages: List[float]) -> Dict:
        """
        Compute GRPO loss and do gradient update.

        Loss = -(1/G) * sum_i A_i * log_pi(y_i | x)

        This is the simplified GRPO (REINFORCE with group-relative advantages),
        which equals full GRPO when ratio=1.0 (fresh generation from current policy).
        """
        self.policy.train()

        text_tokens, text_mask = self._tokenize_text(text)

        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        G = len(candidates)

        for i, (cand, adv) in enumerate(zip(candidates, advantages)):
            audio_tokens = cand["audio_tokens"].to(self.device).T  # (T, 32)

            # Compute log prob of this candidate under current policy
            logprob = compute_sequence_logprob_efficient(
                self.policy,
                text_tokens, text_mask,
                audio_tokens,
                self.device,
                max_frames=self.config.max_frames_logprob,
                mode=self.config.mode,
            )

            # GRPO loss: -A_i * log_pi(y_i | x) / G
            loss_i = -adv * logprob / G
            total_loss = total_loss + loss_i

        # Backward
        total_loss.backward()

        metrics = {
            "loss": total_loss.item(),
            "mean_reward": np.mean([c["reward"] for c in candidates]),
            "std_reward": np.std([c["reward"] for c in candidates]),
            "mean_utmos": np.mean([c["utmos"] for c in candidates]),
            "mean_cer": np.mean([c["cer"] for c in candidates]),
            "mean_sim": np.mean([c["sim"] for c in candidates]),
            "best_utmos": max(c["utmos"] for c in candidates),
            "worst_utmos": min(c["utmos"] for c in candidates),
        }

        return metrics

    @torch.no_grad()
    def evaluate_held_out(self) -> Dict:
        """Evaluate on held-out sentences to check generalization."""
        import torchaudio
        from resemblyzer import preprocess_wav

        utmos_scores, cer_scores, sim_scores = [], [], []

        for text in EVAL_SENTENCES:
            audio = self.generator.generate(
                text=text, speaker=0, context=self.context,
                max_audio_length_ms=5000, temperature=0.8, topk=50,
            )

            # UTMOS
            utmos = score_utmos(audio, self.rewards["utmos"], self.device)
            utmos_scores.append(utmos)

            # CER
            cer = score_cer(audio, text, self.rewards["whisper"], self.device)
            cer_scores.append(cer)

            # Speaker similarity
            sim = score_speaker_similarity(
                audio, self.ref_speaker_embedding, self.rewards["speaker_encoder"]
            )
            sim_scores.append(sim)

        return {
            "eval_utmos": float(np.mean(utmos_scores)),
            "eval_cer": float(np.mean(cer_scores)),
            "eval_sim": float(np.mean(sim_scores)),
            "eval_utmos_std": float(np.std(utmos_scores)),
        }

    def save_checkpoint(self, name: str = None):
        """Save model checkpoint in non-merged format (decoder/projection/audio_head)."""
        if name is None:
            name = f"step-{self.step}"
        out_dir = Path(self.config.output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save non-merged components (higher quality at load time - avoids backbone overwrite)
        torch.save(self.policy.decoder.state_dict(), out_dir / "decoder.pt")
        torch.save(self.policy.projection.state_dict(), out_dir / "projection.pt")
        torch.save(self.policy.audio_head.data, out_dir / "audio_head.pt")

        # Save training state
        with open(out_dir / "training_state.json", "w") as f:
            json.dump({
                "step": self.step,
                "best_reward": self.best_reward,
                "config": {
                    "mode": self.config.mode,
                    "group_size": self.config.group_size,
                    "lr": self.config.lr,
                    "beta": self.config.beta,
                    "w_utmos": self.config.w_utmos,
                    "w_cer": self.config.w_cer,
                    "w_sim": self.config.w_sim,
                },
                "metrics_history": self.metrics_history[-100:],
            }, f, indent=2)

        logger.info(f"Checkpoint saved: {out_dir}")

    def train(self):
        """Full GRPO training loop."""
        logger.info(f"\n{'='*60}")
        logger.info(f"GRPO TRAINING")
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  Group size: {self.config.group_size}")
        logger.info(f"  Steps: {self.config.max_steps}")
        logger.info(f"  LR: {self.config.lr} -> {self.config.min_lr}")
        logger.info(f"  Rewards: UTMOS({self.config.w_utmos}) + CER({self.config.w_cer}) + SIM({self.config.w_sim})")
        logger.info(f"  Grad accum: {self.config.grad_accum}")
        logger.info(f"  Prompts: {len(self.prompts)}")
        logger.info(f"{'='*60}\n")

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Running averages for logging
        running_metrics = {}
        n_running = 0

        for self.step in range(1, self.config.max_steps + 1):
            step_start = time.time()

            # Sample random prompt
            prompt = self.prompts[np.random.randint(len(self.prompts))]

            try:
                # 1. Generate candidates
                t0 = time.time()
                candidates = self.generate_candidates(prompt)
                gen_time = time.time() - t0

                # 2. Score candidates
                t0 = time.time()
                candidates = self.score_candidates(candidates, prompt)
                score_time = time.time() - t0

                # 3. Compute advantages
                advantages = self.compute_advantages(candidates)

                # Check for degenerate case (all rewards identical)
                if np.std([c["reward"] for c in candidates]) < 1e-6:
                    logger.warning(f"Step {self.step}: All rewards identical, skipping")
                    continue

                # 4. GRPO update
                t0 = time.time()
                metrics = self.grpo_step(prompt, candidates, advantages)
                update_time = time.time() - t0

                metrics["gen_time"] = gen_time
                metrics["score_time"] = score_time
                metrics["update_time"] = update_time

                # Accumulate metrics
                for k, v in metrics.items():
                    running_metrics[k] = running_metrics.get(k, 0) + v
                n_running += 1

            except Exception as e:
                logger.warning(f"Step {self.step} failed: {e}")
                import traceback
                traceback.print_exc()
                self.optimizer.zero_grad()
                continue

            # Gradient step
            if self.step % self.config.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trainable_params,
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Sync updated weights to generator for next generation
                self._sync_weights_to_generator()

            # Logging
            if self.step % self.config.log_every == 0 and n_running > 0:
                avg = {k: v / n_running for k, v in running_metrics.items()}
                lr = self.optimizer.param_groups[0]["lr"]
                step_time = time.time() - step_start

                logger.info(
                    f"Step {self.step}/{self.config.max_steps} | "
                    f"Loss: {avg['loss']:.4f} | "
                    f"Reward: {avg['mean_reward']:.3f}±{avg['std_reward']:.3f} | "
                    f"UTMOS: {avg['mean_utmos']:.3f} (best:{avg['best_utmos']:.3f}) | "
                    f"CER: {avg['mean_cer']:.3f} | "
                    f"SIM: {avg['mean_sim']:.3f} | "
                    f"LR: {lr:.2e} | "
                    f"Gen:{avg['gen_time']:.0f}s Score:{avg['score_time']:.0f}s Upd:{avg['update_time']:.0f}s"
                )

                self.metrics_history.append({
                    "step": self.step,
                    **avg,
                    "lr": lr,
                })

                # Track best
                if avg["mean_reward"] > self.best_reward:
                    self.best_reward = avg["mean_reward"]

                running_metrics = {}
                n_running = 0

            # Save checkpoint + held-out evaluation
            if self.step % self.config.save_every == 0:
                self.save_checkpoint()

                # Run held-out evaluation
                logger.info(f"Running held-out evaluation at step {self.step}...")
                eval_results = self.evaluate_held_out()
                logger.info(
                    f"EVAL step {self.step}: "
                    f"UTMOS={eval_results['eval_utmos']:.3f}±{eval_results['eval_utmos_std']:.3f} | "
                    f"CER={eval_results['eval_cer']:.3f} | "
                    f"SIM={eval_results['eval_sim']:.3f}"
                )

                # Also save best if improved
                if self.metrics_history and self.metrics_history[-1].get("mean_reward", 0) >= self.best_reward:
                    self.save_checkpoint("best_model")
                    logger.info("*** New best model! ***")

        # Final save
        self.save_checkpoint("final")
        logger.info(f"\nGRPO Training complete!")
        logger.info(f"Best reward: {self.best_reward:.4f}")

        # Print final summary
        if self.metrics_history:
            first = self.metrics_history[0]
            last = self.metrics_history[-1]
            logger.info(f"\nTraining Summary:")
            logger.info(f"  UTMOS:  {first.get('mean_utmos', 0):.3f} -> {last.get('mean_utmos', 0):.3f}")
            logger.info(f"  CER:    {first.get('mean_cer', 0):.3f} -> {last.get('mean_cer', 0):.3f}")
            logger.info(f"  SIM:    {first.get('mean_sim', 0):.3f} -> {last.get('mean_sim', 0):.3f}")
            logger.info(f"  Reward: {first.get('mean_reward', 0):.3f} -> {last.get('mean_reward', 0):.3f}")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for CSM-1B")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--mode", type=str, default="decoder",
                       choices=["decoder", "backbone", "both"],
                       help="What to train: decoder (codebooks 1-31), backbone (codebook 0), or both")
    parser.add_argument("--group-size", type=int, default=8, help="Candidates per prompt")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty (0=simplified)")
    parser.add_argument("--w-utmos", type=float, default=0.4, help="UTMOS reward weight")
    parser.add_argument("--w-cer", type=float, default=0.4, help="CER reward weight")
    parser.add_argument("--w-sim", type=float, default=0.2, help="Speaker sim weight")
    parser.add_argument("--prompts", type=str, default="", help="Prompts file (one per line)")
    parser.add_argument("--checkpoint", type=str, default="", help="Starting checkpoint (SFT or GRPO merged)")
    parser.add_argument("--max-audio-ms", type=int, default=3000, help="Max audio length for generation")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--output-dir", type=str, default="", help="Custom output directory")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    config = GRPOConfig(
        mode=args.mode,
        group_size=args.group_size,
        max_steps=args.steps,
        lr=args.lr,
        beta=args.beta,
        w_utmos=args.w_utmos,
        w_cer=args.w_cer,
        w_sim=args.w_sim,
        prompts_file=args.prompts,
        max_audio_length_ms=args.max_audio_ms,
        grad_accum=args.grad_accum,
        save_every=args.save_every,
    )

    if args.output_dir:
        config.output_dir = args.output_dir

    if args.checkpoint:
        config.sft_checkpoint = args.checkpoint

    trainer = GRPOTrainer(config, device)
    trainer.train()


if __name__ == "__main__":
    main()
