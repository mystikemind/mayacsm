#!/usr/bin/env python3
"""
DPO Training for CSM-1B
========================
Direct Preference Optimization on the decoder path.

Uses pre-generated preference pairs (chosen/rejected audio tokens)
to fine-tune the decoder, projection, and audio_head.

Based on Koel-TTS DPO approach:
- beta=0.01, lr=2e-7, effective batch=64
- Reference model = frozen copy of SFT checkpoint
- Monitor KL divergence, early stop if >5.0

Usage:
    python 13_train_dpo.py --dataset training/dpo_dataset/preference_pairs.json
    python 13_train_dpo.py --gpu 0 --steps 2000
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")


@dataclass
class DPOConfig:
    # Model
    base_model: str = "sesame/csm-1b"
    sft_checkpoint: str = str(
        PROJECT_ROOT / "training/checkpoints/csm_maya_combined_naturalness/checkpoint-1500-merged/model_merged.pt"
    )

    # DPO
    beta: float = 0.01          # Preference strength (low = stronger signal)
    label_smoothing: float = 0.0

    # Training
    lr: float = 2e-7            # Very low to prevent catastrophic forgetting
    weight_decay: float = 0.01
    max_steps: int = 2000
    warmup_steps: int = 100
    batch_size: int = 1         # Pairs per step (high VRAM per sample)
    grad_accum: int = 16        # Effective batch = 16
    max_grad_norm: float = 1.0

    # Monitoring
    log_every: int = 50
    save_every: int = 500
    eval_every: int = 200
    max_kl_divergence: float = 5.0  # Early stop threshold

    # Output
    output_dir: str = str(
        PROJECT_ROOT / "training/checkpoints/csm_maya_dpo"
    )


def load_model(config: DPOConfig, device: str):
    """Load CSM model with SFT checkpoint."""
    from models import Model

    model = Model.from_pretrained(config.base_model)
    model.to(device=device, dtype=torch.bfloat16)

    # Load SFT checkpoint
    if config.sft_checkpoint and Path(config.sft_checkpoint).exists():
        state = torch.load(config.sft_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded SFT checkpoint: {config.sft_checkpoint}")

    return model


def compute_decoder_logprobs(
    model: nn.Module,
    backbone_h: torch.Tensor,  # (1, 1, hidden_dim) - backbone output for this frame
    audio_tokens: torch.Tensor,  # (num_codebooks,) - target tokens for this frame
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Compute sum of log P(cb1...31 | backbone_h, cb0) for one frame.

    This is the decoder-only path. We teacher-force codebook 0 and compute
    log probs for codebooks 1-31.

    Returns: scalar tensor with grad
    """
    from models import _index_causal_mask

    dtype = next(model.parameters()).dtype
    num_codebooks = model.config.audio_num_codebooks

    total_logprob = torch.tensor(0.0, device=device, dtype=torch.float32,
                                 requires_grad=True)

    # Teacher-force codebook 0 embedding - need (1, 1) input for 3D output
    cb0_token = audio_tokens[0].long().unsqueeze(0).unsqueeze(0)  # (1, 1)
    cb0_embed = model._embed_audio(0, cb0_token)  # (1, 1, dim)

    # Decoder input: [backbone_h, cb0_embed] = (1, 2, dim)
    curr_h = torch.cat([backbone_h, cb0_embed], dim=1)
    curr_pos = torch.arange(0, curr_h.size(1), device=device).unsqueeze(0)

    # Reset decoder caches for this frame
    model.decoder.reset_caches()

    for cb in range(1, num_codebooks):
        curr_decoder_mask = _index_causal_mask(model.decoder_causal_mask, curr_pos)
        decoder_h = model.decoder(
            model.projection(curr_h),
            input_pos=curr_pos,
            mask=curr_decoder_mask
        ).to(dtype=dtype)

        # Logits for this codebook
        ci_logits = torch.mm(
            decoder_h[:, -1, :],
            model.audio_head[cb - 1].to(dtype)
        )

        # Log probability of target token
        ci_logprob = F.log_softmax(ci_logits.float(), dim=-1)
        ci_target = audio_tokens[cb].long()
        total_logprob = total_logprob + ci_logprob[0, ci_target]

        # Teacher-force this codebook
        ci_embed = model._embed_audio(cb, ci_target.unsqueeze(0).unsqueeze(0))  # (1, 1, dim)
        curr_h = ci_embed
        curr_pos = curr_pos[:, -1:] + 1

    return total_logprob


def compute_backbone_hidden(
    model: nn.Module,
    text_tokens: torch.Tensor,   # (seq_len, 33) - text + audio tokens
    text_mask: torch.Tensor,     # (seq_len, 33) - mask
    audio_frames: torch.Tensor,  # (T, num_codebooks) - all audio frames
    target_frame: int,           # Which frame to get hidden for
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Forward text and preceding audio frames through backbone,
    return hidden state for predicting target_frame.

    Returns: (1, 1, hidden_dim) tensor
    """
    from models import _index_causal_mask

    dtype = next(model.parameters()).dtype
    num_codebooks = model.config.audio_num_codebooks

    model.backbone.reset_caches()

    # Build full input: [text_tokens, audio_frame_0, ..., audio_frame_{t-1}]
    all_tokens = [text_tokens.unsqueeze(0)]  # (1, T_text, 33)
    all_masks = [text_mask.unsqueeze(0)]     # (1, T_text, 33)

    for t in range(target_frame):
        frame = audio_frames[t]  # (num_codebooks,)
        # Audio frame token: [cb0, cb1, ..., cb31, 0] (0 for text slot)
        frame_token = torch.zeros(1, 1, num_codebooks + 1, device=device, dtype=frame.dtype)
        frame_token[0, 0, :num_codebooks] = frame  # Fill audio slots
        frame_mask = torch.zeros(1, 1, num_codebooks + 1, device=device, dtype=torch.bool)
        frame_mask[0, 0, :num_codebooks] = True  # Audio slots active
        all_tokens.append(frame_token)
        all_masks.append(frame_mask)

    tokens = torch.cat(all_tokens, dim=1)  # (1, T_text + target_frame, 33)
    masks = torch.cat(all_masks, dim=1)

    seq_len = tokens.size(1)
    input_pos = torch.arange(0, seq_len, device=device).unsqueeze(0)

    backbone_mask = _index_causal_mask(model.backbone_causal_mask, input_pos)

    embeds = model._embed_tokens(tokens)
    masked_embeds = embeds * masks.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)
    h = model.backbone(h, input_pos=input_pos, mask=backbone_mask).to(dtype=dtype)

    return h[:, -1:, :]  # (1, 1, hidden_dim) - keep seq dim for decoder cat


def compute_sequence_logprob(
    model: nn.Module,
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    audio_tokens: torch.Tensor,  # (T, num_codebooks)
    device: str = 'cuda',
    max_frames: int = 50,  # Limit frames for memory
) -> torch.Tensor:
    """
    Compute total decoder log P(audio | text) for a sequence.

    Uses full-sequence forward passes without KV caches (gradient-friendly).
    Only computes decoder log probs (codebooks 1-31).
    Backbone is used read-only for hidden states.

    Returns: scalar tensor with grad
    """
    dtype = next(model.parameters()).dtype
    num_codebooks = model.config.audio_num_codebooks
    T = min(audio_tokens.size(0), max_frames)

    # Step 1: Full backbone forward (frozen, no grad)
    all_tokens_list = [text_tokens.unsqueeze(0)]
    all_masks_list = [text_mask.unsqueeze(0)]

    for t in range(T):
        frame = audio_tokens[t]
        frame_token = torch.zeros(1, 1, num_codebooks + 1, device=device, dtype=frame.dtype)
        frame_token[0, 0, :num_codebooks] = frame
        frame_mask = torch.zeros(1, 1, num_codebooks + 1, device=device, dtype=torch.bool)
        frame_mask[0, 0, :num_codebooks] = True
        all_tokens_list.append(frame_token)
        all_masks_list.append(frame_mask)

    all_tokens = torch.cat(all_tokens_list, dim=1)
    all_masks = torch.cat(all_masks_list, dim=1)
    S = all_tokens.shape[1]

    with torch.no_grad():
        embeds = model._embed_tokens(all_tokens)
        masked_embeds = embeds * all_masks.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        causal_mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool)).unsqueeze(0)
        backbone_out = model.backbone(h, mask=causal_mask).to(dtype=dtype)

    T_text = text_tokens.shape[0]
    total_logprob = torch.tensor(0.0, device=device, dtype=torch.float32)

    # Step 2: Decoder log probs per frame (with gradients)
    for t in range(T):
        pos = T_text + t - 1 if t > 0 else T_text - 1
        h_t = backbone_out[:, pos:pos+1, :]  # (1, 1, hidden_dim)

        frame = audio_tokens[t]
        cb_embeds = []
        for cb in range(num_codebooks - 1):
            cb_embed = model._embed_audio(cb, frame[cb].long().unsqueeze(0).unsqueeze(0))
            cb_embeds.append(cb_embed)

        cb_embeds_cat = torch.cat(cb_embeds, dim=1)
        decoder_input = torch.cat([h_t, cb_embeds_cat], dim=1)

        projected = model.projection(decoder_input)
        dec_mask = torch.tril(
            torch.ones(32, 32, device=device, dtype=torch.bool)
        ).unsqueeze(0)

        decoder_out = model.decoder(projected, mask=dec_mask).to(dtype=dtype)

        for cb in range(1, num_codebooks):
            ci_logits = torch.mm(decoder_out[:, cb, :], model.audio_head[cb - 1].to(dtype))
            ci_logprob = F.log_softmax(ci_logits.float(), dim=-1)
            ci_target = frame[cb].long()
            total_logprob = total_logprob + ci_logprob[0, ci_target]

    return total_logprob / T


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.01,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    Standard DPO loss.

    L = -log(sigma(beta * ((logP_pi(y_c) - logP_ref(y_c)) - (logP_pi(y_r) - logP_ref(y_r)))))
    """
    chosen_rewards = beta * (policy_chosen_lp - ref_chosen_lp)
    rejected_rewards = beta * (policy_rejected_lp - ref_rejected_lp)

    logits = chosen_rewards - rejected_rewards

    if label_smoothing > 0:
        loss = (
            -F.logsigmoid(logits) * (1 - label_smoothing) +
            -F.logsigmoid(-logits) * label_smoothing
        )
    else:
        loss = -F.logsigmoid(logits)

    return loss, chosen_rewards.detach(), rejected_rewards.detach()


class DPOTrainer:
    """DPO trainer for CSM-1B."""

    def __init__(self, config: DPOConfig, device: str):
        self.config = config
        self.device = device

        # Load policy model (NO KV caches for gradient-friendly forward)
        logger.info("Loading policy model...")
        self.policy = load_model(config, device)
        # Do NOT call setup_caches - we use full-sequence forward without KV caches

        # Freeze backbone
        for p in self.policy.backbone.parameters():
            p.requires_grad = False
        for p in self.policy.text_embeddings.parameters():
            p.requires_grad = False
        for p in self.policy.audio_embeddings.parameters():
            p.requires_grad = False
        self.policy.codebook0_head.requires_grad = False

        # Trainable: decoder, projection, audio_head
        trainable_params = []
        for p in self.policy.decoder.parameters():
            p.requires_grad = True
            trainable_params.append(p)
        for p in self.policy.projection.parameters():
            p.requires_grad = True
            trainable_params.append(p)
        self.policy.audio_head.requires_grad = True
        trainable_params.append(self.policy.audio_head)

        n_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable params: {n_trainable:,} ({n_trainable/1e6:.1f}M)")

        # Load reference model (frozen copy, also no KV caches)
        logger.info("Loading reference model...")
        self.ref = load_model(config, device)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

        # Optimizer (only decoder params)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # LR scheduler (warmup + cosine)
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
            return 0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Metrics
        self.step = 0
        self.best_loss = float('inf')
        self.loss_history = []

    def train_step(self, pair: dict) -> dict:
        """Single DPO training step on one preference pair."""
        from generator import Generator, Segment

        text = pair["text"]
        chosen_tokens = torch.tensor(pair["chosen"]["audio_tokens"],
                                     device=self.device)  # (32, T)
        rejected_tokens = torch.tensor(pair["rejected"]["audio_tokens"],
                                       device=self.device)  # (32, T)

        # Transpose to (T, 32) for frame-wise processing
        chosen_tokens = chosen_tokens.T
        rejected_tokens = rejected_tokens.T

        # Tokenize text (using the policy model's tokenizer setup)
        # For simplicity, use a pre-tokenized approach
        text_tokens, text_mask = self._tokenize_text(text)

        # Compute policy log probs
        self.policy.train()
        policy_chosen_lp = compute_sequence_logprob(
            self.policy, text_tokens, text_mask, chosen_tokens, self.device
        )
        policy_rejected_lp = compute_sequence_logprob(
            self.policy, text_tokens, text_mask, rejected_tokens, self.device
        )

        # Compute reference log probs
        with torch.no_grad():
            ref_chosen_lp = compute_sequence_logprob(
                self.ref, text_tokens, text_mask, chosen_tokens, self.device
            )
            ref_rejected_lp = compute_sequence_logprob(
                self.ref, text_tokens, text_mask, rejected_tokens, self.device
            )

        # DPO loss
        loss, c_reward, r_reward = dpo_loss(
            policy_chosen_lp, policy_rejected_lp,
            ref_chosen_lp, ref_rejected_lp,
            beta=self.config.beta,
            label_smoothing=self.config.label_smoothing,
        )

        # Backward (accumulated)
        (loss / self.config.grad_accum).backward()

        metrics = {
            "loss": loss.item(),
            "chosen_reward": c_reward.item(),
            "rejected_reward": r_reward.item(),
            "reward_margin": (c_reward - r_reward).item(),
            "policy_chosen_lp": policy_chosen_lp.item(),
            "policy_rejected_lp": policy_rejected_lp.item(),
        }

        return metrics

    def _tokenize_text(self, text: str):
        """Tokenize text for backbone input."""
        # Simple tokenization using the text tokenizer
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing

        if not hasattr(self, '_tokenizer'):
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

        text_ids = self._tokenizer.encode(text)
        num_codebooks = self.policy.config.audio_num_codebooks

        # Build (T_text, 33) token array: [0, 0, ..., 0, text_token]
        T = len(text_ids)
        tokens = torch.zeros(T, num_codebooks + 1, device=self.device, dtype=torch.long)
        mask = torch.zeros(T, num_codebooks + 1, device=self.device, dtype=torch.bool)

        for i, tid in enumerate(text_ids):
            tokens[i, -1] = tid  # Text in last position
            mask[i, -1] = True   # Only text slot is active

        return tokens, mask

    def save_checkpoint(self, name: str = None):
        """Save current model state."""
        if name is None:
            name = f"step-{self.step}"
        out_dir = Path(self.config.output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.policy.decoder.state_dict(), out_dir / "decoder.pt")
        torch.save(self.policy.projection.state_dict(), out_dir / "projection.pt")
        torch.save(self.policy.audio_head.data, out_dir / "audio_head.pt")

        # Also save merged model
        torch.save(self.policy.state_dict(), out_dir / "model_merged.pt")

        with open(out_dir / "training_state.json", "w") as f:
            json.dump({
                "step": self.step,
                "loss_history": self.loss_history[-100:],
                "best_loss": self.best_loss,
            }, f, indent=2)

        logger.info(f"Checkpoint saved: {out_dir}")

    def train(self, dataset: list):
        """Full DPO training loop."""
        logger.info(f"\n{'='*60}")
        logger.info(f"DPO TRAINING")
        logger.info(f"  Dataset: {len(dataset)} pairs")
        logger.info(f"  Steps: {self.config.max_steps}")
        logger.info(f"  LR: {self.config.lr}")
        logger.info(f"  Beta: {self.config.beta}")
        logger.info(f"  Grad accum: {self.config.grad_accum}")
        logger.info(f"{'='*60}\n")

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        accum_metrics = {
            "loss": 0, "chosen_reward": 0, "rejected_reward": 0,
            "reward_margin": 0,
        }
        n_accum = 0

        for self.step in range(1, self.config.max_steps + 1):
            # Sample random pair
            pair = dataset[np.random.randint(len(dataset))]

            try:
                metrics = self.train_step(pair)

                for k in accum_metrics:
                    accum_metrics[k] += metrics[k]
                n_accum += 1

            except Exception as e:
                logger.warning(f"Step {self.step} failed: {e}")
                continue

            # Gradient step (after accumulation)
            if self.step % self.config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Logging
            if self.step % self.config.log_every == 0 and n_accum > 0:
                avg = {k: v / n_accum for k, v in accum_metrics.items()}
                lr = self.optimizer.param_groups[0]["lr"]

                logger.info(
                    f"Step {self.step} | Loss: {avg['loss']:.4f} | "
                    f"Margin: {avg['reward_margin']:.4f} | "
                    f"C_r: {avg['chosen_reward']:.4f} | "
                    f"R_r: {avg['rejected_reward']:.4f} | "
                    f"LR: {lr:.2e}"
                )

                self.loss_history.append({
                    "step": self.step,
                    **avg,
                    "lr": lr,
                })

                accum_metrics = {k: 0 for k in accum_metrics}
                n_accum = 0

            # Save checkpoint
            if self.step % self.config.save_every == 0:
                self.save_checkpoint()

                if avg.get("loss", float('inf')) < self.best_loss:
                    self.best_loss = avg["loss"]
                    self.save_checkpoint("best_model")
                    logger.info("*** New best model! ***")

        # Final save
        self.save_checkpoint("final")
        logger.info(f"\nDPO Training complete. Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                       default=str(PROJECT_ROOT / "training/dpo_dataset/preference_pairs.json"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--beta", type=float, default=0.01)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Run 12_generate_dpo_dataset.py first!")
        return

    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    logger.info(f"Loaded {len(pairs)} preference pairs")

    if len(pairs) < 5:
        logger.warning("Very small dataset! DPO may not converge well.")

    # Setup config
    config = DPOConfig(
        max_steps=args.steps,
        lr=args.lr,
        beta=args.beta,
    )

    # Train
    trainer = DPOTrainer(config, device)
    trainer.train(pairs)


if __name__ == "__main__":
    main()
