#!/usr/bin/env python3
"""
CSM Fast Fine-Tuning - Production Quality
==========================================

Simple, clean, efficient fine-tuning approach:
1. Train backbone + codebook0_head + embeddings
2. Freeze decoder (it handles audio codec, not voice identity)
3. Backbone-only loss (vectorized, GPU-efficient)

Why this works:
- Backbone learns speaker identity, prosody, and text-to-audio mapping
- Codebook 0 captures coarse audio structure (most important)
- Decoder is a neural audio codec - already trained, doesn't need voice-specific tuning

Target: Sesame AI Maya-level voice quality with efficient training
"""

import os
import sys
import json
import logging
import argparse
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Setup CSM path
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str = "sesame/csm-1b"
    use_bf16: bool = True
    data_dir: str = ""
    max_frames: int = 200

    # Training hyperparameters
    num_epochs: int = 50
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Checkpointing
    save_steps: int = 200
    eval_steps: int = 100
    save_total_limit: int = 3
    output_dir: str = str(CHECKPOINT_DIR / "csm_maya_fast")
    log_steps: int = 10


class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized audio."""

    def __init__(self, metadata_path: Path, data_dir: Path, max_frames: int = 200):
        with open(metadata_path) as f:
            self.samples = json.load(f)

        self.data_dir = data_dir
        self.max_frames = max_frames

        original = len(self.samples)
        self.samples = [s for s in self.samples if s.get("num_frames", 0) <= max_frames]
        logger.info(f"Dataset: {len(self.samples)} samples (filtered {original - len(self.samples)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens_path = self.data_dir / sample["tokens_path"]
        audio_tokens = torch.load(tokens_path, weights_only=True)
        return {
            "audio_tokens": audio_tokens,
            "text": sample.get("text", ""),
        }


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


class CSMFastTrainer:
    """Efficient CSM trainer - backbone-only loss."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda")

        torch.cuda.empty_cache()
        gc.collect()

        self._init_tokenizer()
        self._init_model()
        self._freeze_decoder()
        self._init_datasets()
        self._init_optimizer()

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.training_log = []

    def _init_tokenizer(self):
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing

        logger.info("Loading tokenizer...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(bos, self.text_tokenizer.bos_token_id), (eos, self.text_tokenizer.eos_token_id)],
        )

    def _init_model(self):
        logger.info("=" * 60)
        logger.info("CSM FAST FINE-TUNING")
        logger.info("=" * 60)

        from models import Model
        self.model = Model.from_pretrained(self.config.model_name)

        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

        self.audio_vocab_size = self.model.config.audio_vocab_size
        self.num_codebooks = self.model.config.audio_num_codebooks

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")

        # Causal mask
        self.backbone_mask = _create_causal_mask(2048, self.device)

    def _freeze_decoder(self):
        """Freeze decoder - it doesn't need voice-specific training."""
        logger.info("Freezing decoder...")

        # Freeze decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        # Freeze audio_head (decoder output)
        self.model.audio_head.requires_grad = False

        # Freeze projection (decoder input)
        for param in self.model.projection.parameters():
            param.requires_grad = False

        # Keep backbone, codebook0_head, embeddings trainable
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable:,}")

    def _init_datasets(self):
        data_dir = Path(self.config.data_dir)

        train_path = data_dir / "train_tokenized.json"
        self.train_dataset = PreTokenizedDataset(train_path, data_dir, self.config.max_frames)
        logger.info(f"Train: {len(self.train_dataset)} samples")

        val_path = data_dir / "val_tokenized.json"
        self.val_dataset = PreTokenizedDataset(val_path, data_dir, self.config.max_frames)
        logger.info(f"Val: {len(self.val_dataset)} samples")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _init_optimizer(self):
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
        )

        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=self.config.min_lr)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_steps])

        logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    def _compute_loss(self, sample) -> Optional[torch.Tensor]:
        """Backbone-only loss - vectorized and efficient."""
        audio_tokens = sample["audio_tokens"]
        if audio_tokens.dim() == 3:
            audio_tokens = audio_tokens.squeeze(0)
        audio_tokens = audio_tokens.to(self.device)

        text = sample["text"]
        if isinstance(text, (list, tuple)):
            text = text[0]

        text_tokens = self.text_tokenizer.encode(f"[0]{text}")
        text_len = len(text_tokens)
        audio_len = audio_tokens.size(1)

        if audio_len < 3:
            return None

        # Build sequence
        seq_len = text_len + audio_len
        tokens = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        token_mask = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)

        tokens[0, :text_len, -1] = torch.tensor(text_tokens, device=self.device)
        token_mask[0, :text_len, -1] = True
        tokens[0, text_len:, :-1] = audio_tokens.transpose(0, 1)
        token_mask[0, text_len:, :-1] = True

        # Forward
        embeds = self.model._embed_tokens(tokens)
        masked_embeds = embeds * token_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)

        causal_mask = self.backbone_mask[:seq_len, :seq_len].unsqueeze(0)
        backbone_h = self.model.backbone(h, mask=causal_mask).to(dtype=self.dtype)

        # Codebook 0 loss (vectorized)
        audio_h = backbone_h[0, text_len:text_len + audio_len - 1, :]
        c0_targets = audio_tokens[0, 1:audio_len]
        c0_logits = self.model.codebook0_head(audio_h)

        loss = F.cross_entropy(c0_logits, c0_targets)
        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        self.optimizer.zero_grad()
        accum_loss = 0.0
        batch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_loss(batch)

                if loss is not None and torch.isfinite(loss):
                    (loss / self.config.gradient_accumulation_steps).backward()
                    accum_loss += loss.item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                continue

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accum_loss
                num_steps += 1

                if self.global_step % self.config.log_steps == 0:
                    elapsed = time.time() - batch_start
                    speed = (self.config.gradient_accumulation_steps * self.config.log_steps) / elapsed
                    lr = self.scheduler.get_last_lr()[0]

                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {accum_loss/self.config.gradient_accumulation_steps:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Grad: {grad_norm:.2f} | "
                        f"Speed: {speed:.1f} samp/s"
                    )
                    batch_start = time.time()

                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate()
                    logger.info(f"Step {self.global_step} | Val: {val_loss:.4f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model")
                        logger.info("New best model saved!")

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                accum_loss = 0.0

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in self.val_loader:
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_loss(batch)
                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    n += 1
            except:
                continue

        self.model.train()
        return total_loss / max(n, 1)

    def save_checkpoint(self, name: str):
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full model state (only trainable weights will be different)
        torch.save(self.model.state_dict(), output_dir / "model.pt")

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved: {output_dir}")

    def train(self):
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"LR: {self.config.learning_rate}")
        logger.info("=" * 60)

        start = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*60}\nEPOCH {epoch + 1}/{self.config.num_epochs}\n{'='*60}")

            epoch_start = time.time()
            epoch_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Time: {epoch_time/60:.1f}min")
            self.save_checkpoint(f"epoch-{epoch + 1}")

            torch.cuda.empty_cache()

        total = time.time() - start
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Time: {total/3600:.2f}h | Best val: {self.best_val_loss:.4f}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output", default=str(CHECKPOINT_DIR / "csm_maya_fast"))
    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    trainer = CSMFastTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
