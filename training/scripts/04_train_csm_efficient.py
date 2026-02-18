#!/usr/bin/env python3
"""
Step 4: Fine-tune CSM-1B for Maya Voice - MEMORY EFFICIENT VERSION

This script uses pre-tokenized audio to avoid loading Mimi during training,
saving ~2GB of GPU memory.

Usage:
    python 04_train_csm_efficient.py --data data/csm_ready_ex04
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Setup CSM path
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Setup logging
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
    max_frames: int = 75  # ~3 seconds at 25fps - reduced for memory

    num_epochs: int = 25
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0

    save_steps: int = 200
    eval_steps: int = 100
    save_total_limit: int = 3
    output_dir: str = str(CHECKPOINT_DIR / "csm_maya")
    log_steps: int = 10


class PreTokenizedDataset(Dataset):
    """Dataset loading pre-tokenized audio."""

    def __init__(self, metadata_path: Path, data_dir: Path, max_frames: int = 250):
        with open(metadata_path) as f:
            self.samples = json.load(f)

        self.data_dir = data_dir
        self.max_frames = max_frames

        # Filter by frame count
        original = len(self.samples)
        self.samples = [s for s in self.samples if s.get("num_frames", 0) <= max_frames]
        logger.info(f"Dataset: {len(self.samples)} samples (filtered {original - len(self.samples)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens_path = self.data_dir / sample["tokens_path"]
        audio_tokens = torch.load(tokens_path, weights_only=True)  # (32, num_frames)
        return {
            "audio_tokens": audio_tokens,
            "text": sample.get("text", ""),
            "num_frames": sample.get("num_frames", 0),
        }


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


class CSMTrainer:
    """Memory-efficient trainer for CSM-1B."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        self._init_text_tokenizer()
        self._init_model()
        self._init_datasets()
        self._init_optimizer()

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.training_log = []

    def _init_text_tokenizer(self):
        """Initialize text tokenizer only (no Mimi needed)."""
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing

        logger.info("Loading text tokenizer...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", self.text_tokenizer.bos_token_id), (f"{eos}", self.text_tokenizer.eos_token_id)],
        )

    def _init_model(self):
        """Initialize CSM model."""
        logger.info("=" * 60)
        logger.info("INITIALIZING CSM-1B FOR FINE-TUNING")
        logger.info("=" * 60)

        from models import Model

        logger.info(f"Loading model: {self.config.model_name}")
        self.model = Model.from_pretrained(self.config.model_name)

        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

        self.audio_vocab_size = self.model.config.audio_vocab_size
        self.num_codebooks = self.model.config.audio_num_codebooks

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")

        # Enable gradient checkpointing to save memory
        if hasattr(self.model.backbone, 'gradient_checkpointing_enable'):
            self.model.backbone.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on backbone")

        max_backbone_len = 2048
        self.backbone_mask = _create_causal_mask(max_backbone_len, self.device)
        self.decoder_mask = _create_causal_mask(self.num_codebooks, self.device)

        self.model.train()

    def _init_datasets(self):
        """Initialize datasets using pre-tokenized data."""
        data_dir = Path(self.config.data_dir)
        logger.info(f"Loading datasets from: {data_dir}")

        train_path = data_dir / "train_tokenized.json"
        self.train_dataset = PreTokenizedDataset(train_path, data_dir, self.config.max_frames)
        logger.info(f"Train samples: {len(self.train_dataset)}")

        val_path = data_dir / "val_tokenized.json"
        self.val_dataset = PreTokenizedDataset(val_path, data_dir, self.config.max_frames)
        logger.info(f"Val samples: {len(self.val_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Load in main process to avoid memory overhead
            pin_memory=False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - self.config.warmup_steps, 1), eta_min=self.config.learning_rate * 0.01)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[self.config.warmup_steps])

        logger.info(f"Total training steps: {total_steps}")

    def _compute_loss(self, sample):
        """Compute loss for a single sample."""
        audio_tokens = sample["audio_tokens"]
        # Handle DataLoader batching: (batch, 32, frames) -> (32, frames)
        if audio_tokens.dim() == 3:
            audio_tokens = audio_tokens.squeeze(0)
        audio_tokens = audio_tokens.to(self.device)

        # Handle DataLoader batching: ["text"] -> "text"
        text = sample["text"]
        if isinstance(text, (list, tuple)):
            text = text[0]

        # Text tokenization
        text_tokens = self.text_tokenizer.encode(f"[0]{text}")
        text_len = len(text_tokens)
        audio_len = audio_tokens.size(1)

        if audio_len < 2:
            return None

        # Build sequence
        seq_len = text_len + audio_len
        tokens = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        token_mask = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)

        # Text tokens
        tokens[0, :text_len, -1] = torch.tensor(text_tokens, device=self.device)
        token_mask[0, :text_len, -1] = True

        # Audio tokens
        tokens[0, text_len:, :-1] = audio_tokens.transpose(0, 1)
        token_mask[0, text_len:, :-1] = True

        # Embed and forward backbone
        embeds = self.model._embed_tokens(tokens)
        masked_embeds = embeds * token_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)

        causal_mask = self.backbone_mask[:seq_len, :seq_len].unsqueeze(0)
        h = self.model.backbone(h, mask=causal_mask).to(dtype=self.dtype)

        # Compute loss for each audio frame prediction
        total_loss = 0.0
        num_frames = 0

        for t in range(audio_len - 1):
            pos = text_len + t
            backbone_h = h[:, pos, :]
            target = audio_tokens[:, t + 1]

            # Codebook 0 loss
            c0_logits = self.model.codebook0_head(backbone_h)
            c0_loss = F.cross_entropy(c0_logits, target[0].unsqueeze(0))
            total_loss += c0_loss

            # Teacher forcing for decoder
            c0_embed = self.model._embed_audio(0, audio_tokens[0:1, t].unsqueeze(0))
            curr_h = self.model.projection(torch.cat([backbone_h.unsqueeze(1), c0_embed], dim=1))

            for cb in range(1, self.num_codebooks):
                decoder_len = curr_h.size(1)
                decoder_mask = self.decoder_mask[:decoder_len, :decoder_len].unsqueeze(0)
                decoder_h = self.model.decoder(curr_h, mask=decoder_mask).to(dtype=self.dtype)

                ci_logits = torch.mm(decoder_h[:, -1, :], self.model.audio_head[cb - 1])
                ci_loss = F.cross_entropy(ci_logits, target[cb].unsqueeze(0))
                total_loss += ci_loss

                if cb < self.num_codebooks - 1:
                    ci_embed = self.model._embed_audio(cb, audio_tokens[cb:cb+1, t].unsqueeze(0))
                    curr_h = torch.cat([curr_h, self.model.projection(ci_embed)], dim=1)

            num_frames += 1

        return total_loss / (num_frames * self.num_codebooks) if num_frames > 0 else None

    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx % 50 == 0:
                logger.info(f"Processing batch {batch_idx}/{len(self.train_loader)}")
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_loss(batch)

                if loss is not None:
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    accumulated_loss += loss.item() * self.config.gradient_accumulation_steps

            except Exception as e:
                logger.warning(f"Batch {batch_idx} error: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accumulated_loss
                num_batches += 1

                if self.global_step % self.config.log_steps == 0:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    self.training_log.append({"step": self.global_step, "loss": avg_loss, "lr": lr})

                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate()
                    logger.info(f"Step {self.global_step} | Val Loss: {val_loss:.4f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model")
                        logger.info(f"New best model! (val_loss: {val_loss:.4f})")

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                accumulated_loss = 0.0

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        for batch in self.val_loader:
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_loss(batch)
                if loss is not None:
                    total_loss += loss.item()
                    num_samples += 1
            except:
                continue

        self.model.train()
        return total_loss / max(num_samples, 1)

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), output_dir / "optimizer.pt")
        state = {"global_step": self.global_step, "epoch": self.epoch, "best_val_loss": self.best_val_loss}
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Checkpoint saved: {output_dir}")
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints."""
        output_dir = Path(self.config.output_dir)
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        while len(checkpoints) > self.config.save_total_limit:
            import shutil
            shutil.rmtree(checkpoints.pop(0))

    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("STARTING CSM FINE-TUNING (MEMORY EFFICIENT)")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*60}\nEPOCH {epoch + 1}/{self.config.num_epochs}\n{'='*60}")

            epoch_start = time.time()
            epoch_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Time: {epoch_time/60:.1f}min")
            self.save_checkpoint(f"epoch-{epoch + 1}")

            torch.cuda.empty_cache()
            gc.collect()

        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSM-1B (memory efficient)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "csm_maya"))
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    data_dir = Path(args.data)
    if not (data_dir / "train_tokenized.json").exists():
        logger.error("Pre-tokenized data not found. Run 03b_tokenize_audio.py first.")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    trainer = CSMTrainer(config)

    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path / "model.pt", map_location=trainer.device)
            trainer.model.load_state_dict(state)
            state_file = checkpoint_path / "training_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    ts = json.load(f)
                trainer.global_step = ts["global_step"]
                trainer.epoch = ts["epoch"]
                trainer.best_val_loss = ts["best_val_loss"]

    trainer.train()


if __name__ == "__main__":
    main()
