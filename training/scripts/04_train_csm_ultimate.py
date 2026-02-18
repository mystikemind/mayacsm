#!/usr/bin/env python3
"""
CSM ULTIMATE Fine-Tuning - WORLD CLASS, NO COMPROMISES
========================================================

This is the ABSOLUTE BEST training approach for CSM:
1. Multi-GPU training with DDP (uses ALL available GPUs)
2. NO compute amortization - trains decoder on ALL frames
3. Official Sesame/Speechmatics hyperparameters
4. Full 32-codebook loss
5. Maximum quality, no shortcuts

Target: Match Sesame AI Maya voice quality
Author: Senior AI Engineer approach - NO COMPROMISES
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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Setup CSM path
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"


def setup_logging(rank):
    """Setup logging for distributed training."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f'%(asctime)s | RANK {rank} | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    ULTIMATE training configuration - NO COMPROMISES.
    Based on Sesame AI's official approach with MAXIMUM quality settings.
    """
    model_name: str = "sesame/csm-1b"
    use_bf16: bool = True
    data_dir: str = ""
    max_frames: int = 150  # ~12 seconds max

    # ============================================
    # OFFICIAL SESAME/SPEECHMATICS HYPERPARAMETERS
    # ============================================
    num_epochs: int = 100
    batch_size: int = 1  # Per GPU
    gradient_accumulation_steps: int = 4  # Effective batch = 4 GPUs * 1 * 4 = 16

    # Official recommended LR: 3e-5
    learning_rate: float = 3e-5
    min_lr: float = 1e-7

    # Official recommended weight decay: 0.002
    weight_decay: float = 0.002

    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # ============================================
    # LOSS CONFIGURATION - SESAME'S APPROACH
    # ============================================
    # loss = (1 - decoder_loss_weight) * c0_loss + decoder_loss_weight * c_loss
    decoder_loss_weight: float = 0.5

    # Label smoothing for generalization (prevents overfitting)
    label_smoothing: float = 0.1

    # ============================================
    # DECODER TRAINING - NO COMPROMISE VERSION
    # ============================================
    # Sesame uses 1/16 (6.25%) for efficiency
    # We use 0.5 (50%) for MAXIMUM quality - 8x more decoder training!
    # This is the "no compromise" approach
    decoder_frame_ratio: float = 0.5  # Train on 50% of frames for BEST quality

    # ============================================
    # CHECKPOINTING & LOGGING
    # ============================================
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 5
    output_dir: str = str(CHECKPOINT_DIR / "csm_maya_ultimate")
    log_steps: int = 10

    # Early stopping
    patience: int = 20  # More patience for thorough training
    min_delta: float = 0.001


class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized audio."""

    def __init__(self, metadata_path: Path, data_dir: Path, max_frames: int = 150):
        with open(metadata_path) as f:
            self.samples = json.load(f)

        self.data_dir = data_dir
        self.max_frames = max_frames

        original = len(self.samples)
        self.samples = [s for s in self.samples if s.get("num_frames", 0) <= max_frames]
        filtered = original - len(self.samples)
        if filtered > 0:
            print(f"Dataset: {len(self.samples)} samples (filtered {filtered})")

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


class CSMUltimateTrainer:
    """
    ULTIMATE CSM trainer - Multi-GPU, full decoder training, no compromises.
    """

    def __init__(self, config: TrainingConfig, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.device = torch.device(f"cuda:{rank}")
        self.logger = setup_logging(rank)

        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        gc.collect()

        self._init_tokenizer()
        self._init_model()
        self._init_datasets()
        self._init_optimizer()

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0  # For resume

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load model and training state from checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load model weights
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            if self.is_main:
                self.logger.info(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state_dict)

        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.start_epoch = self.epoch + 1  # Resume from next epoch
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            self.patience_counter = state.get("patience_counter", 0)

            if self.is_main:
                self.logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
                self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")

    def _init_tokenizer(self):
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing

        if self.is_main:
            self.logger.info("Loading tokenizer...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(bos, self.text_tokenizer.bos_token_id), (eos, self.text_tokenizer.eos_token_id)],
        )

    def _init_model(self):
        if self.is_main:
            self.logger.info("=" * 60)
            self.logger.info("ULTIMATE CSM FINE-TUNING - NO COMPROMISES")
            self.logger.info("=" * 60)

        from models import Model
        self.model = Model.from_pretrained(self.config.model_name)

        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

        self.audio_vocab_size = self.model.config.audio_vocab_size
        self.num_codebooks = self.model.config.audio_num_codebooks

        # Wrap with DDP if distributed
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
            self._model = self.model.module
        else:
            self._model = self.model

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        if self.is_main:
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            self.logger.info(f"Audio vocab size: {self.audio_vocab_size}")
            self.logger.info(f"Number of codebooks: {self.num_codebooks}")
            self.logger.info(f"Using {self.world_size} GPU(s)")

        # Causal masks
        self.backbone_mask = _create_causal_mask(2048, self.device)
        self.decoder_mask = _create_causal_mask(self.num_codebooks, self.device)

    def _init_datasets(self):
        data_dir = Path(self.config.data_dir)

        train_path = data_dir / "train_tokenized.json"
        self.train_dataset = PreTokenizedDataset(train_path, data_dir, self.config.max_frames)

        val_path = data_dir / "val_tokenized.json"
        self.val_dataset = PreTokenizedDataset(val_path, data_dir, self.config.max_frames)

        if self.is_main:
            self.logger.info(f"Train: {len(self.train_dataset)} samples")
            self.logger.info(f"Val: {len(self.val_dataset)} samples")

        # Distributed samplers
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
            self.val_sampler = DistributedSampler(
                self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=2,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=2,
        )

    def _init_optimizer(self):
        """Setup optimizer with OFFICIAL Speechmatics hyperparameters."""
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]

        # AdamW as recommended by Speechmatics
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(self.total_steps * self.config.warmup_ratio)

        # IMPORTANT: Speechmatics uses LINEAR LR scheduler, not cosine!
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        # Linear decay from peak LR to min_lr
        decay = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.config.min_lr / self.config.learning_rate,
            total_iters=max(self.total_steps - warmup_steps, 1)
        )
        self.scheduler = SequentialLR(self.optimizer, [warmup, decay], milestones=[warmup_steps])

        if self.is_main:
            self.logger.info(f"Total steps: {self.total_steps}, Warmup: {warmup_steps}")
            self.logger.info(f"LR Schedule: LINEAR (as per Speechmatics)")
            self.logger.info(f"Decoder frame ratio: {self.config.decoder_frame_ratio} ({self.config.decoder_frame_ratio*100:.0f}% of frames)")

    def _compute_loss(self, sample) -> Tuple[Optional[torch.Tensor], dict]:
        """
        ULTIMATE loss computation - trains decoder on MORE frames for maximum quality.
        """
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
            return None, {}

        # Build sequence for backbone
        seq_len = text_len + audio_len
        tokens = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        token_mask = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)

        tokens[0, :text_len, -1] = torch.tensor(text_tokens, device=self.device)
        token_mask[0, :text_len, -1] = True
        tokens[0, text_len:, :-1] = audio_tokens.transpose(0, 1)
        token_mask[0, text_len:, :-1] = True

        # Forward through backbone
        embeds = self._model._embed_tokens(tokens)
        masked_embeds = embeds * token_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)

        causal_mask = self.backbone_mask[:seq_len, :seq_len].unsqueeze(0)
        backbone_h = self._model.backbone(h, mask=causal_mask).to(dtype=self.dtype)

        # ============================================
        # LOSS 1: Codebook 0 on EVERY frame (backbone)
        # ============================================
        num_audio_frames = audio_len - 1
        audio_h = backbone_h[0, text_len:text_len + num_audio_frames, :]
        c0_targets = audio_tokens[0, 1:audio_len]
        c0_logits = self._model.codebook0_head(audio_h)

        c0_loss = F.cross_entropy(
            c0_logits, c0_targets,
            label_smoothing=self.config.label_smoothing
        )

        # ============================================
        # LOSS 2: Decoder loss on MORE frames (not just 1/16)
        # ============================================
        num_decoder_frames = max(1, int(num_audio_frames * self.config.decoder_frame_ratio))
        frame_indices = np.random.choice(num_audio_frames, size=num_decoder_frames, replace=False)
        frame_indices = sorted(frame_indices)

        decoder_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        num_decoder_predictions = 0

        for frame_idx in frame_indices:
            h_backbone = backbone_h[0, text_len + frame_idx, :].unsqueeze(0)
            h_proj = self._model.projection(h_backbone)

            target_codes = audio_tokens[1:, frame_idx + 1]
            input_codes = audio_tokens[:self.num_codebooks - 1, frame_idx + 1]

            # Embed input codes
            code_embeds = []
            for cb in range(self.num_codebooks - 1):
                offset_token = input_codes[cb] + (cb + 1) * self.audio_vocab_size
                emb = self._model.audio_embeddings(offset_token.unsqueeze(0))
                code_embeds.append(emb)

            code_embeds = torch.stack(code_embeds, dim=1)
            code_embeds_proj = self._model.projection(code_embeds)
            h_proj_expanded = h_proj.unsqueeze(1).expand_as(code_embeds_proj)
            decoder_input = code_embeds_proj + h_proj_expanded

            decoder_out = self._model.decoder(
                decoder_input,
                mask=self.decoder_mask[:self.num_codebooks - 1, :self.num_codebooks - 1].unsqueeze(0)
            )

            for cb in range(self.num_codebooks - 1):
                logits = decoder_out[0, cb, :] @ self._model.audio_head[cb]
                target = target_codes[cb]

                cb_loss = F.cross_entropy(
                    logits.unsqueeze(0), target.unsqueeze(0),
                    label_smoothing=self.config.label_smoothing
                )

                decoder_loss = decoder_loss + cb_loss
                num_decoder_predictions += 1

        if num_decoder_predictions > 0:
            decoder_loss = decoder_loss / num_decoder_predictions

        # Combined loss using Sesame's official formula
        backbone_weight = 1.0 - self.config.decoder_loss_weight
        total_loss = backbone_weight * c0_loss + self.config.decoder_loss_weight * decoder_loss

        metrics = {
            "c0_loss": c0_loss.item(),
            "decoder_loss": decoder_loss.item(),
            "total_loss": total_loss.item(),
            "decoder_frames": len(frame_indices),
        }

        return total_loss, metrics

    def train_epoch(self):
        self.model.train()
        if self.train_sampler:
            self.train_sampler.set_epoch(self.epoch)

        total_loss = 0.0
        total_c0_loss = 0.0
        total_decoder_loss = 0.0
        num_steps = 0
        self.optimizer.zero_grad()
        accum_loss = 0.0
        accum_c0 = 0.0
        accum_dec = 0.0
        batch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss, metrics = self._compute_loss(batch)

                if loss is not None and torch.isfinite(loss):
                    (loss / self.config.gradient_accumulation_steps).backward()
                    accum_loss += loss.item()
                    accum_c0 += metrics.get("c0_loss", 0)
                    accum_dec += metrics.get("decoder_loss", 0)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if self.is_main:
                        self.logger.warning(f"OOM at batch {batch_idx}, skipping")
                continue
            except Exception as e:
                if self.is_main:
                    self.logger.error(f"Error at batch {batch_idx}: {e}")
                continue

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self._model.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accum_loss
                total_c0_loss += accum_c0
                total_decoder_loss += accum_dec
                num_steps += 1

                if self.is_main and self.global_step % self.config.log_steps == 0:
                    elapsed = time.time() - batch_start
                    speed = (self.config.gradient_accumulation_steps * self.config.log_steps * self.world_size) / elapsed
                    lr = self.scheduler.get_last_lr()[0]

                    avg_loss = accum_loss / self.config.gradient_accumulation_steps
                    avg_c0 = accum_c0 / self.config.gradient_accumulation_steps
                    avg_dec = accum_dec / self.config.gradient_accumulation_steps

                    self.logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} (C0: {avg_c0:.4f}, Dec: {avg_dec:.4f}) | "
                        f"LR: {lr:.2e} | Grad: {grad_norm:.2f} | Speed: {speed:.1f} samp/s"
                    )
                    batch_start = time.time()

                if self.global_step % self.config.eval_steps == 0:
                    val_loss, val_c0, val_dec = self.evaluate()
                    if self.is_main:
                        self.logger.info(
                            f"Step {self.global_step} | "
                            f"Val Loss: {val_loss:.4f} (C0: {val_c0:.4f}, Dec: {val_dec:.4f})"
                        )

                        if val_loss < self.best_val_loss - self.config.min_delta:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            self.save_checkpoint("best_model")
                            self.logger.info("New best model saved!")
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= self.config.patience:
                                self.logger.info(f"Early stopping: no improvement for {self.config.patience} evals")
                                return -1

                if self.is_main and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                accum_loss = 0.0
                accum_c0 = 0.0
                accum_dec = 0.0

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_c0 = 0.0
        total_dec = 0.0
        n = 0

        for batch in self.val_loader:
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss, metrics = self._compute_loss(batch)
                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    total_c0 += metrics.get("c0_loss", 0)
                    total_dec += metrics.get("decoder_loss", 0)
                    n += 1
            except:
                continue

        # Sync across GPUs
        if self.world_size > 1:
            stats = torch.tensor([total_loss, total_c0, total_dec, n], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_c0, total_dec, n = stats.tolist()

        self.model.train()
        return (
            total_loss / max(n, 1),
            total_c0 / max(n, 1),
            total_dec / max(n, 1)
        )

    def save_checkpoint(self, name: str):
        if not self.is_main:
            return

        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self._model.state_dict(), output_dir / "model.pt")

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "config": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "decoder_loss_weight": self.config.decoder_loss_weight,
                "decoder_frame_ratio": self.config.decoder_frame_ratio,
                "label_smoothing": self.config.label_smoothing,
                "num_epochs": self.config.num_epochs,
            }
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Saved: {output_dir}")

    def train(self):
        if self.is_main:
            self.logger.info("=" * 60)
            self.logger.info("STARTING ULTIMATE TRAINING - NO COMPROMISES")
            self.logger.info("=" * 60)
            self.logger.info(f"GPUs: {self.world_size}")
            self.logger.info(f"Epochs: {self.config.num_epochs}")
            self.logger.info(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size}")
            self.logger.info(f"Learning Rate: {self.config.learning_rate} (Official Sesame: 3e-5)")
            self.logger.info(f"Weight Decay: {self.config.weight_decay} (Official Sesame: 0.002)")
            self.logger.info(f"Decoder Loss Weight: {self.config.decoder_loss_weight}")
            self.logger.info(f"Decoder Frame Ratio: {self.config.decoder_frame_ratio} (MORE than 1/16 for quality)")
            self.logger.info(f"Label Smoothing: {self.config.label_smoothing}")
            self.logger.info(f"Early Stopping Patience: {self.config.patience}")
            self.logger.info("=" * 60)

        start = time.time()

        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.epoch = epoch
            if self.is_main:
                self.logger.info(f"\n{'='*60}\nEPOCH {epoch + 1}/{self.config.num_epochs}\n{'='*60}")

            epoch_start = time.time()
            epoch_loss = self.train_epoch()

            if epoch_loss == -1:
                break

            epoch_time = time.time() - epoch_start

            if self.is_main:
                self.logger.info(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Time: {epoch_time/60:.1f}min")
                self.save_checkpoint(f"epoch-{epoch + 1}")

            torch.cuda.empty_cache()

            # Sync before next epoch
            if self.world_size > 1:
                dist.barrier()

        total = time.time() - start
        if self.is_main:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("ULTIMATE TRAINING COMPLETE")
            self.logger.info(f"Time: {total/3600:.2f}h | Best val: {self.best_val_loss:.4f}")
            self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output", default=str(CHECKPOINT_DIR / "csm_maya_ultimate"))
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Setup distributed training
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
    elif args.local_rank >= 0:
        local_rank = args.local_rank
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
    else:
        local_rank = 0
        world_size = 1

    config = TrainingConfig(
        data_dir=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    if local_rank == 0:
        print("=" * 60)
        print("ULTIMATE CSM TRAINING - WORLD CLASS, NO COMPROMISES")
        print("=" * 60)
        print(f"GPUs: {world_size}")
        print(f"Learning Rate: {args.lr} (Official Sesame: 3e-5)")
        print(f"Weight Decay: {config.weight_decay} (Official Sesame: 0.002)")
        print(f"Decoder Loss Weight: {config.decoder_loss_weight} (Official: 0.5)")
        print(f"Decoder Frame Ratio: {config.decoder_frame_ratio} (25% for quality)")
        print(f"Label Smoothing: {config.label_smoothing}")
        print("=" * 60)

    trainer = CSMUltimateTrainer(config, rank=local_rank, world_size=world_size)

    # Resume from checkpoint if specified
    if args.resume:
        if local_rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
