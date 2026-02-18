#!/usr/bin/env python3
"""
CSM Combined Naturalness Training
===================================

Train decoder (codebooks 1-31) on ALL prepared datasets for maximum naturalness.

Datasets:
  - DisfluencySpeech: 5000 samples (disfluencies, natural pauses, hesitations)
  - NonverbalTTS: 6088 samples (breathing, laughter, emotions)
  - Expresso full: ~11K samples (26 styles, diverse expression)
  - Expresso ex04: 2081 samples (Maya speaker identity)

Strategy:
  - Backbone FROZEN (preserves naturalness from 1M hour pretraining)
  - Decoder + projection + audio_head TRAINED
  - Weighted sampling: naturalistic data weighted higher
  - Codebook-wise loss weighting: lower codebooks weighted higher (more perceptual)
  - Two-phase training: Phase 1 diverse data, Phase 2 Maya-focused

Usage:
    # Full combined training from scratch
    python 07_train_combined_naturalness.py

    # Resume from checkpoint
    python 07_train_combined_naturalness.py --resume /path/to/checkpoint

    # Phase 2 only (after phase 1)
    python 07_train_combined_naturalness.py --phase 2 --resume /path/to/phase1_best
"""

import os
import sys
import json
import logging
import argparse
import time
import gc
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
TRAINING_DATA = PROJECT_ROOT / "training" / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"


# ============================================================
# Dataset Configuration
# ============================================================

DATASET_CONFIGS = {
    "disfluency_speech": {
        "path": TRAINING_DATA / "datasets" / "disfluency_speech_prepared",
        "weight": 2.0,  # High: teaches natural disfluencies
        "description": "Natural disfluencies, hesitations, fillers",
    },
    "nonverbal_tts": {
        "path": TRAINING_DATA / "datasets" / "nonverbal_tts_prepared",
        "weight": 2.0,  # High: teaches breathing, laughter, emotion
        "description": "Breathing, laughter, emotional expressions",
    },
    "expresso_full": {
        "path": TRAINING_DATA / "datasets" / "expresso_full_prepared",
        "weight": 1.0,  # Normal: diverse styles (all speakers)
        "description": "26 expression styles, diverse (all speakers)",
        # NOTE: ex04 speaker samples in this dataset get boosted weight
        # via speaker_weight_boost below
    },
}

# Boost weight for Maya speaker (ex04) samples within expresso_full
# This replaces the old standalone ex04 config which had no text metadata
SPEAKER_WEIGHT_BOOST = {
    "ex04": 1.5,  # ex04 samples get 1.0 * 1.5 = 1.5x weight
}

# Codebook loss weights based on CoVoC 2024 research (arxiv:2412.01100)
# Lower codebooks carry more perceptual information in RVQ
CODEBOOK_WEIGHTS = None  # Will be set based on num_codebooks


def get_codebook_weights(num_codebooks: int) -> torch.Tensor:
    """
    Generate importance weights for codebooks 1-31.

    Based on CoVoC 2024: alpha_k = {5, 2, 1, 0.5} for decreasing levels.
    Lower codebooks (closer to codebook 0) carry more perceptual info.
    """
    weights = []
    for i in range(num_codebooks - 1):
        if i < 4:
            w = 5.0    # Codebooks 1-4: coarse acoustic structure
        elif i < 10:
            w = 2.0    # Codebooks 5-10: mid-level detail
        elif i < 20:
            w = 1.0    # Codebooks 11-20: fine detail
        else:
            w = 0.5    # Codebooks 21-31: ultra-fine texture
        weights.append(w)
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.mean()  # Normalize so mean = 1.0


class NaturalnessDataset(Dataset):
    """
    Combined dataset with per-source tracking for weighted sampling.
    """
    def __init__(self, max_seq_len: int = 1024, phase: int = 1):
        self.samples = []
        self.sample_weights = []
        self.max_seq_len = max_seq_len
        self.source_counts = {}

        for ds_name, ds_config in DATASET_CONFIGS.items():
            ds_path = ds_config["path"]

            # Phase 2: only use expresso_full (which contains ex04 speaker)
            if phase == 2 and ds_name != "expresso_full":
                continue

            if not ds_path.exists():
                logger.info(f"  SKIP {ds_name}: {ds_path} not found")
                continue

            samples_loaded = self._load_dataset(ds_name, ds_path, ds_config["weight"])
            self.source_counts[ds_name] = samples_loaded
            logger.info(f"  {ds_name}: {samples_loaded} samples (weight={ds_config['weight']:.1f})")

        logger.info(f"Total: {len(self.samples)} samples from {len(self.source_counts)} datasets")

    def _load_dataset(self, ds_name: str, ds_path: Path, weight: float) -> int:
        """Load samples from a single dataset source."""
        count = 0

        # Load from tokenized JSON format (all datasets use this now)
        json_path = ds_path / "train_tokenized.json"
        if not json_path.exists():
            logger.warning(f"No train_tokenized.json found in {ds_path}")
            return 0

        with open(json_path) as f:
            samples = json.load(f)

        for s in samples:
            if s.get("num_frames", 0) > self.max_seq_len:
                continue
            s["_data_dir"] = str(ds_path)
            s["_source"] = ds_name
            s["_format"] = "tokenized"
            self.samples.append(s)

            # Apply per-speaker weight boost (e.g., ex04 gets 1.5x)
            sample_weight = weight
            speaker = s.get("speaker", "")
            if speaker in SPEAKER_WEIGHT_BOOST:
                sample_weight *= SPEAKER_WEIGHT_BOOST[speaker]

            self.sample_weights.append(sample_weight)
            count += 1

        return count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data_dir = Path(sample["_data_dir"])

        tokens_path = data_dir / sample["tokens_path"]
        audio_tokens = torch.load(tokens_path, weights_only=True)

        return {
            "audio_tokens": audio_tokens,
            "text": sample.get("text", ""),
            "num_frames": audio_tokens.size(1),
            "source": sample.get("_source", "unknown"),
        }


class CombinedNaturalnessTrainer:
    """
    Train decoder-path for naturalness using all available datasets.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

        torch.cuda.empty_cache()
        gc.collect()

        self._init_tokenizer()
        self._init_model()
        self._freeze_and_setup()
        self._init_datasets()
        self._init_optimizer()

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epoch_losses = []

        # Codebook loss weights (CoVoC 2024 aggressive weighting)
        self.cb_weights = get_codebook_weights(self.num_codebooks).to(self.device)
        logger.info(f"Codebook weights (first 5): {self.cb_weights[:5].tolist()}")

        # Codebook utilization tracking (ERVQ: monitor for collapse)
        self.cb_usage_counts = torch.zeros(
            self.num_codebooks - 1, self.audio_vocab_size,
            dtype=torch.long, device='cpu'
        )

    def _init_tokenizer(self):
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing

        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(bos, self.text_tokenizer.bos_token_id),
                          (eos, self.text_tokenizer.eos_token_id)],
        )

    def _init_model(self):
        from models import Model

        logger.info("=" * 70)
        logger.info("CSM COMBINED NATURALNESS TRAINING")
        logger.info("=" * 70)

        self.model = Model.from_pretrained(self.args.model_name)
        self.dtype = torch.bfloat16
        self.model.to(device=self.device, dtype=self.dtype)

        self.num_codebooks = self.model.config.audio_num_codebooks
        self.audio_vocab_size = self.model.config.audio_vocab_size

        # Resume from checkpoint if specified
        if self.args.resume:
            self._load_checkpoint(self.args.resume)

        logger.info(f"Model: {self.args.model_name}")
        logger.info(f"Codebooks: {self.num_codebooks}, Audio vocab: {self.audio_vocab_size}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained decoder/projection/audio_head from checkpoint."""
        cp = Path(checkpoint_path)

        # Try merged model first
        merged = cp / "model_merged.pt"
        if merged.exists():
            logger.info(f"Loading merged model from {merged}")
            state = torch.load(merged, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            logger.info("Merged model loaded successfully")
            return

        # Try component files
        decoder_path = cp / "decoder.pt"
        proj_path = cp / "projection.pt"
        head_path = cp / "audio_head.pt"

        if decoder_path.exists():
            logger.info(f"Loading decoder from {decoder_path}")
            self.model.decoder.load_state_dict(
                torch.load(decoder_path, map_location=self.device, weights_only=True))

        if proj_path.exists():
            logger.info(f"Loading projection from {proj_path}")
            self.model.projection.load_state_dict(
                torch.load(proj_path, map_location=self.device, weights_only=True))

        if head_path.exists():
            logger.info(f"Loading audio_head from {head_path}")
            head = torch.load(head_path, map_location=self.device, weights_only=True)
            self.model.audio_head.data.copy_(head.data)

        logger.info("Checkpoint components loaded")

    def _freeze_and_setup(self):
        """Freeze backbone, unfreeze decoder path."""
        for param in self.model.parameters():
            param.requires_grad = False

        trainable = 0

        # Unfreeze decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        trainable += sum(p.numel() for p in self.model.decoder.parameters())

        # Unfreeze projection
        for param in self.model.projection.parameters():
            param.requires_grad = True
        trainable += sum(p.numel() for p in self.model.projection.parameters())

        # Unfreeze audio_head
        self.model.audio_head.requires_grad = True
        trainable += self.model.audio_head.numel()

        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
        logger.info(f"Backbone FROZEN | Codebook0_head FROZEN")

    def _init_datasets(self):
        """Initialize train/val datasets with weighted sampling."""
        self.train_dataset = NaturalnessDataset(
            max_seq_len=self.args.max_seq_len,
            phase=self.args.phase,
        )

        # Weighted sampler for balanced dataset representation
        weights = torch.tensor(self.train_dataset.sample_weights, dtype=torch.float64)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )

        # Validation: use a subset of samples from each source
        val_indices = []
        for ds_name in self.train_dataset.source_counts:
            # Get indices for this source
            source_indices = [
                i for i, s in enumerate(self.train_dataset.samples)
                if s.get("_source") == ds_name
            ]
            # Take up to 50 from each source for validation
            if len(source_indices) > 50:
                val_indices.extend(np.random.choice(source_indices, 50, replace=False).tolist())
            else:
                val_indices.extend(source_indices[:25])

        self.val_indices = val_indices
        logger.info(f"Train loader: {len(self.train_loader)} batches")
        logger.info(f"Validation: {len(self.val_indices)} samples")

    def _init_optimizer(self):
        """Initialize optimizer with cosine schedule."""
        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        steps_per_epoch = max(len(self.train_loader) // self.args.grad_accum, 1)
        total_steps = steps_per_epoch * self.args.epochs

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=self.args.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max=max(total_steps - self.args.warmup_steps, 1),
                                   eta_min=1e-7)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine],
                                      milestones=[self.args.warmup_steps])

        logger.info(f"Optimizer: AdamW, lr={self.args.lr}, total_steps={total_steps}")

    def _compute_decoder_loss(self, sample) -> Optional[torch.Tensor]:
        """
        Compute weighted loss on decoder outputs (codebooks 1-31).

        Architecture (verified against CSM source):
        - Position 0: projection(backbone_output)
        - Position 1: projection(codebook_0_embed) → predicts codebook 1
        - Position k: projection(codebook_{k-1}_embed) → predicts codebook k
        - Total: 32 positions with causal mask
        """
        audio_tokens = sample["audio_tokens"]
        if audio_tokens.dim() == 3:
            audio_tokens = audio_tokens.squeeze(0)
        audio_tokens = audio_tokens.to(self.device)
        text = sample["text"][0] if isinstance(sample["text"], (list, tuple)) else sample["text"]

        audio_len = audio_tokens.size(1)
        if audio_len < 5:
            return None

        # Tokenize text
        text_tokens = self.text_tokenizer.encode(f"[0]{text}")
        text_len = len(text_tokens)
        seq_len = text_len + audio_len

        if seq_len > self.args.max_seq_len:
            return None

        # Build backbone input
        tokens = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        token_mask = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)

        tokens[0, :text_len, -1] = torch.tensor(text_tokens, device=self.device)
        token_mask[0, :text_len, -1] = True
        tokens[0, text_len:, :-1] = audio_tokens.transpose(0, 1)
        token_mask[0, text_len:, :-1] = True

        # Backbone forward (FROZEN, no gradient)
        with torch.no_grad():
            embeds = self.model._embed_tokens(tokens)
            masked_embeds = embeds * token_mask.unsqueeze(-1)
            h = masked_embeds.sum(dim=2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
            backbone_h = self.model.backbone(h, mask=causal_mask.unsqueeze(0)).to(dtype=self.dtype)

        # Decoder loss WITH gradients
        total_loss = 0.0
        num_losses = 0

        # Sample frames
        num_frames = min(audio_len - 1, self.args.frames_per_sample)
        frame_indices = np.random.choice(audio_len - 1, size=num_frames, replace=False)

        for frame_idx in frame_indices:
            try:
                h_frame = backbone_h[0, text_len + frame_idx, :]

                # Targets: codebooks 1-31 at next frame
                targets = audio_tokens[1:, frame_idx + 1]

                # Build decoder input: [backbone_h, c0_embed, ..., c30_embed]
                input_codes = audio_tokens[:self.num_codebooks - 1, frame_idx + 1]

                code_embeds = []
                for cb in range(self.num_codebooks - 1):
                    offset = input_codes[cb] + cb * self.audio_vocab_size
                    emb = self.model.audio_embeddings(offset.unsqueeze(0))
                    code_embeds.append(emb)

                code_embeds = torch.stack(code_embeds, dim=1)  # [1, 31, backbone_dim]

                decoder_seq = torch.cat([
                    h_frame.unsqueeze(0).unsqueeze(0),
                    code_embeds
                ], dim=1)  # [1, 32, backbone_dim]

                decoder_input = self.model.projection(decoder_seq)  # [1, 32, decoder_dim]

                seq_size = self.num_codebooks  # 32
                decoder_mask = torch.tril(torch.ones(
                    seq_size, seq_size, dtype=torch.bool, device=self.device
                )).unsqueeze(0)

                decoder_out = self.model.decoder(decoder_input, mask=decoder_mask)

                # Weighted loss per codebook
                frame_loss = 0.0
                for cb in range(self.num_codebooks - 1):
                    logits = torch.mm(
                        decoder_out[0, cb + 1, :].unsqueeze(0),
                        self.model.audio_head[cb].to(decoder_out.dtype)
                    )
                    target = targets[cb]
                    cb_loss = F.cross_entropy(logits, target.unsqueeze(0), ignore_index=0)
                    frame_loss += cb_loss * self.cb_weights[cb]

                    # Track codebook utilization (ERVQ monitoring)
                    pred_token = logits.argmax(dim=-1).item()
                    self.cb_usage_counts[cb, pred_token] += 1

                total_loss += frame_loss / (self.num_codebooks - 1)
                num_losses += 1

            except Exception as e:
                logger.debug(f"Frame {frame_idx} error: {e}")
                continue

        if num_losses > 0:
            return total_loss / num_losses
        return None

    def _log_codebook_utilization(self):
        """Log codebook utilization to detect collapse (ERVQ monitoring)."""
        total_preds = self.cb_usage_counts.sum(dim=1)  # per codebook
        used_codes = (self.cb_usage_counts > 0).sum(dim=1).float()
        utilization = used_codes / self.audio_vocab_size * 100

        # Log summary: first 4, middle, last
        cb_util_str = ", ".join([f"cb{i+1}={utilization[i]:.0f}%" for i in [0, 1, 2, 3, 15, 30]])
        avg_util = utilization.mean().item()
        min_util = utilization.min().item()
        logger.info(f"Codebook utilization: avg={avg_util:.0f}% min={min_util:.0f}% [{cb_util_str}]")

        if min_util < 50:
            logger.warning(f"LOW codebook utilization detected! min={min_util:.0f}% - risk of collapse")

        # Reset counters
        self.cb_usage_counts.zero_()

    def train_epoch(self, epoch: int):
        """Train one epoch with source tracking."""
        self.model.decoder.train()
        self.model.projection.train()

        total_loss = 0.0
        num_steps = 0
        self.optimizer.zero_grad()
        accum_loss = 0.0
        source_losses: Dict[str, List[float]] = {}

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_decoder_loss(batch)

                if loss is not None and torch.isfinite(loss):
                    (loss / self.args.grad_accum).backward()
                    accum_loss += loss.item()

                    # Track per-source loss
                    source = batch.get("source", ["unknown"])
                    source = source[0] if isinstance(source, (list, tuple)) else source
                    source_losses.setdefault(source, []).append(loss.item())

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logger.warning(f"OOM at batch {batch_idx}")
                continue

            if (batch_idx + 1) % self.args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accum_loss
                num_steps += 1

                if self.global_step % self.args.log_steps == 0:
                    avg_loss = accum_loss / self.args.grad_accum
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.4f}")

                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                # Log codebook utilization every 100 steps (ERVQ monitoring)
                if self.global_step % 100 == 0:
                    self._log_codebook_utilization()

                accum_loss = 0.0

        # Log per-source losses
        for source, losses in source_losses.items():
            avg = sum(losses) / len(losses) if losses else 0
            logger.info(f"  {source}: avg_loss={avg:.4f} ({len(losses)} samples)")

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation subset from all sources."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        source_losses: Dict[str, List[float]] = {}

        for idx in self.val_indices:
            try:
                sample = self.train_dataset[idx]
                # Wrap in batch format
                batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                        for k, v in sample.items()}

                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_decoder_loss(batch)

                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    num_samples += 1
                    source = sample.get("source", "unknown")
                    source_losses.setdefault(source, []).append(loss.item())
            except:
                continue

        # Log per-source val losses
        for source, losses in source_losses.items():
            avg = sum(losses) / len(losses) if losses else 0
            logger.info(f"  Val {source}: {avg:.4f} ({len(losses)} samples)")

        return total_loss / max(num_samples, 1)

    def save_checkpoint(self, name: str):
        output_dir = Path(self.args.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.decoder.state_dict(), output_dir / "decoder.pt")
        torch.save(self.model.projection.state_dict(), output_dir / "projection.pt")
        torch.save(self.model.audio_head, output_dir / "audio_head.pt")

        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "phase": self.args.phase,
            "datasets": list(self.train_dataset.source_counts.keys()),
            "dataset_sizes": self.train_dataset.source_counts,
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved: {output_dir}")

    def save_merged(self, name: str = "best_model_merged"):
        output_dir = Path(self.args.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), output_dir / "model_merged.pt")
        logger.info(f"Merged model saved: {output_dir / 'model_merged.pt'}")

    def train(self):
        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE {self.args.phase} TRAINING")
        logger.info(f"Datasets: {list(self.train_dataset.source_counts.keys())}")
        logger.info(f"Total samples: {len(self.train_dataset)}")
        logger.info(f"Epochs: {self.args.epochs}, LR: {self.args.lr}")
        logger.info("=" * 70)

        for epoch in range(self.args.epochs):
            logger.info(f"\n{'='*60}\nEPOCH {epoch + 1}/{self.args.epochs}\n{'='*60}")

            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {epoch_time/60:.1f}min")
            self.epoch_losses.append({"epoch": epoch+1, "train": train_loss, "val": val_loss})

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model")
                self.save_merged("best_model_merged")
                logger.info("*** New best model! ***")

            torch.cuda.empty_cache()

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info("=" * 70)

        # Save loss history
        with open(Path(self.args.output_dir) / "loss_history.json", "w") as f:
            json.dump(self.epoch_losses, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="CSM Combined Naturalness Training")
    parser.add_argument("--model-name", default="sesame/csm-1b")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                       help="Phase 1: all data, Phase 2: Maya voice focus")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--grad-accum", type=int, default=8)  # Effective batch ~8 (Speechmatics recommends 8-32)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    # Compute amortization: Speechmatics trains on 1/16 of frames
    # For 24K samples, use fewer frames to prevent overfitting
    parser.add_argument("--frames-per-sample", type=int, default=16)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=str,
                       default=str(CHECKPOINT_DIR / "csm_maya_combined_naturalness"))

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(args.gpu).total_mem / 1e9:.1f}GB"
                if hasattr(torch.cuda.get_device_properties(args.gpu), 'total_mem')
                else f"VRAM: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f}GB")

    trainer = CombinedNaturalnessTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
