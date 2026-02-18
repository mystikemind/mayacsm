#!/usr/bin/env python3
"""
Step 4: Fine-tune CSM-1B for Maya Voice - PRODUCTION-LEVEL

This script fine-tunes the Sesame CSM-1B model on the preprocessed single-speaker
dataset to achieve Maya-level voice quality.

Key Architecture Understanding:
- CSM has a backbone (Llama 3.2 1B) that processes text+audio and predicts codebook 0
- CSM has a decoder (Llama 3.2 100M) that autoregressively predicts codebooks 1-31
- For training, we use teacher forcing on both backbone and decoder
- No KV caches needed during training - full sequence forward pass

Training Strategy (Based on Sesame Research):
- Full fine-tuning (NOT LoRA) for best quality
- Conservative learning rate (3e-5) for audio models
- Long context (2048 tokens) for conversational dependencies
- Gradient accumulation for effective batch size
- BF16 mixed precision for memory efficiency
- Cosine learning rate schedule with warmup

Usage:
    python 04_train_csm.py --data data/csm_ready_talia

Output:
    checkpoints/csm_maya/
    ├── checkpoint-{step}/
    ├── best_model/
    └── training_log.json
"""

import os
import sys
import json
import logging
import argparse
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for CSM fine-tuning."""

    # Model
    model_name: str = "sesame/csm-1b"
    use_bf16: bool = True

    # Data
    data_dir: str = ""
    max_audio_duration: float = 10.0  # Reduced for memory efficiency

    # Training
    num_epochs: int = 25
    batch_size: int = 1  # Single sample for memory efficiency
    gradient_accumulation_steps: int = 16  # Effective batch 16
    learning_rate: float = 3e-5  # Conservative for audio
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0

    # Saving
    save_steps: int = 200
    eval_steps: int = 100
    save_total_limit: int = 3
    output_dir: str = str(CHECKPOINT_DIR / "csm_maya")

    # Logging
    log_steps: int = 10


class CSMDataset(Dataset):
    """Dataset for CSM fine-tuning."""

    def __init__(
        self,
        metadata_path: Path,
        data_dir: Path,
        max_audio_duration: float = 15.0,
        sample_rate: int = 24000,
    ):
        with open(metadata_path) as f:
            self.samples = json.load(f)

        self.data_dir = data_dir
        self.max_audio_duration = max_audio_duration
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_duration * sample_rate)

        # Filter out samples that are too long
        original_len = len(self.samples)
        self.samples = [
            s for s in self.samples
            if s.get("duration", 0) <= max_audio_duration
        ]
        logger.info(f"Dataset: {len(self.samples)} samples (filtered {original_len - len(self.samples)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio_path = self.data_dir / sample["path"]

        try:
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
        except:
            import torchaudio
            audio, sr = torchaudio.load(str(audio_path))
            audio = audio.squeeze(0).numpy()

        # Ensure float32
        audio = audio.astype(np.float32)

        # Truncate if needed
        if len(audio) > self.max_audio_samples:
            audio = audio[:self.max_audio_samples]

        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "text": sample.get("text", ""),
            "duration": len(audio) / self.sample_rate,
        }


def collate_fn(batch):
    """Collate batch - just return list for per-sample processing."""
    return batch


def _create_causal_mask(seq_len: int, device: torch.device):
    """Create causal attention mask."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


class CSMTrainer:
    """Full fine-tuning trainer for CSM-1B."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self._init_tokenizers()
        self._init_model()
        self._init_datasets()
        self._init_optimizer()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.training_log = []

    def _init_tokenizers(self):
        """Initialize text and audio tokenizers."""
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing
        from moshi.models import loaders
        from huggingface_hub import hf_hub_download

        logger.info("Loading tokenizers...")

        # Clear GPU cache first
        torch.cuda.empty_cache()
        gc.collect()

        # Text tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", self.text_tokenizer.bos_token_id), (f"{eos}", self.text_tokenizer.eos_token_id)],
        )

        # Audio tokenizer (Mimi) - load to CPU first then move to device
        logger.info("Loading Mimi audio tokenizer...")
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.audio_tokenizer = loaders.get_mimi(mimi_weight, device="cpu")
        self.audio_tokenizer.set_num_codebooks(32)
        # Move to GPU after initialization
        self.audio_tokenizer = self.audio_tokenizer.to(self.device)

        logger.info("Tokenizers loaded!")

    def _init_model(self):
        """Initialize CSM model for fine-tuning."""
        logger.info("=" * 60)
        logger.info("INITIALIZING CSM-1B FOR FINE-TUNING")
        logger.info("=" * 60)

        from models import Model

        logger.info(f"Loading model: {self.config.model_name}")
        self.model = Model.from_pretrained(self.config.model_name)

        # Move to device
        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

        # Store model config
        self.audio_vocab_size = self.model.config.audio_vocab_size
        self.num_codebooks = self.model.config.audio_num_codebooks

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # IMPORTANT: Do NOT setup caches - we train without them
        # The backbone/decoder can run without caches using full causal masks

        # Create permanent causal masks for max sequence lengths
        max_backbone_len = 2048
        max_decoder_len = self.num_codebooks
        self.backbone_mask = _create_causal_mask(max_backbone_len, self.device)
        self.decoder_mask = _create_causal_mask(max_decoder_len, self.device)

        # Set training mode
        self.model.train()

    def _init_datasets(self):
        """Initialize training and validation datasets."""
        data_dir = Path(self.config.data_dir)

        logger.info(f"Loading datasets from: {data_dir}")

        # Training dataset
        train_path = data_dir / "train.json"
        self.train_dataset = CSMDataset(
            train_path,
            data_dir,
            max_audio_duration=self.config.max_audio_duration,
        )
        logger.info(f"Train samples: {len(self.train_dataset)}")

        # Validation dataset
        val_path = data_dir / "val.json"
        self.val_dataset = CSMDataset(
            val_path,
            data_dir,
            max_audio_duration=self.config.max_audio_duration,
        )
        logger.info(f"Val samples: {len(self.val_dataset)}")

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        # Use regular AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs

        # Cosine scheduler with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - self.config.warmup_steps, 1),
            eta_min=self.config.learning_rate * 0.01,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )

        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {self.config.warmup_steps}")

    def _prepare_sample(self, sample):
        """Prepare a single sample for training.

        Returns:
            tokens: (seq_len, num_codebooks+1) - last dim is text
            token_mask: (seq_len, num_codebooks+1) - which tokens are valid
            audio_start_idx: index where audio starts (after text)
        """
        audio = sample["audio"].to(self.device)
        text = sample["text"]

        # Tokenize text
        text_tokens = self.text_tokenizer.encode(f"[0]{text}")
        text_len = len(text_tokens)

        # Create text frames
        text_frame = torch.zeros(text_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        text_mask = torch.zeros(text_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_mask[:, -1] = True

        # Tokenize audio
        with torch.no_grad():
            audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]  # (32, num_frames)

        audio_len = audio_tokens.size(1)

        # Create audio frames
        audio_frame = torch.zeros(audio_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        audio_mask = torch.zeros(audio_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)  # (num_frames, 32)
        audio_mask[:, :-1] = True

        # Concatenate
        tokens = torch.cat([text_frame, audio_frame], dim=0)
        token_mask = torch.cat([text_mask, audio_mask], dim=0)

        return tokens, token_mask, text_len, audio_tokens

    def _forward_backbone(self, tokens, token_mask, seq_len):
        """Forward pass through backbone without KV caches.

        Args:
            tokens: (batch=1, seq_len, num_codebooks+1)
            token_mask: (batch=1, seq_len, num_codebooks+1)
            seq_len: sequence length

        Returns:
            h: (batch=1, seq_len, hidden_dim) backbone hidden states
        """
        # Embed tokens
        embeds = self.model._embed_tokens(tokens)  # (1, seq_len, num_codebooks+1, hidden_dim)

        # Apply mask and sum across codebooks
        masked_embeds = embeds * token_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)  # (1, seq_len, hidden_dim)

        # Get causal mask for this sequence length
        causal_mask = self.backbone_mask[:seq_len, :seq_len].unsqueeze(0)  # (1, seq_len, seq_len)

        # Forward through backbone (without caches - just use mask)
        h = self.model.backbone(h, mask=causal_mask)

        return h.to(dtype=self.dtype)

    def _forward_decoder_with_teacher_forcing(self, backbone_h, audio_tokens, frame_idx):
        """Forward decoder for one audio frame with teacher forcing.

        Args:
            backbone_h: (1, hidden_dim) backbone hidden state for this frame
            audio_tokens: (32,) ground truth audio tokens for this frame
            frame_idx: which audio frame we're processing

        Returns:
            logits: list of (1, vocab_size) logits for each codebook
        """
        logits_list = []

        # Codebook 0: predicted directly from backbone
        c0_logits = self.model.codebook0_head(backbone_h)  # (1, vocab_size)
        logits_list.append(c0_logits)

        # Use ground truth c0 for teacher forcing
        c0_embed = self.model._embed_audio(0, audio_tokens[0:1].unsqueeze(0))  # (1, 1, hidden_dim)

        # Initialize decoder input: backbone_h + c0_embed
        curr_h = torch.cat([backbone_h.unsqueeze(1), c0_embed], dim=1)  # (1, 2, hidden_dim)

        # Project for decoder
        curr_h = self.model.projection(curr_h)

        # Codebooks 1-31: predicted by decoder
        for i in range(1, self.num_codebooks):
            # Get causal mask for decoder
            decoder_len = curr_h.size(1)
            decoder_mask = self.decoder_mask[:decoder_len, :decoder_len].unsqueeze(0)

            # Forward through decoder
            decoder_h = self.model.decoder(curr_h, mask=decoder_mask).to(dtype=self.dtype)

            # Get logits for codebook i
            ci_logits = torch.mm(decoder_h[:, -1, :], self.model.audio_head[i - 1])  # (1, vocab_size)
            logits_list.append(ci_logits)

            # Teacher forcing: use ground truth token for next step
            if i < self.num_codebooks - 1:
                ci_embed = self.model._embed_audio(i, audio_tokens[i:i+1].unsqueeze(0))
                ci_embed_proj = self.model.projection(ci_embed)
                curr_h = torch.cat([curr_h, ci_embed_proj], dim=1)

        return logits_list

    def _compute_loss(self, sample):
        """Compute training loss for a single sample.

        The loss is cross-entropy over:
        1. Codebook 0 predictions from backbone
        2. Codebook 1-31 predictions from decoder (with teacher forcing)
        """
        # Prepare sample
        tokens, token_mask, text_len, audio_tokens = self._prepare_sample(sample)

        seq_len = tokens.size(0)
        audio_len = audio_tokens.size(1)

        if audio_len < 2:
            # Not enough audio to train on
            return None

        # Add batch dimension
        tokens = tokens.unsqueeze(0)  # (1, seq_len, 33)
        token_mask = token_mask.unsqueeze(0)  # (1, seq_len, 33)

        # Forward through backbone
        h = self._forward_backbone(tokens, token_mask, seq_len)  # (1, seq_len, hidden_dim)

        # Compute loss for each audio frame
        total_loss = 0.0
        num_frames = 0

        # For each audio frame (after text), predict the next audio frame
        for t in range(audio_len - 1):
            # Position in full sequence (text + audio frames so far)
            pos = text_len + t

            # Get backbone hidden state for predicting next frame
            backbone_h = h[:, pos, :]  # (1, hidden_dim)

            # Target audio tokens for next frame
            target = audio_tokens[:, t + 1]  # (32,)

            # Get logits for all codebooks
            logits_list = self._forward_decoder_with_teacher_forcing(
                backbone_h,
                audio_tokens[:, t],  # Current frame for teacher forcing context
                t
            )

            # Compute cross-entropy loss for each codebook
            frame_loss = 0.0
            for cb, logits in enumerate(logits_list):
                cb_target = target[cb].unsqueeze(0)
                cb_loss = F.cross_entropy(logits, cb_target)
                frame_loss += cb_loss

            total_loss += frame_loss / self.num_codebooks
            num_frames += 1

        if num_frames > 0:
            return total_loss / num_frames
        else:
            return None

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            batch_loss = 0.0
            valid_samples = 0

            for sample in batch:
                try:
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        loss = self._compute_loss(sample)

                    if loss is not None:
                        loss = loss / self.config.gradient_accumulation_steps
                        loss.backward()
                        batch_loss += loss.item() * self.config.gradient_accumulation_steps
                        valid_samples += 1

                except Exception as e:
                    logger.debug(f"Sample error: {e}")
                    continue

            if valid_samples > 0:
                accumulated_loss += batch_loss / valid_samples

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accumulated_loss
                num_batches += 1

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e}"
                    )

                    self.training_log.append({
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": lr,
                    })

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate()
                    logger.info(f"Step {self.global_step} | Validation Loss: {val_loss:.4f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model")
                        logger.info(f"New best model saved! (val_loss: {val_loss:.4f})")

                # Checkpoint saving
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
            for sample in batch:
                try:
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        loss = self._compute_loss(sample)

                    if loss is not None:
                        total_loss += loss.item()
                        num_samples += 1

                except Exception as e:
                    continue

        self.model.train()
        return total_loss / max(num_samples, 1)

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), output_dir / "model.pt")

        # Save optimizer
        torch.save(self.optimizer.state_dict(), output_dir / "optimizer.pt")

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save training log
        with open(output_dir / "training_log.json", "w") as f:
            json.dump(self.training_log, f, indent=2)

        logger.info(f"Checkpoint saved to {output_dir}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space."""
        output_dir = Path(self.config.output_dir)
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )

        # Keep only the most recent N checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("STARTING CSM FINE-TUNING (PRODUCTION-LEVEL)")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")

            epoch_start = time.time()
            epoch_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch + 1} complete. Loss: {epoch_loss:.4f} | Time: {epoch_time/60:.1f}min")

            # Save checkpoint at end of epoch
            self.save_checkpoint(f"epoch-{epoch + 1}")

            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()

        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Checkpoints saved to: {self.config.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSM-1B for Maya voice")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed data directory"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "csm_maya"))
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        data_dir=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    # Validate data directory
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    if not (data_dir / "train.json").exists():
        logger.error(f"train.json not found in {data_dir}")
        sys.exit(1)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)

    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize trainer
    trainer = CSMTrainer(config)

    # Resume if checkpoint provided
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            state = torch.load(checkpoint_path / "model.pt", map_location=trainer.device)
            trainer.model.load_state_dict(state)

            state_file = checkpoint_path / "training_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    training_state = json.load(f)
                trainer.global_step = training_state["global_step"]
                trainer.epoch = training_state["epoch"]
                trainer.best_val_loss = training_state["best_val_loss"]

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
