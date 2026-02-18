#!/usr/bin/env python3
"""
CSM Fine-Tuning - Production Grade v2
======================================

This script implements the CORRECT fine-tuning approach based on:
1. Speechmatics fine-tuning guide (https://blog.speechmatics.com/sesame-finetune)
2. Sesame AI research paper
3. sesame-finetune repository (https://github.com/knottwill/sesame-finetune)

KEY DIFFERENCES FROM PREVIOUS ATTEMPTS:
1. Trains BOTH backbone AND decoder (previous only trained backbone)
2. Uses compute amortization: decoder on 1/16 of frames
3. Combined loss: backbone_loss + decoder_loss with proper weighting
4. Correct hyperparameters: LR 3e-5, 25 epochs, weight decay 0.002

This is the approach Sesame engineers used internally.
"""

import os
import sys
import json
import logging
import argparse
import time
import gc
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

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
class ProductionConfig:
    """
    Production training configuration based on Sesame research.

    Key hyperparameters from Speechmatics guide:
    - learning_rate: 3e-5 (optimal from sweeps)
    - weight_decay: 0.002
    - decoder_loss_weight: 0.5 (equal backbone/decoder)
    - epochs: 25+
    """
    model_name: str = "sesame/csm-1b"
    use_bf16: bool = True
    data_dir: str = ""
    max_frames: int = 250  # ~10 seconds of audio

    # Training hyperparameters (from Speechmatics guide)
    num_epochs: int = 25
    batch_size: int = 1  # Small batch, use gradient accumulation
    gradient_accumulation_steps: int = 8  # Effective batch = 8
    learning_rate: float = 3e-5  # Optimal from Speechmatics
    min_lr: float = 1e-6
    weight_decay: float = 0.002  # From Speechmatics
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # Loss configuration (CRITICAL)
    decoder_loss_weight: float = 0.5  # Equal weighting
    decoder_frame_ratio: float = 1/16  # Compute amortization

    # Train BOTH backbone and decoder (CRITICAL FIX)
    train_backbone: bool = True
    train_decoder: bool = True

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 5
    output_dir: str = str(CHECKPOINT_DIR / "csm_maya_production_v2")
    log_steps: int = 10

    # Audio generation for quality monitoring
    generate_every_steps: int = 500
    generation_texts: List[str] = None

    def __post_init__(self):
        if self.generation_texts is None:
            self.generation_texts = [
                "oh wow thats amazing i love it",
                "hmm let me think about that for a moment",
                "aww im so sorry to hear that",
                "yes that makes sense to me",
                "wait what do you mean by that",
            ]


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
        logger.info(f"Dataset: {len(self.samples)} samples (filtered {original - len(self.samples)} > {max_frames} frames)")

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


class CSMProductionTrainer:
    """
    Production trainer that correctly trains BOTH backbone AND decoder.

    Loss formula (from Sesame research):
        total_loss = (1 - decoder_loss_weight) * backbone_loss + decoder_loss_weight * decoder_loss

    Where:
        - backbone_loss: CrossEntropy on codebook 0 for ALL frames
        - decoder_loss: CrossEntropy on codebooks 1-31 for 1/16 of frames (randomly sampled)
    """

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.cuda.empty_cache()
        gc.collect()

        self._init_text_tokenizer()
        self._init_model()
        self._setup_training()
        self._init_datasets()
        self._init_optimizer()

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.training_log = []

    def _init_text_tokenizer(self):
        """Initialize text tokenizer."""
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
        logger.info("=" * 70)
        logger.info("CSM PRODUCTION FINE-TUNING v2")
        logger.info("Training: BACKBONE + DECODER (correct approach)")
        logger.info("=" * 70)

        from models import Model

        logger.info(f"Loading model: {self.config.model_name}")
        self.model = Model.from_pretrained(self.config.model_name)

        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

        self.audio_vocab_size = self.model.config.audio_vocab_size
        self.num_codebooks = self.model.config.audio_num_codebooks

        total_params = sum(p.numel() for p in self.model.parameters())
        backbone_params = sum(p.numel() for p in self.model.backbone.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Backbone parameters: {backbone_params:,}")
        logger.info(f"Decoder parameters: {decoder_params:,}")

        # Create masks
        max_backbone_len = 2048
        self.backbone_mask = _create_causal_mask(max_backbone_len, self.device)

    def _setup_training(self):
        """Setup which components to train."""
        # Enable gradients for all parameters we want to train
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_params = 0

        if self.config.train_backbone:
            logger.info("Training: Backbone (all layers)")
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            trainable_params += sum(p.numel() for p in self.model.backbone.parameters())

            # Also train backbone output head (codebook 0)
            for param in self.model.codebook0_head.parameters():
                param.requires_grad = True
            trainable_params += sum(p.numel() for p in self.model.codebook0_head.parameters())

        if self.config.train_decoder:
            logger.info("Training: Decoder (all layers)")
            for param in self.model.decoder.parameters():
                param.requires_grad = True
            trainable_params += sum(p.numel() for p in self.model.decoder.parameters())

        # Always train embeddings
        for param in self.model.text_embeddings.parameters():
            param.requires_grad = True
        trainable_params += sum(p.numel() for p in self.model.text_embeddings.parameters())

        for param in self.model.audio_embeddings.parameters():
            param.requires_grad = True
        trainable_params += sum(p.numel() for p in self.model.audio_embeddings.parameters())

        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Decoder loss weight: {self.config.decoder_loss_weight}")
        logger.info(f"Decoder frame ratio: {self.config.decoder_frame_ratio}")

    def _init_datasets(self):
        """Initialize datasets."""
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
            num_workers=0,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def _init_optimizer(self):
        """Initialize optimizer with proper hyperparameters."""
        # Collect all trainable parameters
        params_to_train = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            params_to_train,
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

        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Weight decay: {self.config.weight_decay}")

    def _compute_combined_loss(self, sample) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Compute combined backbone + decoder loss.

        This is the CORRECT loss function based on Sesame's research:
        - Backbone loss: CrossEntropy on codebook 0 for ALL frames
        - Decoder loss: CrossEntropy on codebooks 1-31 for 1/16 of frames

        Returns:
            total_loss: Combined loss tensor
            metrics: Dict with individual loss components
        """
        audio_tokens = sample["audio_tokens"]
        if audio_tokens.dim() == 3:
            audio_tokens = audio_tokens.squeeze(0)
        audio_tokens = audio_tokens.to(self.device)  # (32, num_frames)

        text = sample["text"]
        if isinstance(text, (list, tuple)):
            text = text[0]

        # Tokenize text
        text_tokens = self.text_tokenizer.encode(f"[0]{text}")
        text_len = len(text_tokens)
        audio_len = audio_tokens.size(1)

        if audio_len < 3:
            return None, {}

        # Build input sequence
        seq_len = text_len + audio_len
        tokens = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        token_mask = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)

        # Text tokens
        tokens[0, :text_len, -1] = torch.tensor(text_tokens, device=self.device)
        token_mask[0, :text_len, -1] = True

        # Audio tokens
        tokens[0, text_len:, :-1] = audio_tokens.transpose(0, 1)
        token_mask[0, text_len:, :-1] = True

        # Embed tokens
        embeds = self.model._embed_tokens(tokens)
        masked_embeds = embeds * token_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)

        # ===== BACKBONE FORWARD =====
        causal_mask = self.backbone_mask[:seq_len, :seq_len].unsqueeze(0)
        backbone_h = self.model.backbone(h, mask=causal_mask).to(dtype=self.dtype)

        # ===== BACKBONE LOSS (codebook 0, ALL frames) =====
        # Predict next frame's codebook 0 from each audio position
        audio_h = backbone_h[0, text_len:text_len + audio_len - 1, :]  # (audio_len-1, dim)
        c0_targets = audio_tokens[0, 1:audio_len]  # (audio_len-1,)
        c0_logits = self.model.codebook0_head(audio_h)  # (audio_len-1, vocab)
        backbone_loss = F.cross_entropy(c0_logits, c0_targets)

        # ===== DECODER LOSS (codebooks 1-31, 1/16 of frames) =====
        decoder_loss = torch.tensor(0.0, device=self.device)

        if self.config.train_decoder and audio_len > 2:
            # Sample 1/16 of frames for decoder training (compute amortization)
            num_decoder_frames = max(1, int(audio_len * self.config.decoder_frame_ratio))
            decoder_frame_indices = sorted(random.sample(range(audio_len - 1), min(num_decoder_frames, audio_len - 1)))

            if decoder_frame_indices:
                # For each sampled frame, predict codebooks 1-31
                decoder_losses = []

                for frame_idx in decoder_frame_indices:
                    # Get backbone hidden state for this frame
                    frame_h = backbone_h[0, text_len + frame_idx, :].unsqueeze(0).unsqueeze(0)  # (1, 1, dim)

                    # Get c0 token for this frame (input to decoder)
                    c0_token = audio_tokens[0, frame_idx].unsqueeze(0).unsqueeze(0)  # (1, 1)

                    # Decoder predicts codebooks 1-31
                    # The decoder takes backbone hidden state and predicts acoustic codebooks
                    try:
                        decoder_out = self.model.decoder(
                            frame_h,
                            c0_token,
                        )  # (1, 1, num_codebooks-1, vocab)

                        # Target: codebooks 1-31 for next frame
                        if frame_idx + 1 < audio_len:
                            targets = audio_tokens[1:, frame_idx + 1]  # (31,)

                            # Reshape decoder output
                            decoder_logits = decoder_out.squeeze(0).squeeze(0)  # (31, vocab)

                            # Cross-entropy for each codebook
                            frame_loss = F.cross_entropy(decoder_logits, targets)
                            decoder_losses.append(frame_loss)
                    except Exception as e:
                        # Skip frames that cause issues
                        continue

                if decoder_losses:
                    decoder_loss = torch.stack(decoder_losses).mean()

        # ===== COMBINED LOSS =====
        total_loss = (1 - self.config.decoder_loss_weight) * backbone_loss + \
                     self.config.decoder_loss_weight * decoder_loss

        metrics = {
            "backbone_loss": backbone_loss.item(),
            "decoder_loss": decoder_loss.item() if isinstance(decoder_loss, torch.Tensor) else decoder_loss,
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics

    def train_epoch(self):
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        total_backbone_loss = 0.0
        total_decoder_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_backbone = 0.0
        accumulated_decoder = 0.0
        batch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss, metrics = self._compute_combined_loss(batch)

                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    accumulated_loss += metrics["total_loss"]
                    accumulated_backbone += metrics["backbone_loss"]
                    accumulated_decoder += metrics["decoder_loss"]

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"Batch {batch_idx}: OOM")
                    torch.cuda.empty_cache()
                continue

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accumulated_loss
                total_backbone_loss += accumulated_backbone
                total_decoder_loss += accumulated_decoder
                num_batches += 1

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0
                    avg_backbone = total_backbone_loss / num_batches if num_batches > 0 else 0
                    avg_decoder = total_decoder_loss / num_batches if num_batches > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - batch_start_time

                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {accumulated_loss/self.config.gradient_accumulation_steps:.4f} "
                        f"(B:{accumulated_backbone/self.config.gradient_accumulation_steps:.4f} "
                        f"D:{accumulated_decoder/self.config.gradient_accumulation_steps:.4f}) | "
                        f"LR: {lr:.2e} | Grad: {grad_norm:.2f}"
                    )

                    self.training_log.append({
                        "step": self.global_step,
                        "total_loss": accumulated_loss / self.config.gradient_accumulation_steps,
                        "backbone_loss": accumulated_backbone / self.config.gradient_accumulation_steps,
                        "decoder_loss": accumulated_decoder / self.config.gradient_accumulation_steps,
                        "lr": lr,
                    })
                    batch_start_time = time.time()

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate()
                    logger.info(f"Step {self.global_step} | Val Loss: {val_loss:.4f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model")
                        logger.info(f"New best model! (val_loss: {val_loss:.4f})")

                # Checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                # Audio generation for quality monitoring
                if self.global_step % self.config.generate_every_steps == 0:
                    self.generate_samples()

                accumulated_loss = 0.0
                accumulated_backbone = 0.0
                accumulated_decoder = 0.0

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
                    loss, _ = self._compute_combined_loss(batch)
                if loss is not None and not torch.isnan(loss):
                    total_loss += loss.item()
                    num_samples += 1
            except:
                continue

        self.model.train()
        return total_loss / max(num_samples, 1)

    @torch.no_grad()
    def generate_samples(self):
        """Generate audio samples to monitor quality."""
        self.model.eval()

        output_dir = Path(self.config.output_dir) / "generations" / f"step-{self.global_step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating samples at step {self.global_step}...")

        try:
            from generator import Generator
            import scipy.io.wavfile as wav

            # Create generator with current model weights
            generator = Generator(self.model)

            for i, text in enumerate(self.config.generation_texts[:3]):  # Generate 3 samples
                try:
                    audio = generator.generate(
                        text=text,
                        speaker=0,
                        context=[],
                        max_audio_length_ms=5000,
                        temperature=0.8,
                        topk=50,
                    )

                    # Save audio
                    audio_np = audio.cpu().numpy()
                    wav.write(
                        str(output_dir / f"sample_{i}.wav"),
                        24000,
                        (audio_np * 32767).astype(np.int16)
                    )
                    logger.info(f"  Generated: {text[:30]}...")
                except Exception as e:
                    logger.warning(f"  Failed to generate sample {i}: {e}")
        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")

        self.model.train()

    def save_checkpoint(self, name: str):
        """Save checkpoint with full model weights."""
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full model (for production use)
        torch.save(self.model.state_dict(), output_dir / "model.pt")

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "training_log": self.training_log[-100:],  # Last 100 entries
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {output_dir}")

    def train(self):
        """Main training loop."""
        logger.info("=" * 70)
        logger.info("STARTING PRODUCTION TRAINING")
        logger.info("=" * 70)
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Weight decay: {self.config.weight_decay}")
        logger.info(f"Decoder loss weight: {self.config.decoder_loss_weight}")
        logger.info("=" * 70)

        start_time = time.time()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*70}\nEPOCH {epoch + 1}/{self.config.num_epochs}\n{'='*70}")

            epoch_start = time.time()
            epoch_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Time: {epoch_time/60:.1f}min")
            self.save_checkpoint(f"epoch-{epoch + 1}")

            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CSM Production Fine-Tuning v2")
    parser.add_argument("--data", type=str, required=True, help="Path to tokenized data directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.002, help="Weight decay")
    parser.add_argument("--decoder-loss-weight", type=float, default=0.5, help="Decoder loss weight (0-1)")
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "csm_maya_production_v2"))
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    config = ProductionConfig(
        data_dir=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        decoder_loss_weight=args.decoder_loss_weight,
        output_dir=args.output,
    )

    # Verify data
    data_dir = Path(args.data)
    if not (data_dir / "train_tokenized.json").exists():
        logger.error("Pre-tokenized data not found.")
        logger.error(f"Looking for: {data_dir / 'train_tokenized.json'}")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    trainer = CSMProductionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
