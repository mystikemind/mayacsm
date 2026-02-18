#!/usr/bin/env python3
"""
CSM Fine-Tuning - CORRECT APPROACH
===================================

Based on davidbrowne17/csm-streaming which ACTUALLY WORKS.

Key differences from our failed attempts:
1. Learning rate: 1e-6 (NOT 1e-4 or 1e-5)
2. Epochs: 5 (NOT 100)
3. Max grad norm: 0.1 (NOT 1.0)
4. Train ALL 32 codebooks (NOT just codebook 0)
5. Target modules include w1, w2, w3 (MLP layers)

Source: https://github.com/davidbrowne17/csm-streaming/blob/main/lora.py
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
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Setup paths
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"


@dataclass
class CorrectConfig:
    """
    Configuration based on WORKING csm-streaming implementation.
    Source: https://github.com/davidbrowne17/csm-streaming/blob/main/lora.py
    """
    model_name: str = "sesame/csm-1b"
    data_dir: str = ""

    # CRITICAL: These are the WORKING hyperparameters
    learning_rate: float = 1e-6      # NOT 1e-4! Much lower.
    num_epochs: int = 5              # NOT 100! Much fewer.
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 0.1       # NOT 1.0! Much stricter.
    warmup_steps: int = 50

    # LoRA configuration
    lora_rank: int = 32
    lora_alpha: int = 32             # scaling = alpha/rank = 1.0
    lora_dropout: float = 0.01

    # Target ALL important modules (not just attention)
    lora_target_modules: List[str] = None

    # Training settings
    max_seq_len: int = 1024
    use_bf16: bool = True

    # Output
    output_dir: str = str(CHECKPOINT_DIR / "csm_maya_correct")
    save_steps: int = 100
    log_steps: int = 10

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Include MLP layers (w1, w2, w3) - this is critical!
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "output_proj",
                "w1", "w2", "w3"  # MLP layers - we missed these before!
            ]


class LoRALayer(nn.Module):
    """LoRA adapter layer."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int, dropout: float = 0.01):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # Initialize A with small random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta."""
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class TokenizedDataset(Dataset):
    """Load pre-tokenized audio data."""

    def __init__(self, metadata_path: Path, data_dir: Path, max_seq_len: int = 1024):
        with open(metadata_path) as f:
            self.samples = json.load(f)
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

        # Filter samples that are too long
        original = len(self.samples)
        self.samples = [s for s in self.samples if s.get("num_frames", 0) <= max_seq_len]
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
            "num_frames": sample.get("num_frames", 0),
        }


class CSMCorrectTrainer:
    """
    Trainer using the CORRECT approach from csm-streaming.

    Key insight: Train loss on ALL 32 codebooks, not just codebook 0.
    """

    def __init__(self, config: CorrectConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.cuda.empty_cache()
        gc.collect()

        self._init_tokenizer()
        self._init_model()
        self._setup_lora()
        self._init_datasets()
        self._init_optimizer()

        self.global_step = 0
        self.best_val_loss = float("inf")

    def _init_tokenizer(self):
        """Initialize text tokenizer."""
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing

        logger.info("Loading text tokenizer...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(bos, self.text_tokenizer.bos_token_id), (eos, self.text_tokenizer.eos_token_id)],
        )

    def _init_model(self):
        """Load CSM model."""
        logger.info("=" * 70)
        logger.info("CSM FINE-TUNING - CORRECT APPROACH")
        logger.info("Based on davidbrowne17/csm-streaming (which WORKS)")
        logger.info("=" * 70)

        from models import Model

        self.model = Model.from_pretrained(self.config.model_name)
        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

        self.num_codebooks = self.model.config.audio_num_codebooks
        self.audio_vocab_size = self.model.config.audio_vocab_size

        logger.info(f"Model loaded: {self.config.model_name}")
        logger.info(f"Codebooks: {self.num_codebooks}")
        logger.info(f"Audio vocab: {self.audio_vocab_size}")

    def _setup_lora(self):
        """Setup LoRA adapters on correct target modules."""
        logger.info(f"\nSetting up LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")
        logger.info(f"Target modules: {self.config.lora_target_modules}")

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.lora_layers = nn.ModuleDict()

        # Get backbone layers
        backbone = self.model.backbone

        for layer_idx, layer in enumerate(backbone.layers):
            # Attention projections
            attn = layer.attn
            for name in ["q_proj", "k_proj", "v_proj", "output_proj"]:
                if name in self.config.lora_target_modules:
                    if hasattr(attn, name):
                        proj = getattr(attn, name)
                        in_f = proj.in_features if hasattr(proj, 'in_features') else proj.weight.shape[1]
                        out_f = proj.out_features if hasattr(proj, 'out_features') else proj.weight.shape[0]

                        key = f"layer{layer_idx}_{name}"
                        self.lora_layers[key] = LoRALayer(
                            in_f, out_f,
                            self.config.lora_rank,
                            self.config.lora_alpha,
                            self.config.lora_dropout
                        ).to(self.device, dtype=self.dtype)

            # MLP layers (w1, w2, w3) - CRITICAL: we missed these before!
            mlp = layer.mlp
            for name in ["w1", "w2", "w3"]:
                if name in self.config.lora_target_modules:
                    if hasattr(mlp, name):
                        proj = getattr(mlp, name)
                        in_f = proj.in_features if hasattr(proj, 'in_features') else proj.weight.shape[1]
                        out_f = proj.out_features if hasattr(proj, 'out_features') else proj.weight.shape[0]

                        key = f"layer{layer_idx}_{name}"
                        self.lora_layers[key] = LoRALayer(
                            in_f, out_f,
                            self.config.lora_rank,
                            self.config.lora_alpha,
                            self.config.lora_dropout
                        ).to(self.device, dtype=self.dtype)

        # Enable training for output heads (ALL codebooks)
        for param in self.model.codebook0_head.parameters():
            param.requires_grad = True

        # CRITICAL: Train decoder and audio_head for ALL codebooks
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        if hasattr(self.model, 'audio_head'):
            for head in self.model.audio_head:
                head.requires_grad = True

        # Count trainable params
        lora_params = sum(p.numel() for p in self.lora_layers.parameters())
        head_params = sum(p.numel() for p in self.model.codebook0_head.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)

        total_trainable = lora_params + head_params + decoder_params
        logger.info(f"LoRA params: {lora_params:,}")
        logger.info(f"Head params: {head_params:,}")
        logger.info(f"Decoder params: {decoder_params:,}")
        logger.info(f"Total trainable: {total_trainable:,}")

    def _init_datasets(self):
        """Initialize datasets."""
        data_dir = Path(self.config.data_dir)

        train_path = data_dir / "train_tokenized.json"
        self.train_dataset = TokenizedDataset(train_path, data_dir, self.config.max_seq_len)

        val_path = data_dir / "val_tokenized.json"
        self.val_dataset = TokenizedDataset(val_path, data_dir, self.config.max_seq_len)

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

        logger.info(f"Train: {len(self.train_dataset)} samples")
        logger.info(f"Val: {len(self.val_dataset)} samples")

    def _init_optimizer(self):
        """Initialize optimizer with CORRECT hyperparameters."""
        params = list(self.lora_layers.parameters())
        params += list(self.model.codebook0_head.parameters())
        params += [p for p in self.model.decoder.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Cosine scheduler
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - self.config.warmup_steps, 1),
            eta_min=1e-8
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            [warmup, cosine],
            milestones=[self.config.warmup_steps]
        )

        logger.info(f"\nOptimizer settings (CORRECT values):")
        logger.info(f"  Learning rate: {self.config.learning_rate} (was 1e-4, now 1e-6)")
        logger.info(f"  Epochs: {self.config.num_epochs} (was 100, now 5)")
        logger.info(f"  Max grad norm: {self.config.max_grad_norm} (was 1.0, now 0.1)")
        logger.info(f"  Total steps: {total_steps}")

    def _compute_loss(self, sample) -> Optional[torch.Tensor]:
        """
        Compute loss on ALL 32 codebooks.

        CRITICAL: Previous implementation only computed loss on codebook 0.
        This implementation follows csm-streaming and computes loss on ALL codebooks.
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
            return None

        # Build input sequence
        seq_len = text_len + audio_len
        if seq_len > self.config.max_seq_len:
            return None

        tokens = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        token_mask = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)

        # Text tokens
        tokens[0, :text_len, -1] = torch.tensor(text_tokens, device=self.device)
        token_mask[0, :text_len, -1] = True

        # Audio tokens
        tokens[0, text_len:, :-1] = audio_tokens.transpose(0, 1)
        token_mask[0, text_len:, :-1] = True

        # Embed and forward
        embeds = self.model._embed_tokens(tokens)
        masked_embeds = embeds * token_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)

        # Backbone forward
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
        backbone_h = self.model.backbone(h, mask=causal_mask.unsqueeze(0)).to(dtype=self.dtype)

        total_loss = 0.0
        num_losses = 0

        # ===== CODEBOOK 0 LOSS (backbone) =====
        audio_positions = backbone_h[0, text_len:-1, :]  # All but last
        c0_targets = audio_tokens[0, 1:]  # Shifted targets
        c0_logits = self.model.codebook0_head(audio_positions)

        c0_loss = F.cross_entropy(c0_logits, c0_targets, ignore_index=0)
        total_loss += c0_loss
        num_losses += 1

        # ===== CODEBOOKS 1-31 LOSS (decoder) =====
        # This is CRITICAL - we weren't training these before!
        num_decoder_frames = min(audio_len - 1, 32)  # Limit for memory
        frame_indices = np.random.choice(audio_len - 1, size=num_decoder_frames, replace=False)

        for frame_idx in frame_indices:
            try:
                h_frame = backbone_h[0, text_len + frame_idx, :].unsqueeze(0)
                h_proj = self.model.projection(h_frame)

                # Targets for codebooks 1-31
                targets = audio_tokens[1:, frame_idx + 1]  # (31,)

                # Get input codes for decoder
                input_codes = audio_tokens[:self.num_codebooks - 1, frame_idx + 1]  # (31,)

                # Embed and project
                code_embeds = []
                for cb in range(self.num_codebooks - 1):
                    offset = input_codes[cb] + (cb + 1) * self.audio_vocab_size
                    emb = self.model.audio_embeddings(offset.unsqueeze(0))
                    code_embeds.append(emb)

                code_embeds = torch.stack(code_embeds, dim=1)  # (1, 31, dim)
                code_embeds_proj = self.model.projection(code_embeds)

                h_proj_exp = h_proj.unsqueeze(1).expand_as(code_embeds_proj)
                decoder_input = code_embeds_proj + h_proj_exp

                # Decoder forward
                decoder_mask = torch.tril(torch.ones(
                    self.num_codebooks - 1, self.num_codebooks - 1,
                    dtype=torch.bool, device=self.device
                )).unsqueeze(0)

                decoder_out = self.model.decoder(decoder_input, mask=decoder_mask)

                # Loss for each codebook
                for cb in range(self.num_codebooks - 1):
                    logits = decoder_out[0, cb, :] @ self.model.audio_head[cb]
                    target = targets[cb]

                    cb_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0), ignore_index=0)
                    total_loss += cb_loss
                    num_losses += 1

            except Exception:
                continue

        if num_losses > 0:
            return total_loss / num_losses
        return None

    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        self.lora_layers.train()

        total_loss = 0.0
        num_steps = 0
        self.optimizer.zero_grad()
        accum_loss = 0.0

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
                    logger.warning(f"OOM at batch {batch_idx}")
                continue

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # CRITICAL: Use low gradient norm (0.1, not 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.lora_layers.parameters()) +
                    list(self.model.codebook0_head.parameters()) +
                    [p for p in self.model.decoder.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accum_loss
                num_steps += 1

                if self.global_step % self.config.log_steps == 0:
                    avg_loss = accum_loss / self.config.gradient_accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.4f}")

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                accum_loss = 0.0

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        self.lora_layers.eval()

        total_loss = 0.0
        num_samples = 0

        for batch in self.val_loader:
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_loss(batch)
                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    num_samples += 1
            except:
                continue

        self.model.train()
        self.lora_layers.train()

        return total_loss / max(num_samples, 1)

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        torch.save(self.lora_layers.state_dict(), output_dir / "lora_weights.pt")

        # Save decoder and head weights
        torch.save(self.model.codebook0_head.state_dict(), output_dir / "codebook0_head.pt")
        torch.save(self.model.decoder.state_dict(), output_dir / "decoder.pt")

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": {
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "max_grad_norm": self.config.max_grad_norm,
            }
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved: {output_dir}")

    def train(self):
        """Main training loop."""
        logger.info("\n" + "=" * 70)
        logger.info("STARTING CORRECT TRAINING")
        logger.info("=" * 70)
        logger.info(f"Learning rate: {self.config.learning_rate} (1e-6, not 1e-4)")
        logger.info(f"Epochs: {self.config.num_epochs} (5, not 100)")
        logger.info(f"Max grad norm: {self.config.max_grad_norm} (0.1, not 1.0)")
        logger.info(f"Training ALL 32 codebooks (not just codebook 0)")
        logger.info("=" * 70)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}\nEPOCH {epoch + 1}/{self.config.num_epochs}\n{'='*60}")

            epoch_start = time.time()
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch + 1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {epoch_time/60:.1f}min")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model")
                logger.info("New best model!")

            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CSM Fine-Tuning - CORRECT APPROACH")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "csm_maya_correct"))

    args = parser.parse_args()

    config = CorrectConfig(
        data_dir=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    # Verify data
    data_dir = Path(args.data)
    if not (data_dir / "train_tokenized.json").exists():
        logger.error(f"Data not found: {data_dir / 'train_tokenized.json'}")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    trainer = CSMCorrectTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
