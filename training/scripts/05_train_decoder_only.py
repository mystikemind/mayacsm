#!/usr/bin/env python3
"""
CSM Fine-Tuning: DECODER-ONLY Strategy
========================================

THE BREAKTHROUGH: Train ONLY the decoder (codebooks 1-31) for voice identity,
keep the backbone COMPLETELY FROZEN to preserve naturalness.

Why this works:
- Backbone (codebook 0) = prosody, breathing, pauses, naturalness
  → Trained by Sesame on 1M hours of conversation → FREEZE IT
- Decoder (codebooks 1-31) = timbre, speaker identity, acoustic detail
  → Fine-tune this for Maya's voice identity

Research backing:
- SpeechTokenizer: "Speaker identity is barely present in first RVQ scale,
  highly prevalent on fourth scale" (arxiv:2308.16692)
- FreeCodec: Timbre is separable from prosody (arxiv:2412.01053)
- Freeze-Omni: Frozen backbone + trained decoder preserves naturalness (arxiv:2411.00774)
- LoRP-TTS: LoRA fine-tuning preserves more naturalness than full fine-tune (arxiv:2502.07562)

Usage:
    python 05_train_decoder_only.py --data /path/to/data --epochs 5
"""

import os
import sys
import json
import logging
import argparse
import time
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"


@dataclass
class DecoderOnlyConfig:
    """
    Decoder-only fine-tuning configuration.

    KEY INSIGHT: We ONLY modify the decoder path (codebooks 1-31).
    The backbone (codebook 0) is completely frozen.
    This preserves the base model's naturalness while adapting voice identity.
    """
    model_name: str = "sesame/csm-1b"
    data_dir: str = ""

    # Training hyperparameters (conservative for decoder-only)
    learning_rate: float = 5e-5    # Higher than backbone training since decoder is smaller
    num_epochs: int = 10           # More epochs OK since decoder is smaller
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.5     # Less strict than backbone training
    warmup_steps: int = 100

    # LoRA for decoder (optional - can also do full fine-tune of decoder)
    use_lora: bool = False         # Full decoder fine-tune is fine since it's only 100M
    lora_rank: int = 16
    lora_alpha: int = 16

    # What to train
    train_decoder: bool = True     # Train decoder transformer (100M params)
    train_projection: bool = True  # Train projection layer (backbone -> decoder)
    train_audio_head: bool = True  # Train audio output heads for codebooks 1-31
    train_backbone: bool = False   # FROZEN - preserves naturalness
    train_codebook0_head: bool = False  # FROZEN - preserves prosody prediction

    # Training settings
    max_seq_len: int = 1024
    use_bf16: bool = True
    decoder_frames_per_sample: int = 64  # More frames for decoder training

    # Output
    output_dir: str = str(CHECKPOINT_DIR / "csm_maya_decoder_only")
    save_steps: int = 200
    log_steps: int = 10
    eval_steps: int = 200


class LoRALayer(nn.Module):
    """LoRA adapter for decoder layers."""
    def __init__(self, in_features, out_features, rank, alpha, dropout=0.01):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class MultiSourceDataset(Dataset):
    """
    Load training data from multiple sources.
    Supports both pre-tokenized data and raw audio.
    """
    def __init__(self, data_dirs: List[str], max_seq_len: int = 1024):
        self.samples = []
        self.max_seq_len = max_seq_len

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            # Try tokenized format first
            tokenized_json = data_dir / "train_tokenized.json"
            if tokenized_json.exists():
                with open(tokenized_json) as f:
                    samples = json.load(f)
                for s in samples:
                    if s.get("num_frames", 0) <= max_seq_len:
                        s["_data_dir"] = str(data_dir)
                        s["_format"] = "tokenized"
                        self.samples.append(s)
                logger.info(f"  {data_dir.name}: {len(samples)} tokenized samples")
                continue

            # Try raw format (train.json with audio paths)
            raw_json = data_dir / "train.json"
            if raw_json.exists():
                with open(raw_json) as f:
                    samples = json.load(f)
                for s in samples:
                    s["_data_dir"] = str(data_dir)
                    s["_format"] = "raw"
                    self.samples.append(s)
                logger.info(f"  {data_dir.name}: {len(samples)} raw samples")

        logger.info(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data_dir = Path(sample["_data_dir"])

        if sample["_format"] == "tokenized":
            tokens_path = data_dir / sample["tokens_path"]
            audio_tokens = torch.load(tokens_path, weights_only=True)
            return {
                "audio_tokens": audio_tokens,
                "text": sample.get("text", ""),
                "num_frames": sample.get("num_frames", 0),
            }
        else:
            # Raw audio - will tokenize on-the-fly (slower but flexible)
            return {
                "audio_path": str(data_dir / sample["path"]),
                "text": sample.get("text", ""),
                "duration": sample.get("duration", 0),
                "_format": "raw",
            }


class DecoderOnlyTrainer:
    """
    Fine-tune ONLY the decoder path for voice identity.
    Backbone is COMPLETELY FROZEN to preserve naturalness.
    """

    def __init__(self, config: DecoderOnlyConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.cuda.empty_cache()
        gc.collect()

        self._init_tokenizer()
        self._init_model()
        self._freeze_and_setup()
        self._init_datasets()
        self._init_optimizer()
        self._init_audio_tokenizer()

        self.global_step = 0
        self.best_val_loss = float("inf")

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
        logger.info("CSM DECODER-ONLY FINE-TUNING")
        logger.info("Backbone FROZEN → Naturalness preserved")
        logger.info("Decoder TRAINED → Voice identity adapted")
        logger.info("=" * 70)

        self.model = Model.from_pretrained(self.config.model_name)
        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

        self.num_codebooks = self.model.config.audio_num_codebooks
        self.audio_vocab_size = self.model.config.audio_vocab_size

        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Codebooks: {self.num_codebooks}")

    def _init_audio_tokenizer(self):
        """Load Mimi codec for raw audio tokenization."""
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(32)
        logger.info("Mimi codec loaded for audio tokenization")

    def _freeze_and_setup(self):
        """Freeze backbone, setup decoder training."""
        # STEP 1: Freeze EVERYTHING first
        for param in self.model.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"\nFrozen ALL parameters: {frozen_params:,}")

        # STEP 2: Unfreeze ONLY decoder-path components
        trainable_params = 0

        if self.config.train_decoder:
            for param in self.model.decoder.parameters():
                param.requires_grad = True
            decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
            trainable_params += decoder_params
            logger.info(f"  UNFROZEN decoder: {decoder_params:,} params")

        if self.config.train_projection:
            for param in self.model.projection.parameters():
                param.requires_grad = True
            proj_params = sum(p.numel() for p in self.model.projection.parameters())
            trainable_params += proj_params
            logger.info(f"  UNFROZEN projection: {proj_params:,} params")

        if self.config.train_audio_head:
            self.model.audio_head.requires_grad = True
            head_params = self.model.audio_head.numel()
            trainable_params += head_params
            logger.info(f"  UNFROZEN audio_head: {head_params:,} params")

        # Verify backbone is frozen
        backbone_frozen = all(not p.requires_grad for p in self.model.backbone.parameters())
        c0_frozen = all(not p.requires_grad for p in self.model.codebook0_head.parameters())

        logger.info(f"\n  Backbone FROZEN: {backbone_frozen}")
        logger.info(f"  Codebook0 head FROZEN: {c0_frozen}")
        logger.info(f"  Total trainable: {trainable_params:,} ({trainable_params/frozen_params*100:.1f}%)")
        logger.info(f"  Total frozen: {frozen_params - trainable_params:,}")

    def _init_datasets(self):
        """Initialize datasets from multiple sources."""
        data_dirs = [self.config.data_dir]

        # Check for additional data directories
        datasets_dir = PROJECT_ROOT / "training" / "data" / "datasets"
        extra_dirs = []
        for name in ["disfluency_speech_prepared", "nonverbal_tts_prepared", "expresso_full_prepared"]:
            d = datasets_dir / name
            if d.exists() and (d / "train_tokenized.json").exists():
                extra_dirs.append(str(d))

        all_dirs = data_dirs + extra_dirs
        logger.info(f"\nData sources: {len(all_dirs)}")

        self.train_dataset = MultiSourceDataset(all_dirs, self.config.max_seq_len)

        # Use primary data dir for validation
        val_path = Path(self.config.data_dir) / "val_tokenized.json"
        if val_path.exists():
            self.val_dataset = MultiSourceDataset([self.config.data_dir], self.config.max_seq_len)
            # Hack: use val json
            with open(Path(self.config.data_dir) / "val_tokenized.json") as f:
                val_samples = json.load(f)
            self.val_dataset.samples = []
            for s in val_samples:
                if s.get("num_frames", 0) <= self.config.max_seq_len:
                    s["_data_dir"] = self.config.data_dir
                    s["_format"] = "tokenized"
                    self.val_dataset.samples.append(s)
        else:
            self.val_dataset = self.train_dataset  # Fallback

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=0,
        )
        logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def _init_optimizer(self):
        """Initialize optimizer for decoder-path parameters only."""
        params = []
        if self.config.train_decoder:
            params.extend(list(self.model.decoder.parameters()))
        if self.config.train_projection:
            params.extend(list(self.model.projection.parameters()))
        if self.config.train_audio_head:
            params.append(self.model.audio_head)

        # Filter for requires_grad
        params = [p for p in params if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        steps_per_epoch = max(len(self.train_loader) // self.config.gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * self.config.num_epochs

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=self.config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max=max(total_steps - self.config.warmup_steps, 1),
                                   eta_min=1e-7)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine],
                                      milestones=[self.config.warmup_steps])

        logger.info(f"\nOptimizer: AdamW, lr={self.config.learning_rate}")
        logger.info(f"Total steps: {total_steps}")

    @torch.no_grad()
    def _tokenize_raw_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Tokenize raw audio file with Mimi codec."""
        import torchaudio
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)
            audio = audio.mean(dim=0) if audio.dim() > 1 and audio.size(0) > 1 else audio.squeeze(0)
            audio = audio.to(self.device)
            tokens = self.mimi.encode(audio.unsqueeze(0).unsqueeze(0))[0]  # (32, num_frames)
            return tokens
        except Exception as e:
            logger.warning(f"Failed to tokenize {audio_path}: {e}")
            return None

    def _compute_decoder_loss(self, sample) -> Optional[torch.Tensor]:
        """
        Compute loss ONLY on decoder outputs (codebooks 1-31).

        The backbone generates codebook 0 (FROZEN, no loss).
        We only train the decoder to match codebooks 1-31 for voice identity.
        """
        # Handle different data formats
        if sample.get("_format") == "raw" or "audio_path" in sample:
            audio_path = sample.get("audio_path", [""])[0] if isinstance(sample.get("audio_path"), list) else sample.get("audio_path", "")
            if not audio_path or not os.path.exists(audio_path):
                return None
            audio_tokens = self._tokenize_raw_audio(audio_path)
            if audio_tokens is None:
                return None
            text = sample["text"][0] if isinstance(sample["text"], list) else sample["text"]
        else:
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

        if seq_len > self.config.max_seq_len:
            return None

        # Build input sequence (same as before)
        tokens = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.long, device=self.device)
        token_mask = torch.zeros(1, seq_len, self.num_codebooks + 1, dtype=torch.bool, device=self.device)

        tokens[0, :text_len, -1] = torch.tensor(text_tokens, device=self.device)
        token_mask[0, :text_len, -1] = True
        tokens[0, text_len:, :-1] = audio_tokens.transpose(0, 1)
        token_mask[0, text_len:, :-1] = True

        # Forward through backbone (FROZEN, no gradient)
        with torch.no_grad():
            embeds = self.model._embed_tokens(tokens)
            masked_embeds = embeds * token_mask.unsqueeze(-1)
            h = masked_embeds.sum(dim=2)

            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
            backbone_h = self.model.backbone(h, mask=causal_mask.unsqueeze(0)).to(dtype=self.dtype)

        # Now compute decoder loss WITH gradients
        # Architecture: The decoder input sequence is:
        #   Position 0: projection(backbone_output)  ← "condition" token
        #   Position 1: projection(codebook_0_embed) ← predicts codebook 1
        #   Position 2: projection(codebook_1_embed) ← predicts codebook 2
        #   ...
        #   Position 31: projection(codebook_30_embed) ← predicts codebook 31
        # Predictions at positions 1-31 map to audio_head[0]-audio_head[30]
        total_loss = 0.0
        num_losses = 0

        # Sample frames for decoder training
        num_frames = min(audio_len - 1, self.config.decoder_frames_per_sample)
        frame_indices = np.random.choice(audio_len - 1, size=num_frames, replace=False)

        for frame_idx in frame_indices:
            try:
                # Get backbone output at this frame (detached, no backbone gradient)
                h_frame = backbone_h[0, text_len + frame_idx, :]  # [backbone_dim]

                # Targets for codebooks 1-31 at the NEXT frame
                targets = audio_tokens[1:, frame_idx + 1]

                # Build decoder input sequence matching generate_frame architecture:
                # Position 0: backbone output (in backbone dim, will be projected)
                # Positions 1-31: codebook embeddings 0-30 (in backbone dim)
                input_codes = audio_tokens[:self.num_codebooks - 1, frame_idx + 1]

                code_embeds = []
                for cb in range(self.num_codebooks - 1):
                    offset = input_codes[cb] + cb * self.audio_vocab_size
                    emb = self.model.audio_embeddings(offset.unsqueeze(0))  # [1, backbone_dim]
                    code_embeds.append(emb)

                code_embeds = torch.stack(code_embeds, dim=1)  # [1, 31, backbone_dim]

                # Prepend backbone output as position 0
                decoder_seq = torch.cat([
                    h_frame.unsqueeze(0).unsqueeze(0),  # [1, 1, backbone_dim]
                    code_embeds                          # [1, 31, backbone_dim]
                ], dim=1)  # [1, 32, backbone_dim]

                # Project entire sequence to decoder dimension (THIS HAS GRADIENTS)
                decoder_input = self.model.projection(decoder_seq)  # [1, 32, decoder_dim]

                # Causal mask for 32 positions
                seq_size = self.num_codebooks  # 32
                decoder_mask = torch.tril(torch.ones(
                    seq_size, seq_size,
                    dtype=torch.bool, device=self.device
                )).unsqueeze(0)

                decoder_out = self.model.decoder(decoder_input, mask=decoder_mask)

                # Loss for each codebook 1-31
                # Position cb+1 predicts codebook cb+1 via audio_head[cb]
                for cb in range(self.num_codebooks - 1):
                    logits = torch.mm(
                        decoder_out[0, cb + 1, :].unsqueeze(0),
                        self.model.audio_head[cb].to(decoder_out.dtype)
                    )
                    target = targets[cb]
                    cb_loss = F.cross_entropy(logits, target.unsqueeze(0), ignore_index=0)
                    total_loss += cb_loss
                    num_losses += 1

            except Exception as e:
                logger.debug(f"Frame {frame_idx} error: {e}")
                continue

        if num_losses > 0:
            return total_loss / num_losses
        return None

    def train_epoch(self):
        self.model.decoder.train()
        self.model.projection.train()

        total_loss = 0.0
        num_steps = 0
        self.optimizer.zero_grad()
        accum_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_decoder_loss(batch)

                if loss is not None and torch.isfinite(loss):
                    (loss / self.config.gradient_accumulation_steps).backward()
                    accum_loss += loss.item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logger.warning(f"OOM at batch {batch_idx}")
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
                    avg_loss = accum_loss / self.config.gradient_accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.4f}")

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                accum_loss = 0.0

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        for batch in self.val_loader:
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss = self._compute_decoder_loss(batch)
                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    num_samples += 1
            except:
                continue

        return total_loss / max(num_samples, 1)

    def save_checkpoint(self, name: str):
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save ONLY the components we trained
        torch.save(self.model.decoder.state_dict(), output_dir / "decoder.pt")
        torch.save(self.model.projection.state_dict(), output_dir / "projection.pt")
        torch.save(self.model.audio_head, output_dir / "audio_head.pt")

        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": {
                "strategy": "decoder_only",
                "backbone_frozen": True,
                "codebook0_frozen": True,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
            }
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved: {output_dir}")

    def save_merged(self, name: str = "model_merged"):
        """Save a complete merged model that can be loaded directly."""
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the full model state dict
        # Backbone is base weights (frozen), decoder/projection/audio_head are trained
        torch.save(self.model.state_dict(), output_dir / "model_merged.pt")
        logger.info(f"Merged model saved: {output_dir / 'model_merged.pt'}")

    def train(self):
        logger.info("\n" + "=" * 70)
        logger.info("STARTING DECODER-ONLY TRAINING")
        logger.info("=" * 70)
        logger.info(f"Strategy: FREEZE backbone + codebook0_head")
        logger.info(f"          TRAIN  decoder + projection + audio_head")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info("=" * 70)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}\nEPOCH {epoch + 1}/{self.config.num_epochs}\n{'='*60}")

            epoch_start = time.time()
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {epoch_time/60:.1f}min")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model")
                self.save_merged("best_model_merged")
                logger.info("*** New best model! ***")

            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CSM Decoder-Only Fine-Tuning")
    parser.add_argument("--data", type=str, required=True, help="Primary training data directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "csm_maya_decoder_only"))
    parser.add_argument("--frames-per-sample", type=int, default=64)

    args = parser.parse_args()

    config = DecoderOnlyConfig(
        data_dir=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
        decoder_frames_per_sample=args.frames_per_sample,
    )

    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    trainer = DecoderOnlyTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
