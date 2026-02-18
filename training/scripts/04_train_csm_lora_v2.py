#!/usr/bin/env python3
"""
CSM-1B LoRA Fine-Tuning - Production Grade
===========================================

Uses LoRA (Low-Rank Adaptation) to fine-tune ONLY the backbone,
preserving all pre-trained text-to-speech capabilities while adapting
voice characteristics to match Maya's voice.

Key design choices (from verified community approaches + research):
1. LoRA on backbone attention + MLP (rank 32, ~22M trainable params)
2. Decoder fully frozen (preserves audio codec quality)
3. Text/audio embeddings frozen (prevents catastrophic forgetting)
4. codebook0_head trainable (adapts voice-specific c0 predictions)
5. C0-only loss with label smoothing
6. Generation quality monitoring at checkpoints (catches forgetting early)
7. Multi-GPU DDP support

Based on:
- davidbrowne17/csm-streaming: verified working LoRA (rank 32, lr 1e-6)
- keanteng/sesame-csm-elise-lora: HuggingFace PEFT approach
- LoRP-TTS (Samsung R&D, 2025): optimal rank ~2-3% of params
- ICML 2024 "LoRA Learns Less and Forgets Less": regularization effect
- Speechmatics fine-tuning guide: loss formula and data handling

Usage:
  # Single GPU:
  python 04_train_csm_lora_v2.py --data training/data/csm_ready_ex04

  # Multi-GPU (4x A10G):
  torchrun --nproc_per_node=4 04_train_csm_lora_v2.py \\
    --data training/data/csm_ready_ex04

  # Resume from checkpoint:
  python 04_train_csm_lora_v2.py --data training/data/csm_ready_ex04 \\
    --resume training/checkpoints/csm_maya_lora/checkpoint-500
"""

import os
import sys
import json
import math
import logging
import argparse
import time
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Setup CSM path
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"


def setup_logging(rank):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f'%(asctime)s | RANK {rank} | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


# ============================================================================
# LORA IMPLEMENTATION
# ============================================================================

class LoRALinear(nn.Module):
    """
    LoRA wrapper around an existing nn.Linear layer.

    Computation: output = base_linear(x) + (alpha/rank) * B(A(dropout(x)))

    The base linear's weights are FROZEN. Only A and B are trainable.
    B is initialized to zero so the initial model output is identical
    to the pre-trained model.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 32,
                 alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        # LoRA decomposition matrices - MUST be on same device/dtype as base
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Critical: A → random init, B → zeros
        # This ensures initial output = base output (LoRA contribution = 0)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base weights
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return base_out + self.scaling * lora_out

    @property
    def weight(self):
        """For compatibility with code that accesses .weight directly."""
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into base for inference. Returns plain nn.Linear."""
        merged = nn.Linear(
            self.in_features, self.out_features,
            bias=self.base.bias is not None
        )
        # W_merged = W_base + (alpha/rank) * B @ A
        with torch.no_grad():
            merged.weight.copy_(
                self.base.weight + self.scaling *
                (self.lora_B.weight @ self.lora_A.weight)
            )
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def inject_lora_into_backbone(model, rank: int = 32, alpha: float = 32.0,
                               dropout: float = 0.05,
                               target_attn: bool = True,
                               target_mlp: bool = True) -> int:
    """
    Replace backbone linear layers with LoRA-wrapped versions.

    Targets attention projections (q_proj, k_proj, v_proj, output_proj)
    and optionally MLP layers (w1/gate, w2/down, w3/up).

    Returns the number of LoRA modules injected.
    """
    attn_targets = {"q_proj", "k_proj", "v_proj", "output_proj"}
    mlp_targets = {"w1", "w2", "w3"}

    targets = set()
    if target_attn:
        targets |= attn_targets
    if target_mlp:
        targets |= mlp_targets

    count = 0

    # Navigate the backbone's module tree
    for layer_idx, layer in enumerate(model.backbone.layers):
        for name, module in layer.named_children():
            # Check sub-modules of attention and mlp blocks
            for sub_name, sub_module in module.named_children():
                if sub_name in targets and isinstance(sub_module, nn.Linear):
                    lora_module = LoRALinear(
                        sub_module, rank=rank, alpha=alpha, dropout=dropout
                    )
                    setattr(module, sub_name, lora_module)
                    count += 1

    return count


def merge_lora_weights(model) -> None:
    """Merge all LoRA weights back into base model for inference."""
    for layer in model.backbone.layers:
        for name, module in layer.named_children():
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, LoRALinear):
                    merged = sub_module.merge_weights()
                    merged.to(device=sub_module.base.weight.device,
                              dtype=sub_module.base.weight.dtype)
                    setattr(module, sub_name, merged)


# ============================================================================
# DATASET
# ============================================================================

class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized CSM audio."""

    def __init__(self, metadata_path: Path, data_dir: Path,
                 max_frames: int = 150):
        with open(metadata_path) as f:
            self.samples = json.load(f)

        self.data_dir = data_dir
        original = len(self.samples)
        self.samples = [
            s for s in self.samples
            if s.get("num_frames", 0) <= max_frames and s.get("num_frames", 0) >= 5
        ]
        filtered = original - len(self.samples)
        if filtered > 0:
            print(f"  Dataset: {len(self.samples)} samples (filtered {filtered})")

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


# ============================================================================
# TRAINING CONFIG
# ============================================================================

@dataclass
class LoRAConfig:
    """
    LoRA fine-tuning configuration.
    Based on verified community approaches + research.
    """
    model_name: str = "sesame/csm-1b"
    use_bf16: bool = True
    data_dir: str = ""
    max_frames: int = 150

    # LoRA hyperparameters (from davidbrowne17 + LoRP-TTS consensus)
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_attn: bool = True
    lora_target_mlp: bool = True

    # Which extra components to train (besides LoRA params)
    train_codebook0_head: bool = True   # Adapts c0 voice predictions
    train_projection: bool = False      # Bridge to decoder (frozen)
    train_embeddings: bool = False      # FREEZE to prevent forgetting

    # Training hyperparameters
    num_epochs: int = 15
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch = 8 per GPU

    learning_rate: float = 5e-5     # Between 1e-6 (too slow) and 1e-4 (too fast)
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 0.5      # Conservative clipping

    # Loss
    label_smoothing: float = 0.1

    # Checkpointing
    save_steps: int = 250
    eval_steps: int = 125
    log_steps: int = 10
    generation_test_steps: int = 500  # Full generation test interval
    output_dir: str = str(CHECKPOINT_DIR / "csm_maya_lora")

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001


# ============================================================================
# TRAINER
# ============================================================================

def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


class CSMLoRATrainer:
    """
    LoRA trainer for CSM-1B.
    Trains only LoRA parameters in backbone + codebook0_head.
    """

    def __init__(self, config: LoRAConfig, rank: int = 0, world_size: int = 1):
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
        self._init_lora()
        self._init_datasets()
        self._init_optimizer()

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0

        # Generation test components (loaded lazily)
        self._mimi = None
        self._voice_prompt_segment = None

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
            special_tokens=[
                (bos, self.text_tokenizer.bos_token_id),
                (eos, self.text_tokenizer.eos_token_id)
            ],
        )

    def _init_model(self):
        if self.is_main:
            self.logger.info("=" * 60)
            self.logger.info("CSM-1B LoRA FINE-TUNING")
            self.logger.info("=" * 60)

        from models import Model
        self.model = Model.from_pretrained(self.config.model_name)
        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

        self.audio_vocab_size = self.model.config.audio_vocab_size
        self.num_codebooks = self.model.config.audio_num_codebooks

        # Causal mask (reusable)
        self.backbone_mask = _create_causal_mask(2048, self.device)

    def _init_lora(self):
        """Inject LoRA into backbone and configure trainable parameters."""
        # Step 1: Freeze EVERYTHING
        for param in self.model.parameters():
            param.requires_grad = False

        if self.is_main:
            self.logger.info("")
            self.logger.info("Injecting LoRA into backbone...")

        # Step 2: Inject LoRA into backbone
        num_lora = inject_lora_into_backbone(
            self.model,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_attn=self.config.lora_target_attn,
            target_mlp=self.config.lora_target_mlp,
        )

        if self.is_main:
            self.logger.info(f"  Injected {num_lora} LoRA modules (rank={self.config.lora_rank})")

        # Step 3: Unfreeze additional components
        if self.config.train_codebook0_head:
            for param in self.model.codebook0_head.parameters():
                param.requires_grad = True
            if self.is_main:
                self.logger.info("  codebook0_head: TRAINABLE")

        if self.config.train_projection:
            for param in self.model.projection.parameters():
                param.requires_grad = True
            if self.is_main:
                self.logger.info("  projection: TRAINABLE")

        if self.config.train_embeddings:
            for param in self.model.text_embeddings.parameters():
                param.requires_grad = True
            for param in self.model.audio_embeddings.parameters():
                param.requires_grad = True
            if self.is_main:
                self.logger.info("  text_embeddings: TRAINABLE")
                self.logger.info("  audio_embeddings: TRAINABLE")

        # Report parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel() for n, p in self.model.named_parameters()
            if p.requires_grad and ('lora_A' in n or 'lora_B' in n)
        )

        if self.is_main:
            self.logger.info("")
            self.logger.info(f"  Total parameters:     {total_params:>12,}")
            self.logger.info(f"  Trainable parameters: {trainable_params:>12,} ({100*trainable_params/total_params:.2f}%)")
            self.logger.info(f"  LoRA parameters:      {lora_params:>12,}")
            self.logger.info(f"  Other trainable:      {trainable_params - lora_params:>12,}")
            self.logger.info("")

            # Verify frozen components
            frozen_components = []
            for name in ['decoder', 'text_embeddings', 'audio_embeddings',
                         'projection', 'audio_head']:
                module = getattr(self.model, name, None)
                if module is not None:
                    if isinstance(module, nn.Parameter):
                        is_frozen = not module.requires_grad
                    else:
                        is_frozen = all(not p.requires_grad for p in module.parameters())
                    if is_frozen:
                        frozen_components.append(name)
            self.logger.info(f"  Frozen: {', '.join(frozen_components)}")

        # Synchronize LoRA parameters across GPUs (they're randomly initialized)
        if self.world_size > 1:
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            if self.is_main:
                self.logger.info("  Synchronized parameters across GPUs")

    def _init_datasets(self):
        data_dir = Path(self.config.data_dir)

        train_path = data_dir / "train_tokenized.json"
        self.train_dataset = PreTokenizedDataset(
            train_path, data_dir, self.config.max_frames
        )

        val_path = data_dir / "val_tokenized.json"
        self.val_dataset = PreTokenizedDataset(
            val_path, data_dir, self.config.max_frames
        )

        if self.is_main:
            self.logger.info(f"Train: {len(self.train_dataset)} samples")
            self.logger.info(f"Val:   {len(self.val_dataset)} samples")

        # Distributed samplers
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                self.train_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True
            )
            self.val_sampler = DistributedSampler(
                self.val_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=False
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
        """Setup optimizer for LoRA params + trainable heads."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        steps_per_epoch = max(
            1, len(self.train_loader) // self.config.gradient_accumulation_steps
        )
        self.total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(self.total_steps * self.config.warmup_ratio)

        # Cosine annealing (better for LoRA than linear decay)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.total_steps - warmup_steps, 1),
            eta_min=self.config.min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[warmup_steps]
        )

        if self.is_main:
            self.logger.info(f"Steps per epoch: {steps_per_epoch}")
            self.logger.info(f"Total steps: {self.total_steps}, Warmup: {warmup_steps}")
            self.logger.info(f"LR: {self.config.learning_rate} → {self.config.min_lr} (cosine)")

    # ========================================================================
    # LOSS COMPUTATION
    # ========================================================================

    def _compute_c0_loss(self, sample) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Backbone-only loss: predict codebook 0 tokens from text + audio context.
        This is the key metric for text-to-speech alignment.
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
        if seq_len > 2048:
            return None, {}  # Skip sequences that exceed max length

        tokens = torch.zeros(
            1, seq_len, self.num_codebooks + 1,
            dtype=torch.long, device=self.device
        )
        token_mask = torch.zeros(
            1, seq_len, self.num_codebooks + 1,
            dtype=torch.bool, device=self.device
        )

        # Text tokens in last column (dim 32)
        tokens[0, :text_len, -1] = torch.tensor(text_tokens, device=self.device)
        token_mask[0, :text_len, -1] = True

        # Audio tokens in first 32 columns (dims 0-31)
        tokens[0, text_len:, :-1] = audio_tokens.transpose(0, 1)
        token_mask[0, text_len:, :-1] = True

        # Forward through backbone
        embeds = self.model._embed_tokens(tokens)
        masked_embeds = embeds * token_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)

        causal_mask = self.backbone_mask[:seq_len, :seq_len].unsqueeze(0)
        backbone_h = self.model.backbone(h, mask=causal_mask).to(dtype=self.dtype)

        # Predict next frame's codebook 0
        num_audio_frames = audio_len - 1
        audio_h = backbone_h[0, text_len:text_len + num_audio_frames, :]
        c0_targets = audio_tokens[0, 1:audio_len]

        c0_logits = self.model.codebook0_head(audio_h)

        c0_loss = F.cross_entropy(
            c0_logits, c0_targets,
            label_smoothing=self.config.label_smoothing
        )

        # Compute accuracy metrics (for monitoring forgetting)
        with torch.no_grad():
            c0_preds = c0_logits.argmax(dim=-1)
            c0_accuracy = (c0_preds == c0_targets).float().mean().item()
            c0_top5 = (
                c0_logits.topk(5, dim=-1).indices == c0_targets.unsqueeze(-1)
            ).any(dim=-1).float().mean().item()

        metrics = {
            "c0_loss": c0_loss.item(),
            "c0_accuracy": c0_accuracy,
            "c0_top5_accuracy": c0_top5,
            "num_frames": num_audio_frames,
        }

        return c0_loss, metrics

    # ========================================================================
    # GENERATION QUALITY TEST
    # ========================================================================

    @torch.inference_mode()
    def run_generation_test(self) -> dict:
        """
        Generate audio samples and check quality.
        This catches catastrophic forgetting that loss curves can miss.
        """
        if not self.is_main:
            return {}

        self.logger.info("Running generation quality test...")
        self.model.eval()

        test_prompts = [
            "yeah, totally.",
            "aw man, that sounds really rough, im so sorry.",
            "oh hey, hi! its really nice to meet you.",
        ]

        # Load Mimi if not already loaded
        if self._mimi is None:
            try:
                from moshi.models import loaders
                from huggingface_hub import hf_hub_download
                mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
                self._mimi = loaders.get_mimi(mimi_weight, device=self.device)
                self._mimi.set_num_codebooks(32)
            except Exception as e:
                self.logger.warning(f"Could not load Mimi for generation test: {e}")
                return {}

        # Load voice prompt context
        if self._voice_prompt_segment is None:
            try:
                vp_path = PROJECT_ROOT / "assets" / "voice_prompt" / "maya_voice_prompt.pt"
                if vp_path.exists():
                    vp_data = torch.load(vp_path, weights_only=False)
                    audio = vp_data['audio']
                    if audio.dim() > 1:
                        audio = audio.squeeze()
                    # Truncate to 2 seconds for short context
                    audio = audio[:2 * 24000].to(self.device)
                    text = "oh hey! yeah im doing pretty good"

                    # Tokenize as context segment
                    vp_text_tokens = self.text_tokenizer.encode(f"[0]{text}")
                    vp_text_frame = torch.zeros(len(vp_text_tokens), 33).long().to(self.device)
                    vp_text_mask = torch.zeros(len(vp_text_tokens), 33).bool().to(self.device)
                    vp_text_frame[:, -1] = torch.tensor(vp_text_tokens).to(self.device)
                    vp_text_mask[:, -1] = True

                    audio_tokens = self._mimi.encode(audio.unsqueeze(0).unsqueeze(0))[0]
                    eos_frame = torch.zeros(audio_tokens.size(0), 1).long().to(self.device)
                    audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

                    vp_audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
                    vp_audio_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
                    vp_audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
                    vp_audio_mask[:, :-1] = True

                    self._voice_prompt_segment = {
                        'tokens': torch.cat([vp_text_frame, vp_audio_frame], dim=0),
                        'mask': torch.cat([vp_text_mask, vp_audio_mask], dim=0),
                    }
                    self.logger.info(f"  Voice prompt context: {self._voice_prompt_segment['tokens'].size(0)} frames")
            except Exception as e:
                self.logger.warning(f"Could not load voice prompt: {e}")

        # Setup caches for generation
        self.model.setup_caches(1)

        results = []
        for prompt_text in test_prompts:
            self.model.reset_caches()

            # Tokenize generation text
            gen_tokens_ids = self.text_tokenizer.encode(f"[0]{prompt_text}")
            gen_frame = torch.zeros(len(gen_tokens_ids), 33).long().to(self.device)
            gen_mask = torch.zeros(len(gen_tokens_ids), 33).bool().to(self.device)
            gen_frame[:, -1] = torch.tensor(gen_tokens_ids).to(self.device)
            gen_mask[:, -1] = True

            # Build prompt: context + generation text
            if self._voice_prompt_segment is not None:
                all_tokens = torch.cat([
                    self._voice_prompt_segment['tokens'], gen_frame
                ], dim=0)
                all_mask = torch.cat([
                    self._voice_prompt_segment['mask'], gen_mask
                ], dim=0)
            else:
                all_tokens = gen_frame
                all_mask = gen_mask

            # Generate
            curr_tokens = all_tokens.unsqueeze(0)
            curr_mask = all_mask.unsqueeze(0)
            curr_pos = torch.arange(all_tokens.size(0)).unsqueeze(0).to(self.device)

            frames = []
            max_frames = 75  # ~6 seconds max for test

            try:
                for _ in range(max_frames):
                    sample = self.model.generate_frame(
                        curr_tokens, curr_mask, curr_pos,
                        temperature=0.9, topk=50
                    )

                    if torch.all(sample == 0):
                        break

                    frames.append(sample.clone())

                    curr_tokens = torch.cat(
                        [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
                    ).unsqueeze(1)
                    curr_mask = torch.cat(
                        [torch.ones_like(sample).bool(),
                         torch.zeros(1, 1).bool().to(self.device)], dim=1
                    ).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1

                num_gen_frames = len(frames)
                eos_produced = num_gen_frames < max_frames
                duration_s = num_gen_frames * 0.08  # 80ms per frame

                # Decode and check amplitude if we got frames
                amplitude_range = (0.0, 0.0)
                transcription = ""
                if frames:
                    stacked = torch.stack(frames).permute(1, 2, 0)
                    audio_out = self._mimi.decode(stacked).squeeze(0).squeeze(0)
                    amplitude_range = (audio_out.min().item(), audio_out.max().item())

                    # Save sample
                    import torchaudio
                    sample_dir = Path(self.config.output_dir) / "gen_tests"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    sample_path = sample_dir / f"step{self.global_step}_{prompt_text[:20].replace(' ', '_')}.wav"
                    torchaudio.save(
                        str(sample_path),
                        audio_out.unsqueeze(0).cpu().float(),
                        24000
                    )

                results.append({
                    "prompt": prompt_text[:30],
                    "frames": num_gen_frames,
                    "eos": eos_produced,
                    "duration": f"{duration_s:.1f}s",
                    "amplitude": f"[{amplitude_range[0]:.2f}, {amplitude_range[1]:.2f}]",
                })

            except Exception as e:
                results.append({
                    "prompt": prompt_text[:30],
                    "error": str(e)[:50],
                })

        # Teardown KV-caches COMPLETELY to return to training mode.
        # torchtune gates inference mode on MultiHeadAttention.cache_enabled (bool).
        # setup_caches() sets cache_enabled=True; we must set it back to False
        # AND delete the KVCache objects, otherwise forward() requires input_pos.
        self.model.reset_caches()
        for layer in self.model.backbone.layers:
            if hasattr(layer, 'attn'):
                layer.attn.kv_cache = None
                layer.attn.cache_enabled = False
        for layer in self.model.decoder.layers:
            if hasattr(layer, 'attn'):
                layer.attn.kv_cache = None
                layer.attn.cache_enabled = False

        # Log results
        eos_count = sum(1 for r in results if r.get("eos", False))
        self.logger.info(f"  Generation test: {eos_count}/{len(results)} produced EOS")
        for r in results:
            if "error" in r:
                self.logger.info(f"    '{r['prompt']}' → ERROR: {r['error']}")
            else:
                eos_str = "EOS" if r["eos"] else "MAX"
                self.logger.info(
                    f"    '{r['prompt']}' → {r['duration']} ({eos_str}) amp={r['amplitude']}"
                )

        self.model.train()
        return {"generation_eos_rate": eos_count / max(len(results), 1)}

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================

    def train_epoch(self):
        self.model.train()
        if self.train_sampler:
            self.train_sampler.set_epoch(self.epoch)

        total_loss = 0.0
        total_acc = 0.0
        num_steps = 0
        self.optimizer.zero_grad()
        accum_loss = 0.0
        accum_acc = 0.0
        batch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss, metrics = self._compute_c0_loss(batch)

                if loss is not None and torch.isfinite(loss):
                    (loss / self.config.gradient_accumulation_steps).backward()
                    accum_loss += loss.item()
                    accum_acc += metrics.get("c0_accuracy", 0)

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
                # Synchronize gradients in distributed mode
                if self.world_size > 1:
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= self.world_size

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accum_loss
                total_acc += accum_acc
                num_steps += 1

                # Log
                if self.is_main and self.global_step % self.config.log_steps == 0:
                    elapsed = time.time() - batch_start
                    lr = self.scheduler.get_last_lr()[0]
                    avg_loss = accum_loss / self.config.gradient_accumulation_steps
                    avg_acc = accum_acc / self.config.gradient_accumulation_steps

                    self.logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"C0 Acc: {avg_acc:.2%} | "
                        f"LR: {lr:.2e} | Grad: {grad_norm:.2f}"
                    )
                    batch_start = time.time()

                # Evaluate
                if self.global_step % self.config.eval_steps == 0:
                    val_loss, val_acc, val_top5 = self.evaluate()
                    if self.is_main:
                        self.logger.info(
                            f"Step {self.global_step} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Val C0 Acc: {val_acc:.2%} | "
                            f"Val Top5: {val_top5:.2%}"
                        )

                        if val_loss < self.best_val_loss - self.config.min_delta:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            self.save_checkpoint("best_model")
                            self.logger.info("  >>> New best model saved! <<<")
                        else:
                            self.patience_counter += 1
                            self.logger.info(
                                f"  Patience: {self.patience_counter}/{self.config.patience}"
                            )
                            if self.patience_counter >= self.config.patience:
                                self.logger.info(
                                    f"Early stopping: no improvement for "
                                    f"{self.config.patience} evals"
                                )
                                return -1

                # Generation test
                if (self.is_main and
                        self.global_step % self.config.generation_test_steps == 0):
                    gen_results = self.run_generation_test()

                # Save checkpoint
                if self.is_main and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                accum_loss = 0.0
                accum_acc = 0.0

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float]:
        """Evaluate on validation set. Returns (loss, accuracy, top5_accuracy)."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_top5 = 0.0
        num_batches = 0

        for batch in self.val_loader:
            try:
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    loss, metrics = self._compute_c0_loss(batch)
                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    total_acc += metrics.get("c0_accuracy", 0)
                    total_top5 += metrics.get("c0_top5_accuracy", 0)
                    num_batches += 1
            except Exception:
                continue

        self.model.train()

        if num_batches == 0:
            return float("inf"), 0.0, 0.0

        # Synchronize in distributed mode
        if self.world_size > 1:
            vals = torch.tensor(
                [total_loss, total_acc, total_top5, num_batches],
                device=self.device
            )
            dist.all_reduce(vals, op=dist.ReduceOp.SUM)
            total_loss, total_acc, total_top5, num_batches = vals.tolist()

        return (
            total_loss / num_batches,
            total_acc / num_batches,
            total_top5 / num_batches,
        )

    # ========================================================================
    # CHECKPOINTING
    # ========================================================================

    def save_checkpoint(self, name: str):
        """Save LoRA adapter weights + training state."""
        if not self.is_main:
            return

        save_dir = Path(self.config.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FULL model state dict (base + LoRA)
        # This allows direct loading for inference
        model_path = save_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Also save LoRA-only state dict (smaller, for adapter loading)
        lora_state = {}
        for k, v in self.model.state_dict().items():
            if 'lora_A' in k or 'lora_B' in k:
                lora_state[k] = v
            elif 'codebook0_head' in k:
                lora_state[k] = v
        lora_path = save_dir / "lora_adapter.pt"
        torch.save(lora_state, lora_path)

        # Save merged model (base weights + LoRA merged)
        # This is what you'd use for inference (no LoRA overhead)
        # Save for best_model and step checkpoints (enables easy evaluation)
        if name == "best_model" or name.startswith("checkpoint-"):
            import copy
            merged_model = copy.deepcopy(self.model)
            merge_lora_weights(merged_model)
            merged_path = save_dir / "model_merged.pt"
            torch.save(merged_model.state_dict(), merged_path)
            del merged_model

        # Training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "config": asdict(self.config),
        }
        with open(save_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        model_size = model_path.stat().st_size / 1e9
        lora_size = lora_path.stat().st_size / 1e6
        self.logger.info(
            f"  Saved {name}: model={model_size:.1f}GB, lora={lora_size:.1f}MB"
        )

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load from checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            if self.is_main:
                self.logger.info(f"Loading checkpoint from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.start_epoch = self.epoch + 1
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            self.patience_counter = state.get("patience_counter", 0)
            if self.is_main:
                self.logger.info(f"  Resumed: step {self.global_step}, epoch {self.epoch}")

    # ========================================================================
    # MAIN TRAINING FLOW
    # ========================================================================

    def train(self):
        """Full training loop with monitoring."""
        if self.is_main:
            self.logger.info("=" * 60)
            self.logger.info("STARTING LoRA TRAINING")
            self.logger.info("=" * 60)
            self.logger.info("")

        # Step 0: Baseline evaluation BEFORE any training
        val_loss, val_acc, val_top5 = self.evaluate()
        if self.is_main:
            self.logger.info(
                f"BASELINE (before training): "
                f"Val Loss: {val_loss:.4f} | "
                f"C0 Acc: {val_acc:.2%} | "
                f"Top5: {val_top5:.2%}"
            )

        # Step 0b: Baseline generation test
        if self.is_main:
            gen_results = self.run_generation_test()
            self.logger.info("")

        train_start = time.time()

        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            if self.is_main:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                self.logger.info(f"{'='*60}")

            result = self.train_epoch()
            if result == -1:  # Early stopping
                break

            epoch_time = time.time() - epoch_start
            if self.is_main:
                self.logger.info(
                    f"Epoch {epoch + 1} complete in {epoch_time:.0f}s | "
                    f"Best val loss: {self.best_val_loss:.4f}"
                )

            # Save epoch checkpoint
            if self.is_main:
                self.save_checkpoint(f"epoch-{epoch}")

                # Cleanup old epoch checkpoints (keep last 3)
                output_dir = Path(self.config.output_dir)
                epoch_dirs = sorted(
                    output_dir.glob("epoch-*"),
                    key=lambda d: int(d.name.split("-")[1])
                )
                for old_dir in epoch_dirs[:-3]:
                    import shutil
                    shutil.rmtree(old_dir)

        total_time = time.time() - train_start
        if self.is_main:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info(f"TRAINING COMPLETE in {total_time/3600:.1f} hours")
            self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")
            self.logger.info(f"Best model: {self.config.output_dir}/best_model/")
            self.logger.info("=" * 60)

            # Final generation test
            self.logger.info("\nFinal generation quality test:")
            self.run_generation_test()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CSM-1B LoRA Fine-Tuning")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--no-mlp", action="store_true", help="Don't target MLP layers")
    parser.add_argument("--no-c0-head", action="store_true", help="Don't train codebook0_head")
    parser.add_argument("--gen-steps", type=int, default=500, help="Generation test interval")

    args = parser.parse_args()

    # Distributed setup
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
    else:
        rank = 0
        world_size = 1

    config = LoRAConfig(
        data_dir=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_target_mlp=not args.no_mlp,
        train_codebook0_head=not args.no_c0_head,
        generation_test_steps=args.gen_steps,
    )
    if args.output:
        config.output_dir = args.output

    logger = setup_logging(rank)

    if rank == 0:
        logger.info("Configuration:")
        for k, v in asdict(config).items():
            logger.info(f"  {k}: {v}")
        logger.info("")

    trainer = CSMLoRATrainer(config, rank=rank, world_size=world_size)

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
