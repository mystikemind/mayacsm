#!/usr/bin/env python3
"""
Orpheus 3B LoRA Fine-Tuning Script (Production v2)
====================================================

Trains Orpheus 3B with LoRA for Maya voice quality improvement.
All anti-overfitting measures are baked in from deep research.

Architecture:
- Base: canopylabs/orpheus-3b-0.1-ft (LlamaForCausalLM, 28 layers)
- LoRA: r=64, alpha=64, all 7 linear modules (q/k/v/o_proj, gate/up/down_proj)
- Training: BF16, gradient checkpointing, AdamW

Anti-overfitting measures (from deep research):
1. Label masking: only train on audio output tokens (text prompt tokens masked with -100)
2. 90/10 train/val split with style stratification
3. UTMOS-based early stopping (patience=3 checkpoints)
4. LoRA dropout 0.05 (sparsity regularizer, arXiv:2404.09610)
5. Label smoothing 0.1 (prevents overconfident audio token predictions)
6. Learning rate 1e-4 (safer for small datasets, Raschka research)
7. Cosine LR schedule with 5% warmup
8. Weight decay 0.01 (regularizes toward base model weights with LoRA)
9. Checkpoints every 100 steps for fine-grained selection
10. Validation loss tracking + UTMOS tracking for dual overfitting detection

Key design decisions:
- 16-bit LoRA (NOT 4-bit) - research shows better TTS quality
- Do NOT modify embed_tokens/lm_head (tie_word_embeddings=true)
- Only train on audio tokens (text-to-audio mapping learned via context)
- Best checkpoint selected by UTMOS score, NOT lowest train loss

Usage:
    CUDA_VISIBLE_DEVICES=3 python 21_train_orpheus_lora.py --gpu 3 --lr 1e-4 --epochs 3
    CUDA_VISIBLE_DEVICES=3 python 21_train_orpheus_lora.py --gpu 3 --lr 5e-5 --epochs 2

Requirements:
    - GPU with >=20GB VRAM (16-bit model + LoRA + optimizer states)
    - Prepared data from 20_prepare_orpheus_data.py
"""

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE any imports that might initialize CUDA
# torch lazy-initializes CUDA on first use, but we must set this before import torch
import os
import sys

# Parse --gpu from argv early, before importing torch
_gpu = 3  # default
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = int(sys.argv[i + 1])
        break

os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/.cache/huggingface"
_cudnn = os.path.join(os.path.dirname(os.__file__), "..", "nvidia", "cudnn", "lib")
if os.path.isdir(_cudnn):
    os.environ["LD_LIBRARY_PATH"] = _cudnn + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import argparse
import json
import logging
import time
import math
import gc
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
DATA_DIR = PROJECT_ROOT / "training" / "data" / "orpheus_finetune"
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints" / "orpheus_lora"

# Special token IDs (official Orpheus)
PAD_TOKEN = 128263
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MAX = 128266 + (7 * 4096) - 1


# =============================================================================
# DATASET WITH LABEL MASKING
# =============================================================================

class OrpheusDataset(Dataset):
    """Dataset for Orpheus fine-tuning with proper label masking.

    Label masking strategy (from research):
    - Set all text/prompt tokens to -100 (ignored by CrossEntropyLoss)
    - Only compute loss on audio output tokens + termination tokens
    - This focuses learning capacity on audio generation, not text prediction

    Masking:
        input_ids: [BOS, START_HUMAN, text..., END_HUMAN, START_AI, START_SPEECH, audio..., END_SPEECH, END_AI]
        labels:    [-100, -100,        -100..., -100,      -100,     -100,         audio..., END_SPEECH, END_AI]
    """

    def __init__(self, data_path: str, max_seq_length: int = 8192, mask_text_labels: bool = True):
        self.max_seq_length = max_seq_length
        self.mask_text_labels = mask_text_labels
        self.samples = []

        logger.info(f"Loading training data from {data_path}...")
        with open(data_path) as f:
            for line in f:
                sample = json.loads(line.strip())
                if len(sample["input_ids"]) <= max_seq_length:
                    self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples (max_seq={max_seq_length})")

        lengths = [len(s["input_ids"]) for s in self.samples]
        logger.info(f"  Seq length: mean={np.mean(lengths):.0f}, "
                     f"median={np.median(lengths):.0f}, "
                     f"max={max(lengths)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)

        # Build labels with masking
        labels = input_ids.clone()

        if self.mask_text_labels:
            # Find START_OF_SPEECH position and mask everything up to and including it
            speech_positions = (input_ids == START_OF_SPEECH).nonzero(as_tuple=True)[0]
            if len(speech_positions) > 0:
                mask_end = speech_positions[0].item() + 1  # +1 to include START_OF_SPEECH itself
                labels[:mask_end] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def split_dataset(dataset: OrpheusDataset, val_ratio: float = 0.1, seed: int = 42):
    """Split dataset into train and validation with style stratification.

    Stratifies by style to ensure each style is proportionally represented
    in both train and validation sets.
    """
    # Group samples by style (read from metadata if available, else just random split)
    metadata_path = Path(dataset.samples[0].get("_metadata_path", "")) if dataset.samples else None

    # Try to load metadata for style info
    data_dir = Path(os.path.dirname(dataset.samples[0].get("_source", ""))) if dataset.samples else DATA_DIR
    meta_path = DATA_DIR / "metadata.json"

    style_indices = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        if len(metadata) == len(dataset.samples):
            for i, meta in enumerate(metadata):
                style = meta.get("style", "unknown")
                if style not in style_indices:
                    style_indices[style] = []
                style_indices[style].append(i)
            logger.info(f"Stratified split by {len(style_indices)} styles")
        else:
            logger.warning(f"Metadata length ({len(metadata)}) != dataset length ({len(dataset.samples)})")

    # If no metadata or mismatch, do random split
    if not style_indices:
        indices = list(range(len(dataset.samples)))
        random.seed(seed)
        random.shuffle(indices)
        val_size = int(len(indices) * val_ratio)
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])
    else:
        # Stratified split
        train_indices = set()
        val_indices = set()
        random.seed(seed)

        for style, indices in style_indices.items():
            random.shuffle(indices)
            val_size = max(1, int(len(indices) * val_ratio))  # At least 1 per style
            val_indices.update(indices[:val_size])
            train_indices.update(indices[val_size:])

    # Create split datasets
    train_dataset = OrpheusDataset.__new__(OrpheusDataset)
    train_dataset.max_seq_length = dataset.max_seq_length
    train_dataset.mask_text_labels = dataset.mask_text_labels
    train_dataset.samples = [dataset.samples[i] for i in sorted(train_indices)]

    val_dataset = OrpheusDataset.__new__(OrpheusDataset)
    val_dataset.max_seq_length = dataset.max_seq_length
    val_dataset.mask_text_labels = dataset.mask_text_labels
    val_dataset.samples = [dataset.samples[i] for i in sorted(val_indices)]

    logger.info(f"Split: {len(train_dataset)} train / {len(val_dataset)} val "
                f"({val_ratio*100:.0f}% validation)")

    return train_dataset, val_dataset


class PaddingCollator:
    """Dynamic padding collator - pads to longest sequence in batch."""

    def __init__(self, pad_token_id: int = PAD_TOKEN, max_length: int = 8192):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length
        )

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            attention_mask = f["attention_mask"]

            # Truncate if needed
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            attention_mask = attention_mask[:max_len]

            # Pad
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_mask),
        }


# =============================================================================
# UTMOS EVALUATOR
# =============================================================================

class UTMOSEvaluator:
    """Evaluates checkpoint quality using UTMOS score."""

    def __init__(self, device: str):
        self.device = device

        # Load UTMOS
        logger.info("Loading UTMOS model...")
        try:
            self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            self.utmos = self.utmos.to(self.device).eval()
            self.has_utmos = True
        except Exception as e:
            logger.warning(f"UTMOS not available: {e}")
            self.has_utmos = False

        # Load SNAC for decoding
        logger.info("Loading SNAC for evaluation...")
        from snac import SNAC
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

    def decode_tokens_to_audio(self, token_ids: list) -> Optional[torch.Tensor]:
        """Decode SNAC tokens to audio waveform."""
        n = (len(token_ids) // 7) * 7
        if n < 7:
            return None

        token_ids = token_ids[:n]
        l0, l1, l2 = [], [], []
        for i in range(n // 7):
            b = 7 * i
            l0.append(max(0, min(4095, token_ids[b] - AUDIO_TOKEN_BASE)))
            l1.append(max(0, min(4095, token_ids[b+1] - AUDIO_TOKEN_BASE - 4096)))
            l2.append(max(0, min(4095, token_ids[b+2] - AUDIO_TOKEN_BASE - 2*4096)))
            l2.append(max(0, min(4095, token_ids[b+3] - AUDIO_TOKEN_BASE - 3*4096)))
            l1.append(max(0, min(4095, token_ids[b+4] - AUDIO_TOKEN_BASE - 4*4096)))
            l2.append(max(0, min(4095, token_ids[b+5] - AUDIO_TOKEN_BASE - 5*4096)))
            l2.append(max(0, min(4095, token_ids[b+6] - AUDIO_TOKEN_BASE - 6*4096)))

        codes = [
            torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(self.device),
            torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(self.device),
            torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(self.device),
        ]
        with torch.inference_mode():
            audio = self.snac.decode(codes)
        return audio.squeeze().cpu()

    def score_audio(self, audio: torch.Tensor) -> Optional[float]:
        """Compute UTMOS score for audio tensor."""
        if not self.has_utmos or audio is None:
            return None
        try:
            import torchaudio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            resampled = torchaudio.functional.resample(audio, 24000, 16000)
            resampled = resampled.to(self.device)
            with torch.inference_mode():
                score = self.utmos(resampled, 16000)
            return float(score.item())
        except Exception:
            return None

    def evaluate_model(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        voice_name: str = "maya",
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Dict:
        """Generate speech from test prompts and evaluate quality."""
        import torchaudio

        BOS = 128000

        scores = []
        for prompt_text in test_prompts:
            text = f"{voice_name}: {prompt_text}"
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            input_ids = [BOS, START_OF_HUMAN] + text_ids + [END_OF_HUMAN, START_OF_AI]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.device)

            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.15,
                    do_sample=True,
                    pad_token_id=PAD_TOKEN,
                    eos_token_id=END_OF_SPEECH,
                )

            # Extract generated audio tokens
            generated = output[0][len(input_ids):].tolist()
            audio_tokens = []
            for t in generated:
                if t == END_OF_SPEECH:
                    break
                if AUDIO_TOKEN_BASE <= t <= AUDIO_TOKEN_MAX:
                    audio_tokens.append(t)

            if len(audio_tokens) >= 7:
                audio = self.decode_tokens_to_audio(audio_tokens)
                if audio is not None and audio.numel() > 0:
                    score = self.score_audio(audio)
                    if score is not None:
                        scores.append(score)

        if scores:
            return {
                "utmos_mean": float(np.mean(scores)),
                "utmos_std": float(np.std(scores)),
                "utmos_min": float(min(scores)),
                "utmos_max": float(max(scores)),
                "n_evaluated": len(scores),
                "n_total": len(test_prompts),
            }
        return {"utmos_mean": 0.0, "n_evaluated": 0, "n_total": len(test_prompts)}


# =============================================================================
# EVALUATION + EARLY STOPPING CALLBACK
# =============================================================================

class UTMOSEvalAndEarlyStoppingCallback(TrainerCallback):
    """Evaluate UTMOS at each checkpoint + early stopping.

    Combines UTMOS evaluation with early stopping to avoid overfitting.
    Early stopping triggers if UTMOS doesn't improve for `patience` consecutive evals.
    Also tracks train loss patterns for overfitting detection.
    """

    def __init__(
        self,
        evaluator: UTMOSEvaluator,
        tokenizer,
        test_prompts: List[str],
        voice_name: str,
        save_dir: str,
        patience: int = 3,
        min_delta: float = 0.02,
    ):
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.voice_name = voice_name
        self.save_dir = Path(save_dir)
        self.patience = patience
        self.min_delta = min_delta

        self.best_utmos = 0.0
        self.best_step = 0
        self.wait = 0
        self.results = []
        self.train_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track training loss for overfitting detection."""
        if logs and "loss" in logs:
            self.train_losses.append({
                "step": state.global_step,
                "loss": logs["loss"],
            })
            # Warn if train loss drops suspiciously low
            if logs["loss"] < 0.5:
                logger.warning(
                    f"  OVERFITTING WARNING: Train loss={logs['loss']:.3f} at step {state.global_step} "
                    f"is suspiciously low. Model may be memorizing training sequences."
                )

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        step = state.global_step
        logger.info(f"\n[Eval @ step {step}] Generating {len(self.test_prompts)} test utterances...")

        t0 = time.time()
        result = self.evaluator.evaluate_model(
            model=model,
            tokenizer=self.tokenizer,
            test_prompts=self.test_prompts,
            voice_name=self.voice_name,
        )
        eval_time = time.time() - t0

        result["step"] = step
        result["eval_time"] = eval_time
        self.results.append(result)

        utmos = result.get("utmos_mean", 0)
        logger.info(
            f"  UTMOS: {utmos:.3f} +/- {result.get('utmos_std', 0):.3f} "
            f"(range: {result.get('utmos_min', 0):.3f}-{result.get('utmos_max', 0):.3f}) "
            f"[{result.get('n_evaluated', 0)}/{result.get('n_total', 0)} samples] "
            f"in {eval_time:.1f}s"
        )

        # Early stopping logic
        if utmos > self.best_utmos + self.min_delta:
            self.best_utmos = utmos
            self.best_step = step
            self.wait = 0
            logger.info(f"  *** NEW BEST UTMOS: {utmos:.3f} at step {step} ***")
        else:
            self.wait += 1
            logger.info(
                f"  No improvement for {self.wait}/{self.patience} evals "
                f"(best={self.best_utmos:.3f} at step {self.best_step})"
            )
            if self.wait >= self.patience:
                logger.info(
                    f"\n  EARLY STOPPING: UTMOS hasn't improved for {self.patience} evaluations. "
                    f"Best UTMOS={self.best_utmos:.3f} at step {self.best_step}. "
                    f"Stopping to prevent overfitting."
                )
                control.should_training_stop = True

        # Save eval results
        results_path = self.save_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "evaluations": self.results,
                "train_losses": self.train_losses[-50:],  # Last 50 loss entries
                "best_utmos": self.best_utmos,
                "best_step": self.best_step,
                "early_stopped": self.wait >= self.patience,
            }, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Orpheus 3B LoRA (Production v2)")
    parser.add_argument("--gpu", type=int, default=3, help="GPU index for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4, research-backed)")
    parser.add_argument("--epochs", type=int, default=3, help="Max epochs (early stopping may stop sooner)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (regularizer)")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence length")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio (5% of total steps)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--eval-prompts", type=int, default=15, help="Number of eval prompts for UTMOS")
    parser.add_argument("--skip-eval", action="store_true", help="Skip UTMOS evaluation")
    parser.add_argument("--early-stop-patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--no-label-masking", action="store_true", help="Disable label masking (train on full sequence)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else CHECKPOINT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # CUDA_VISIBLE_DEVICES already set at top of script (before torch import)
    device = "cuda:0"  # Physical GPU args.gpu is mapped to cuda:0

    logger.info("=" * 80)
    logger.info("  ORPHEUS 3B LoRA FINE-TUNING (Production v2)")
    logger.info("=" * 80)
    logger.info(f"  GPU: {args.gpu} (physical) -> cuda:0 (visible)")
    logger.info(f"  LR: {args.lr} | Epochs: {args.epochs} (max, early stopping enabled)")
    logger.info(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"  Batch: {args.batch_size} x {args.grad_accum} grad_accum = {args.batch_size * args.grad_accum} effective")
    logger.info(f"  Max seq: {args.max_seq_length} | Save every: {args.save_steps} steps")
    logger.info(f"  Label smoothing: {args.label_smoothing} | Label masking: {not args.no_label_masking}")
    logger.info(f"  Val ratio: {args.val_ratio} | Early stop patience: {args.early_stop_patience}")
    logger.info(f"  Data: {data_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 80)

    # Load dataset
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Run 20_prepare_orpheus_data.py first!")
        return

    full_dataset = OrpheusDataset(
        str(train_path),
        max_seq_length=args.max_seq_length,
        mask_text_labels=not args.no_label_masking,
    )

    # Split into train and validation
    train_dataset, val_dataset = split_dataset(
        full_dataset,
        val_ratio=args.val_ratio,
        seed=42,
    )

    # Calculate training steps
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info(f"\nTraining plan:")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Steps/epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Checkpoints: ~{total_steps // args.save_steps}")

    # Load model
    logger.info("\nLoading Orpheus 3B base model (BF16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Map to the visible GPU
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
        attn_implementation="sdpa",
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  VRAM: ~{torch.cuda.memory_allocated() / 1e9:.1f}GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = PAD_TOKEN

    # Apply LoRA with dropout for regularization
    logger.info(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,  # 0.05 for regularization (arXiv:2404.09610)
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Setup evaluator with early stopping
    eval_callback = None
    test_prompts = [
        "yeah im doing pretty good, hows everything with you",
        "oh thats so cool, tell me more about that",
        "aww that sounds really tough, im sorry youre dealing with that",
        "oh my gosh thats hilarious, i cant believe that happened",
        "hmm yeah that makes sense, ive been thinking about that too",
        "honestly i think thats a great idea, you should definitely go for it",
        "yeah i know what you mean, sometimes things just dont work out",
        "oh really",
        "yeah for sure",
        "hmm thats interesting",
        "well you know what i think, lets just go for it",
        "that reminds me of something that happened last week",
        "mhm",
        "wow thats amazing",
        "okay cool, thanks for letting me know",
    ][:args.eval_prompts]

    if not args.skip_eval:
        try:
            evaluator = UTMOSEvaluator(device=device)
            eval_callback = UTMOSEvalAndEarlyStoppingCallback(
                evaluator=evaluator,
                tokenizer=tokenizer,
                test_prompts=test_prompts,
                voice_name="maya",
                save_dir=output_dir,
                patience=args.early_stop_patience,
                min_delta=0.02,
            )
            logger.info(f"UTMOS evaluator + early stopping ready "
                        f"({len(test_prompts)} prompts, patience={args.early_stop_patience})")
        except Exception as e:
            logger.warning(f"Could not setup evaluator: {e}")
            logger.warning("Training will proceed without UTMOS evaluation")

    # Training arguments (production-level anti-overfitting)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,            # Must be 1 - label smoothing log_softmax on 156K vocab OOMs at batch>1
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,                       # Regularizes toward base model weights
        label_smoothing_factor=args.label_smoothing,  # 0.1 - prevents overconfident predictions
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=8,                      # Keep more checkpoints for selection
        eval_strategy="steps",
        eval_steps=args.save_steps,              # Eval at same frequency as save
        eval_accumulation_steps=8,               # Accumulate eval predictions to save memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
        seed=42,
        load_best_model_at_end=True,             # Load best model (by eval_loss) at end
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Setup callbacks
    callbacks = []
    if eval_callback:
        callbacks.append(eval_callback)

    # Create trainer with validation set
    collator = PaddingCollator(pad_token_id=PAD_TOKEN, max_length=args.max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("  STARTING TRAINING")
    logger.info("  Anti-overfitting measures active:")
    logger.info(f"    - Label masking: {not args.no_label_masking}")
    logger.info(f"    - Label smoothing: {args.label_smoothing}")
    logger.info(f"    - LoRA dropout: {args.lora_dropout}")
    logger.info(f"    - Train/val split: {len(train_dataset)}/{len(val_dataset)}")
    logger.info(f"    - Early stopping: patience={args.early_stop_patience}")
    logger.info(f"    - Weight decay: 0.01 (toward base model)")
    logger.info(f"    - Max grad norm: 1.0")
    logger.info("=" * 80)

    t0 = time.time()
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Save final model
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Final LoRA adapter saved to {final_dir}")

    # Print evaluation summary
    if eval_callback and eval_callback.results:
        logger.info("\n" + "=" * 80)
        logger.info("  EVALUATION SUMMARY")
        logger.info("=" * 80)
        for r in eval_callback.results:
            marker = " <-- BEST" if r.get("utmos_mean", 0) == eval_callback.best_utmos else ""
            logger.info(
                f"  Step {r['step']:>5}: UTMOS={r['utmos_mean']:.3f} +/- {r.get('utmos_std', 0):.3f} "
                f"({r.get('n_evaluated', 0)}/{r.get('n_total', 0)} samples){marker}"
            )
        logger.info(f"\n  Best UTMOS: {eval_callback.best_utmos:.3f} at step {eval_callback.best_step}")

        if eval_callback.wait >= eval_callback.patience:
            logger.info(f"  Training was EARLY STOPPED (patience={eval_callback.patience})")
            logger.info(f"  Use checkpoint at step {eval_callback.best_step} for deployment")
        else:
            logger.info(f"  Training completed all {args.epochs} epochs")

    # Save training config
    config = {
        "model": "canopylabs/orpheus-3b-0.1-ft",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "warmup_steps": warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "label_smoothing": args.label_smoothing,
        "weight_decay": 0.01,
        "label_masking": not args.no_label_masking,
        "val_ratio": args.val_ratio,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_steps": total_steps,
        "training_time_min": elapsed / 60,
        "gpu": args.gpu,
        "early_stopping_patience": args.early_stop_patience,
        "anti_overfitting_measures": [
            "label_masking (text tokens excluded from loss)",
            f"label_smoothing={args.label_smoothing}",
            f"lora_dropout={args.lora_dropout}",
            f"train_val_split={1-args.val_ratio:.0%}/{args.val_ratio:.0%}",
            f"early_stopping (patience={args.early_stop_patience})",
            "weight_decay=0.01 (toward base model)",
            "max_grad_norm=1.0",
            f"cosine_lr with {args.warmup_ratio:.0%} warmup",
        ],
    }

    if eval_callback and eval_callback.results:
        config["best_utmos"] = eval_callback.best_utmos
        config["best_step"] = eval_callback.best_step
        config["early_stopped"] = eval_callback.wait >= eval_callback.patience

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Checkpoints: {output_dir}")
    logger.info(f"  Final adapter: {final_dir}")
    if eval_callback:
        logger.info(f"  Best UTMOS: {eval_callback.best_utmos:.3f} (step {eval_callback.best_step})")
        logger.info(f"  Use best checkpoint for merge+GGUF conversion:")
        logger.info(f"    python 22_merge_and_convert.py --checkpoint {output_dir}/checkpoint-{eval_callback.best_step}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
