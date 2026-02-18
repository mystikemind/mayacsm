#!/usr/bin/env python3
"""
Orpheus 3B LoRA Fine-Tuning (SOTA v3)
======================================

Rebuilt from scratch based on exhaustive research:
- Official Orpheus config: lr=5e-5, epochs=1, batch_size=1
- Unsloth TTS guide: lr=2e-4 for LoRA, r=16-32, 16-bit (NOT 4-bit)
- Full autoregressive labels (labels=input_ids) - official approach
- NO label smoothing (not in official)
- Production eval params (temp=0.6, top_p=0.9, rep=1.1)

Key changes from v2:
1. lr=2e-4 (Unsloth LoRA recommendation, NOT 1e-4)
2. r=32 (good for ~800 samples, vs r=64 which is overkill)
3. Full autoregressive (NO label masking - matches official)
4. NO label smoothing (0.0, matches official)
5. Eval with production params (NOT temp=0.8)
6. 1 epoch (official recommendation, prevents overfitting)
7. Gradient accumulation 4 (effective batch=4)

Usage:
    CUDA_VISIBLE_DEVICES=3 python 25_train_orpheus_lora_v3.py --gpu 3
"""

# CRITICAL: Set env before any imports
import os
import sys

_gpu = 3
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
import random
from pathlib import Path
from typing import Optional, Dict, List
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
DEFAULT_DATA_DIR = PROJECT_ROOT / "training" / "data" / "orpheus_finetune_v3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "checkpoints" / "orpheus_lora_v3"

# Token IDs
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
# DATASET (Full Autoregressive - Official Orpheus)
# =============================================================================

class OrpheusDataset(Dataset):
    """Dataset with full autoregressive labels (official Orpheus approach).

    labels = input_ids (train on entire sequence including text).
    This is intentional: "text token training boosts TTS performance
    while preserving semantic reasoning ability."

    Source: Official Orpheus finetune/train.py, Unsloth notebook
    """

    def __init__(self, data_path: str, max_seq_length: int = 8192):
        self.max_seq_length = max_seq_length
        self.samples = []

        logger.info(f"Loading training data from {data_path}...")
        with open(data_path) as f:
            for line in f:
                sample = json.loads(line.strip())
                if len(sample["input_ids"]) <= max_seq_length:
                    self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples (max_seq={max_seq_length})")
        lengths = [len(s["input_ids"]) for s in self.samples]
        logger.info(f"  Seq length: mean={np.mean(lengths):.0f}, max={max(lengths)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        # Full autoregressive: labels = input_ids (official)
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def split_dataset(dataset: OrpheusDataset, val_ratio: float = 0.1, seed: int = 42,
                  metadata_path: Optional[str] = None):
    """Split with style stratification."""
    style_indices = {}

    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        if len(metadata) == len(dataset.samples):
            for i, meta in enumerate(metadata):
                style = meta.get("style", "unknown")
                style_indices.setdefault(style, []).append(i)
            logger.info(f"Stratified split by {len(style_indices)} styles")

    if not style_indices:
        indices = list(range(len(dataset.samples)))
        random.seed(seed)
        random.shuffle(indices)
        val_size = int(len(indices) * val_ratio)
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])
    else:
        train_indices, val_indices = set(), set()
        random.seed(seed)
        for style, indices in style_indices.items():
            random.shuffle(indices)
            val_size = max(1, int(len(indices) * val_ratio))
            val_indices.update(indices[:val_size])
            train_indices.update(indices[val_size:])

    train_ds = OrpheusDataset.__new__(OrpheusDataset)
    train_ds.max_seq_length = dataset.max_seq_length
    train_ds.samples = [dataset.samples[i] for i in sorted(train_indices)]

    val_ds = OrpheusDataset.__new__(OrpheusDataset)
    val_ds.max_seq_length = dataset.max_seq_length
    val_ds.samples = [dataset.samples[i] for i in sorted(val_indices)]

    logger.info(f"Split: {len(train_ds)} train / {len(val_ds)} val")
    return train_ds, val_ds


class PaddingCollator:
    """Dynamic padding to longest in batch."""

    def __init__(self, pad_token_id: int = PAD_TOKEN, max_length: int = 8192):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            for key in batch:
                t = f[key][:max_len]
                pad_len = max_len - len(t)
                if pad_len > 0:
                    pad_val = self.pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
                    t = torch.cat([t, torch.full((pad_len,), pad_val, dtype=torch.long)])
                batch[key].append(t)

        return {k: torch.stack(v) for k, v in batch.items()}


# =============================================================================
# UTMOS EVALUATOR
# =============================================================================

class UTMOSEvaluator:
    """Evaluate checkpoints with production-matched inference parameters."""

    def __init__(self, device: str):
        self.device = device

        logger.info("Loading UTMOS + SNAC for evaluation...")
        try:
            self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            self.utmos = self.utmos.to(device).eval()
            self.has_utmos = True
        except Exception as e:
            logger.warning(f"UTMOS not available: {e}")
            self.has_utmos = False

        from snac import SNAC
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    def decode_tokens(self, token_ids: list) -> Optional[torch.Tensor]:
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
        if not self.has_utmos or audio is None:
            return None
        try:
            import torchaudio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            resampled = torchaudio.functional.resample(audio, 24000, 16000).to(self.device)
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
    ) -> Dict:
        """Generate and evaluate with PRODUCTION parameters.

        CRITICAL: Use temp=0.6, top_p=0.9, rep=1.1 (same as production).
        Previous versions used temp=0.8/top_p=0.95/rep=1.15 which gave
        non-comparable UTMOS scores.
        """
        import torchaudio

        scores = []
        for prompt_text in test_prompts:
            text = f"{voice_name}: {prompt_text}"
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            input_ids = [128000, START_OF_HUMAN] + text_ids + [END_OF_HUMAN, START_OF_AI]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.device)

            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_tensor,
                    max_new_tokens=max_new_tokens,
                    # PRODUCTION PARAMETERS (community-proven stable)
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=PAD_TOKEN,
                    eos_token_id=END_OF_SPEECH,
                )

            generated = output[0][len(input_ids):].tolist()
            audio_tokens = []
            for t in generated:
                if t == END_OF_SPEECH:
                    break
                if AUDIO_TOKEN_BASE <= t <= AUDIO_TOKEN_MAX:
                    audio_tokens.append(t)

            if len(audio_tokens) >= 7:
                audio = self.decode_tokens(audio_tokens)
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
# CALLBACKS
# =============================================================================

class UTMOSEvalCallback(TrainerCallback):
    """UTMOS evaluation + early stopping at each checkpoint."""

    def __init__(self, evaluator, tokenizer, test_prompts, voice_name, save_dir, patience=3):
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.voice_name = voice_name
        self.save_dir = Path(save_dir)
        self.patience = patience

        self.best_utmos = 0.0
        self.best_step = 0
        self.wait = 0
        self.results = []
        self.train_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append({"step": state.global_step, "loss": logs["loss"]})

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        step = state.global_step
        logger.info(f"\n[Eval @ step {step}] Generating {len(self.test_prompts)} test utterances (production params)...")

        t0 = time.time()
        result = self.evaluator.evaluate_model(
            model=model, tokenizer=self.tokenizer,
            test_prompts=self.test_prompts, voice_name=self.voice_name,
        )
        eval_time = time.time() - t0

        result["step"] = step
        result["eval_time"] = eval_time
        self.results.append(result)

        utmos = result.get("utmos_mean", 0)
        logger.info(
            f"  UTMOS: {utmos:.3f} +/- {result.get('utmos_std', 0):.3f} "
            f"(range: {result.get('utmos_min', 0):.3f}-{result.get('utmos_max', 0):.3f}) "
            f"[{result.get('n_evaluated', 0)}/{result.get('n_total', 0)}] in {eval_time:.1f}s"
        )

        if utmos > self.best_utmos + 0.02:
            self.best_utmos = utmos
            self.best_step = step
            self.wait = 0
            logger.info(f"  *** NEW BEST UTMOS: {utmos:.3f} at step {step} ***")
        else:
            self.wait += 1
            logger.info(f"  No improvement {self.wait}/{self.patience} (best={self.best_utmos:.3f} @ step {self.best_step})")
            if self.wait >= self.patience:
                logger.info(f"\n  EARLY STOPPING at step {step}. Best={self.best_utmos:.3f} @ step {self.best_step}")
                control.should_training_stop = True

        # Save results
        with open(self.save_dir / "eval_results.json", "w") as f:
            json.dump({
                "evaluations": self.results,
                "train_losses": self.train_losses[-50:],
                "best_utmos": self.best_utmos,
                "best_step": self.best_step,
                "early_stopped": self.wait >= self.patience,
                "eval_params": "PRODUCTION: temp=0.6, top_p=0.9, rep_penalty=1.1",
            }, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Orpheus LoRA SOTA v3")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4, help="LR (2e-4 = Unsloth LoRA recommendation)")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs (1 = official recommendation)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (32 for ~800 samples)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (= rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint every N steps (more granular)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-prompts", type=int, default=15)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (more patient with 1 epoch)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"

    logger.info("=" * 80)
    logger.info("  ORPHEUS 3B LoRA FINE-TUNING (SOTA v3)")
    logger.info("=" * 80)
    logger.info(f"  GPU: {args.gpu} -> cuda:0")
    logger.info(f"  LR: {args.lr} (Unsloth LoRA recommendation)")
    logger.info(f"  Epochs: {args.epochs} (official recommendation)")
    logger.info(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"  Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    logger.info(f"  Labels: full autoregressive (official, NO masking)")
    logger.info(f"  Label smoothing: 0.0 (official, NONE)")
    logger.info(f"  Eval params: PRODUCTION (temp=0.6, top_p=0.9, rep=1.1)")
    logger.info(f"  Data: {data_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 80)

    # Load dataset
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Run 24_prepare_orpheus_data_v3.py first!")
        return

    full_dataset = OrpheusDataset(str(train_path), max_seq_length=args.max_seq_length)

    metadata_path = data_dir / "metadata.json"
    train_dataset, val_dataset = split_dataset(
        full_dataset, val_ratio=args.val_ratio, seed=42,
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
    )

    # Training step calculations
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info(f"\nTraining plan:")
    logger.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    logger.info(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}")
    logger.info(f"  Checkpoints: ~{total_steps // args.save_steps}")

    # Load model (from fine-tuned base for better starting point)
    logger.info("\nLoading Orpheus 3B (orpheus-3b-0.1-ft, BF16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
        attn_implementation="sdpa",
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s ({torch.cuda.memory_allocated()/1e9:.1f}GB)")

    tokenizer = AutoTokenizer.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = PAD_TOKEN

    # Apply LoRA
    logger.info(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Eval prompts (with MIXED CASE to match training data format)
    test_prompts = [
        "Yeah I'm doing pretty good, how's everything with you?",
        "Oh that's so cool, tell me more about that.",
        "Aww that sounds really tough, I'm sorry you're dealing with that.",
        "Oh my gosh that's hilarious, I can't believe that happened!",
        "Hmm yeah that makes sense, I've been thinking about that too.",
        "Honestly I think that's a great idea, you should definitely go for it.",
        "Yeah I know what you mean, sometimes things just don't work out.",
        "Oh really?",
        "Yeah for sure.",
        "Hmm that's interesting.",
        "Well you know what I think, let's just go for it!",
        "That reminds me of something that happened last week.",
        "Mhm.",
        "Wow that's amazing!",
        "Okay cool, thanks for letting me know.",
    ][:args.eval_prompts]

    eval_callback = None
    if not args.skip_eval:
        try:
            evaluator = UTMOSEvaluator(device=device)
            eval_callback = UTMOSEvalCallback(
                evaluator=evaluator, tokenizer=tokenizer,
                test_prompts=test_prompts, voice_name="maya",
                save_dir=output_dir, patience=args.patience,
            )
            logger.info(f"UTMOS evaluator ready ({len(test_prompts)} prompts, patience={args.patience})")
        except Exception as e:
            logger.warning(f"Evaluator setup failed: {e}")

    # Training arguments (SOTA v3)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        label_smoothing_factor=0.0,  # NONE (official approach)
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        eval_accumulation_steps=8,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = [eval_callback] if eval_callback else []
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
    logger.info("  STARTING TRAINING (SOTA v3)")
    logger.info("  Key differences from v2:")
    logger.info("    - Full autoregressive labels (NO text masking)")
    logger.info("    - lr=2e-4 (Unsloth LoRA, NOT 1e-4)")
    logger.info("    - r=32 (NOT r=64)")
    logger.info("    - NO label smoothing (was 0.1)")
    logger.info("    - Production eval params (temp=0.6/top_p=0.9/rep=1.1)")
    logger.info("    - 1 epoch (official, NOT 3)")
    logger.info("    - LUFS-normalized + HPF data (v3 pipeline)")
    logger.info("=" * 80)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    logger.info(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Save final
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Final LoRA adapter saved to {final_dir}")

    # Summary
    if eval_callback and eval_callback.results:
        logger.info("\n" + "=" * 80)
        logger.info("  EVALUATION SUMMARY")
        logger.info("=" * 80)
        for r in eval_callback.results:
            marker = " <-- BEST" if r.get("utmos_mean", 0) == eval_callback.best_utmos else ""
            logger.info(
                f"  Step {r['step']:>5}: UTMOS={r['utmos_mean']:.3f} +/- {r.get('utmos_std', 0):.3f}{marker}"
            )
        logger.info(f"\n  Best UTMOS: {eval_callback.best_utmos:.3f} at step {eval_callback.best_step}")
        logger.info(f"  Production baseline: 4.391 (must beat this)")

    # Save config
    config = {
        "version": "v3_SOTA",
        "model": "canopylabs/orpheus-3b-0.1-ft",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "label_smoothing": 0.0,
        "label_masking": False,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "val_ratio": args.val_ratio,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_steps": total_steps,
        "training_time_min": elapsed / 60,
        "eval_params": "PRODUCTION: temp=0.6, top_p=0.9, rep_penalty=1.1",
        "data_pipeline": "v3: HPF(80Hz) + LUFS(-12dB) + SNAC + dedup + UTMOS(3.5)",
        "key_changes_from_v2": [
            "Full autoregressive (no label masking)",
            "lr=2e-4 (was 1e-4)",
            "r=32 (was r=64)",
            "label_smoothing=0.0 (was 0.1)",
            "Production eval params (was temp=0.8)",
            "1 epoch (was 3)",
            "LUFS normalized data (was peak normalized)",
            "Mixed case text (was lowercase)",
        ],
    }

    if eval_callback and eval_callback.results:
        config["best_utmos"] = eval_callback.best_utmos
        config["best_step"] = eval_callback.best_step

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("  TRAINING COMPLETE (SOTA v3)")
    logger.info(f"  Output: {output_dir}")
    if eval_callback:
        logger.info(f"  Best UTMOS: {eval_callback.best_utmos:.3f} @ step {eval_callback.best_step}")
        logger.info(f"  Next: python 22_merge_and_convert.py --checkpoint {output_dir}/checkpoint-{eval_callback.best_step}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
