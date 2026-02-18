#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for Orpheus TTS
==========================================================

Pushes TTS quality ABOVE the 4.391 UTMOS production baseline by training
the model to generate better audio through reinforcement learning.

Key insight: SFT degrades quality because SNAC-encoded training targets
(~3.9 UTMOS) are lower quality than what the base model generates (~4.2+).
GRPO trains the model on its own best generations using UTMOS as reward.

Algorithm (per step):
  1. Sample B prompts (grad_accum_steps) from training data
  2. For each prompt, generate G=8 audio completions with current policy
  3. Score each completion with UTMOS (1-5 scale)
  4. Compute group-relative advantages: A_i = (R_i - mean(R)) / std(R)
  5. Forward pass: compute per-token log probs under current policy
  6. Loss = -sum(A_i * log_prob_t * mask_t) / sum(mask)
  7. Accumulate gradients over B prompts, then optimizer step

With mu=1 (single policy update per batch), ratio = pi_theta/pi_old = 1,
so clipping has no effect. This simplifies to advantage-weighted MLE
(REINFORCE with group-relative baseline).

Based on:
- arXiv:2509.18798 (GRPO for TTS with LLMs)
- arXiv:2511.21270 (Multi-Reward GRPO for TTS at Scale)
- GLM-TTS (open-source GRPO for LLM-based TTS)

Usage:
    CUDA_VISIBLE_DEVICES=3 python 28_train_orpheus_grpo.py --gpu 3
"""

# =============================================================================
# ENVIRONMENT SETUP (before any imports)
# =============================================================================

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

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import logging
import time
import random
import gc
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")

# Orpheus token IDs
PAD_TOKEN = 128263
BOS_TOKEN = 128000
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MAX = 128266 + (7 * 4096) - 1


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GRPOConfig:
    """GRPO training configuration.

    Hyperparameters based on:
    - arXiv:2511.21270: G=8-12, beta=0.04-0.1, lr=1e-6, temp=1.1
    - arXiv:2509.18798: G=8, beta=0.1, lr=1e-5
    - GLM-TTS: custom implementation, similar params
    - Adjusted for A10G 23GB memory constraints
    """
    # Model
    model_name: str = "canopylabs/orpheus-3b-0.1-ft"
    voice_name: str = "maya"

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0  # No dropout for RL (want consistent behavior)
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # GRPO generation (higher temp than production for exploration)
    group_size: int = 8           # G: completions per prompt
    max_new_tokens: int = 1024    # ~12 seconds of audio (84 tokens/sec)
    gen_temperature: float = 0.9  # Higher than production (0.6) for diversity
    gen_top_p: float = 0.95
    gen_top_k: int = 50
    gen_repetition_penalty: float = 1.1

    # Optimization
    learning_rate: float = 5e-6   # LoRA GRPO: lower than SFT (was 2e-4)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    grad_accum_steps: int = 4     # Prompts per optimizer step (4*8=32 completions)

    # Training schedule
    num_steps: int = 200          # Total optimizer steps
    eval_every: int = 25          # Eval every N steps
    save_every: int = 50          # Save checkpoint every N steps
    log_every: int = 5            # Log metrics every N steps

    # Evaluation (production params for fair comparison)
    eval_temperature: float = 0.6
    eval_top_p: float = 0.9
    eval_repetition_penalty: float = 1.1
    eval_max_new_tokens: int = 500
    eval_n_prompts: int = 15

    # Data
    prompts_file: str = str(PROJECT_ROOT / "training" / "data" / "grpo_prompts.json")
    output_dir: str = str(PROJECT_ROOT / "training" / "checkpoints" / "orpheus_grpo_v1")

    # Reward
    min_audio_tokens: int = 28    # Minimum 4 SNAC frames (4*7=28 tokens)
    reward_floor: float = 1.0    # Minimum reward for degenerate audio

    # Reproducibility
    seed: int = 42


# =============================================================================
# SNAC DECODER + UTMOS SCORER
# =============================================================================

class AudioRewardScorer:
    """Decodes audio tokens via SNAC and scores with UTMOS.

    Memory usage: SNAC ~0.5GB + UTMOS ~0.3GB = ~0.8GB total
    """

    def __init__(self, device: str):
        self.device = device

        # Load SNAC codec
        logger.info("Loading SNAC 24kHz codec...")
        from snac import SNAC
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        for p in self.snac.parameters():
            p.requires_grad = False

        # Load UTMOS quality predictor
        logger.info("Loading UTMOS quality scorer...")
        self.utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        self.utmos = self.utmos.to(device).eval()
        for p in self.utmos.parameters():
            p.requires_grad = False

        logger.info(f"Reward scorer ready (GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB)")

    def decode_tokens(self, token_ids: list) -> Optional[torch.Tensor]:
        """Decode Orpheus audio tokens to 24kHz waveform via SNAC.

        Orpheus SNAC frame format (7 tokens per frame):
            [L0, L1+4096, L2+8192, L2+12288, L1+16384, L2+20480, L2+24576]

        Args:
            token_ids: Raw Orpheus audio token IDs (128266-based)

        Returns:
            Audio tensor (samples,) at 24kHz, or None if too short
        """
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

    def score_utmos(self, audio: torch.Tensor) -> float:
        """Score audio quality with UTMOS (1-5 scale).

        Args:
            audio: Audio tensor at 24kHz

        Returns:
            UTMOS score (1.0-5.0), or 1.0 for degenerate audio
        """
        if audio is None or audio.numel() < 3200:  # < ~133ms at 24kHz
            return 1.0
        try:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            resampled = torchaudio.functional.resample(audio, 24000, 16000).to(self.device)
            with torch.inference_mode():
                score = self.utmos(resampled, 16000)
            result = float(score.item())
            # Clamp to valid range (safety against NaN/Inf)
            return max(1.0, min(5.0, result))
        except Exception as e:
            logger.debug(f"UTMOS scoring failed: {e}")
            return 1.0

    def score_completions(
        self,
        all_token_ids: List[List[int]],
        min_tokens: int = 28,
    ) -> List[float]:
        """Score a batch of audio token completions with UTMOS.

        Args:
            all_token_ids: List of audio token lists (one per completion)
            min_tokens: Minimum tokens for valid audio

        Returns:
            List of UTMOS scores (one per completion)
        """
        scores = []
        for tokens in all_token_ids:
            if len(tokens) < min_tokens:
                scores.append(1.0)
                continue
            audio = self.decode_tokens(tokens)
            score = self.score_utmos(audio)
            scores.append(score)
        return scores


# =============================================================================
# GRPO TRAINER
# =============================================================================

class OrpheusGRPOTrainer:
    """GRPO trainer for Orpheus TTS quality optimization.

    Custom implementation (not TRL GRPOTrainer) because:
    - TTS-specific SNAC token decoding in reward function
    - Orpheus-specific prompt formatting with special tokens
    - Need fine control over generation and stopping criteria
    - All published TTS GRPO implementations use custom loops

    Memory budget (A10G 23GB):
    - Orpheus 3B BF16:  ~6.4GB
    - LoRA adapters:    ~0.2GB
    - SNAC + UTMOS:     ~0.8GB
    - KV cache (gen):   ~2.8GB (8 sequences * 1024 tokens)
    - Activations:      ~3-4GB (gradient checkpointing)
    - Total peak:       ~13-14GB (fits in 23GB)
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = "cuda:0"
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Load model + LoRA
        self._load_model()

        # Load reward scorer
        self.scorer = AudioRewardScorer(self.device)

        # Load training prompts
        self.prompts = self._load_prompts()

        # Optimizer (LoRA params only)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable_params)
        logger.info(f"Optimizer: AdamW over {n_params:,} trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Eval prompts (same as SFT eval for fair comparison to baselines)
        self.eval_prompts = [
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
        ][:config.eval_n_prompts]

        # Training state
        self.global_step = 0
        self.best_utmos = 0.0
        self.best_step = 0
        self.metrics_history = []

        logger.info(f"Total GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f}GB / "
                     f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

    def _load_model(self):
        """Load Orpheus 3B base model + LoRA adapters."""
        config = self.config

        logger.info(f"Loading {config.model_name} (BF16)...")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
            attn_implementation="sdpa",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = PAD_TOKEN

        logger.info(f"Model loaded in {time.time()-t0:.1f}s ({torch.cuda.memory_allocated()/1e9:.1f}GB)")

        # Apply LoRA
        logger.info(f"Applying LoRA (r={config.lora_r}, alpha={config.lora_alpha}, "
                     f"dropout={config.lora_dropout})...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        logger.info(f"Model + LoRA GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def _load_prompts(self) -> List[Dict]:
        """Load GRPO training prompts."""
        with open(self.config.prompts_file) as f:
            prompts = json.load(f)
        logger.info(f"Loaded {len(prompts)} training prompts from {self.config.prompts_file}")

        # Category distribution
        from collections import Counter
        cats = Counter(p.get("category", "unknown") for p in prompts)
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {count}")

        return prompts

    def format_prompt(self, text: str) -> torch.Tensor:
        """Format text into Orpheus prompt token IDs.

        Format: [BOS, START_HUMAN] + text_tokens + [END_HUMAN, START_OF_AI]
        Text format: "{voice_name}: {text}"
        """
        prompt_text = f"{self.config.voice_name}: {text}"
        text_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = [BOS_TOKEN, START_OF_HUMAN] + text_ids + [END_OF_HUMAN, START_OF_AI]
        return torch.tensor([input_ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate_group(self, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Generate G completions for one prompt.

        Args:
            prompt_ids: (1, T_prompt) prompt token IDs

        Returns:
            full_ids: (G, T_max) - prompt + completion + padding (right-padded)
            prompt_length: int - length of the prompt portion
        """
        G = self.config.group_size
        prompt_length = prompt_ids.shape[1]

        # Replicate prompt G times
        expanded = prompt_ids.expand(G, -1)

        self.model.eval()

        outputs = self.model.generate(
            input_ids=expanded,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.gen_temperature,
            top_p=self.config.gen_top_p,
            top_k=self.config.gen_top_k,
            repetition_penalty=self.config.gen_repetition_penalty,
            do_sample=True,
            pad_token_id=PAD_TOKEN,
            eos_token_id=END_OF_SPEECH,
        )

        return outputs, prompt_length

    def extract_audio_tokens(
        self, full_ids: torch.Tensor, prompt_length: int
    ) -> List[List[int]]:
        """Extract valid audio tokens from generated sequences.

        Stops at END_OF_SPEECH or PAD_TOKEN.
        Filters to valid audio token range [128266, 128266 + 7*4096).
        """
        all_audio_tokens = []
        for i in range(full_ids.shape[0]):
            seq = full_ids[i, prompt_length:].tolist()
            audio_tokens = []
            for t in seq:
                if t == END_OF_SPEECH or t == PAD_TOKEN:
                    break
                if AUDIO_TOKEN_BASE <= t <= AUDIO_TOKEN_MAX:
                    audio_tokens.append(t)
            all_audio_tokens.append(audio_tokens)
        return all_audio_tokens

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """Compute group-relative advantages.

        A_i = (R_i - mean(R_1...R_G)) / (std(R_1...R_G) + eps)

        This normalizes rewards within the group, removing prompt difficulty bias.
        Positive advantages = better than group average.
        Negative advantages = worse than group average.
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()
        # Avoid division by zero when all rewards are identical
        if std_r < 1e-8:
            return torch.zeros_like(rewards_tensor, device=self.device)
        advantages = (rewards_tensor - mean_r) / (std_r + 1e-8)
        return advantages.to(self.device)

    def compute_grpo_loss_and_backward(
        self,
        full_ids: torch.Tensor,
        prompt_length: int,
        advantages: torch.Tensor,
        loss_scale: float = 1.0,
    ) -> Dict:
        """Compute GRPO loss and backward pass, one sequence at a time.

        Memory-efficient: processes each of G sequences independently through
        the model, computes per-token log probs, and immediately backward()s.
        This avoids materializing the full (G, T, V=128K) logits tensor.

        With mu=1 (single iteration), the loss simplifies to:
            L = -sum_i,t(A_i * log pi(a_{i,t} | context) * mask_{i,t}) / sum(mask)

        Memory per sequence: ~260MB logits (BF16) + ~520MB log_probs (FP32)
        vs batched: ~4.2GB logits + ~8.4GB log_probs = OOM

        Args:
            full_ids: (G, T_total) full sequences including prompt + completion
            prompt_length: length of prompt portion
            advantages: (G,) group-relative advantages
            loss_scale: multiply loss by this before backward (for grad accumulation)

        Returns:
            info dict with loss value and debugging metrics
        """
        self.model.train()

        G, T_total = full_ids.shape
        T_comp = T_total - prompt_length

        if T_comp <= 0:
            return {"loss": 0.0, "n_valid_tokens": 0, "mean_log_prob": 0.0}

        # Precompute attention mask and total valid tokens for normalization
        attention_mask = (full_ids != PAD_TOKEN).long()
        total_valid = attention_mask[:, prompt_length:].float().sum().item()
        if total_valid < 1:
            return {"loss": 0.0, "n_valid_tokens": 0, "mean_log_prob": 0.0}

        total_loss_value = 0.0
        total_log_prob_sum = 0.0

        # Process one sequence at a time to avoid OOM on logits (G, T, V=128K)
        for i in range(G):
            # Skip zero-advantage sequences (no gradient signal)
            if abs(advantages[i].item()) < 1e-8:
                continue

            # Forward pass for single sequence (gradient checkpointing active)
            outputs = self.model(
                input_ids=full_ids[i:i+1],
                attention_mask=attention_mask[i:i+1],
            )
            logits = outputs.logits  # (1, T_total, V) ~260MB in BF16

            # Extract completion logits
            shift_logits = logits[:, prompt_length - 1 : T_total - 1, :]  # (1, T_comp, V)
            shift_labels = full_ids[i:i+1, prompt_length:]  # (1, T_comp)

            # Per-token log probs (float32 for numerical stability)
            # This creates a (1, T_comp, V) float32 tensor (~520MB) - fits in memory
            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            per_token_lp = torch.gather(
                log_probs, dim=2, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # (1, T_comp)

            # Free large tensors immediately
            del logits, shift_logits, log_probs, outputs

            # Completion mask for this sequence
            comp_mask = attention_mask[i:i+1, prompt_length:].float()  # (1, T_comp)

            # This sequence's contribution to the total loss
            weighted = advantages[i] * per_token_lp * comp_mask  # (1, T_comp)
            loss_i = -weighted.sum() / total_valid

            # Backward with gradient accumulation scaling
            (loss_i * loss_scale).backward()

            # Track metrics
            total_loss_value += loss_i.item()
            total_log_prob_sum += (per_token_lp.detach() * comp_mask).sum().item()

            del per_token_lp, comp_mask, weighted, loss_i
            torch.cuda.empty_cache()

        return {
            "loss": total_loss_value,
            "n_valid_tokens": int(total_valid),
            "mean_log_prob": total_log_prob_sum / total_valid if total_valid > 0 else 0.0,
        }

    def train_step(self) -> Dict:
        """One complete GRPO optimizer step.

        Processes grad_accum_steps prompts, each with G completions.
        Total completions per step = grad_accum_steps * group_size.
        """
        step_rewards = []
        step_advantages = []
        step_losses = []
        step_audio_lengths = []
        step_log_probs = []
        step_n_valid = 0
        step_n_degenerate = 0

        self.optimizer.zero_grad()

        for accum_idx in range(self.config.grad_accum_steps):
            # Sample random prompt
            prompt_data = random.choice(self.prompts)
            text = prompt_data["text"]

            # Format prompt
            prompt_ids = self.format_prompt(text)

            # Generate G completions
            t_gen = time.time()
            full_ids, prompt_length = self.generate_group(prompt_ids)
            gen_time = time.time() - t_gen

            # Extract audio tokens
            audio_tokens_list = self.extract_audio_tokens(full_ids, prompt_length)
            audio_lengths = [len(t) for t in audio_tokens_list]
            step_audio_lengths.extend(audio_lengths)

            # Count degenerate completions
            n_degen = sum(1 for l in audio_lengths if l < self.config.min_audio_tokens)
            step_n_degenerate += n_degen

            # Score with UTMOS
            t_score = time.time()
            rewards = self.scorer.score_completions(
                audio_tokens_list,
                min_tokens=self.config.min_audio_tokens,
            )
            score_time = time.time() - t_score

            step_rewards.extend(rewards)

            # Compute group-relative advantages
            advantages = self.compute_advantages(rewards)
            step_advantages.extend(advantages.tolist())

            # Skip if all rewards identical (zero advantages, no gradient signal)
            if advantages.abs().max() < 1e-6:
                logger.debug(f"  Accum {accum_idx}: skipped (uniform rewards)")
                del full_ids, advantages
                torch.cuda.empty_cache()
                continue

            # Compute loss and backward pass (one sequence at a time for memory)
            t_loss = time.time()
            loss_info = self.compute_grpo_loss_and_backward(
                full_ids, prompt_length, advantages,
                loss_scale=1.0 / self.config.grad_accum_steps,
            )
            loss_time = time.time() - t_loss

            step_losses.append(loss_info.get("loss", 0))
            step_log_probs.append(loss_info.get("mean_log_prob", 0))
            step_n_valid += loss_info.get("n_valid_tokens", 0)

            # Free memory
            del full_ids, advantages
            torch.cuda.empty_cache()

        # Gradient clipping
        if step_losses:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
        else:
            grad_norm = 0.0

        # Optimizer step
        self.optimizer.step()
        self.global_step += 1

        # Collect metrics
        metrics = {
            "step": self.global_step,
            "loss": float(np.mean(step_losses)) if step_losses else 0.0,
            "reward_mean": float(np.mean(step_rewards)) if step_rewards else 0.0,
            "reward_std": float(np.std(step_rewards)) if step_rewards else 0.0,
            "reward_min": float(min(step_rewards)) if step_rewards else 0.0,
            "reward_max": float(max(step_rewards)) if step_rewards else 0.0,
            "advantage_std": float(np.std(step_advantages)) if step_advantages else 0.0,
            "audio_tokens_mean": float(np.mean(step_audio_lengths)) if step_audio_lengths else 0.0,
            "audio_tokens_min": int(min(step_audio_lengths)) if step_audio_lengths else 0,
            "audio_tokens_max": int(max(step_audio_lengths)) if step_audio_lengths else 0,
            "mean_log_prob": float(np.mean(step_log_probs)) if step_log_probs else 0.0,
            "grad_norm": float(grad_norm) if isinstance(grad_norm, (int, float)) else float(grad_norm.item()),
            "n_completions": len(step_rewards),
            "n_degenerate": step_n_degenerate,
            "gpu_gb": torch.cuda.memory_allocated() / 1e9,
        }

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate current model with production parameters.

        Uses the same prompts as SFT eval for direct comparison:
        - Base model (HF generate, maya voice): 4.206 UTMOS
        - Production (llama-server GGUF): 4.391 UTMOS
        """
        self.model.eval()

        scores = []
        per_prompt = {}

        for i, prompt_text in enumerate(self.eval_prompts):
            prompt_ids = self.format_prompt(prompt_text)

            output = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=self.config.eval_max_new_tokens,
                temperature=self.config.eval_temperature,
                top_p=self.config.eval_top_p,
                repetition_penalty=self.config.eval_repetition_penalty,
                do_sample=True,
                pad_token_id=PAD_TOKEN,
                eos_token_id=END_OF_SPEECH,
            )

            prompt_length = prompt_ids.shape[1]
            audio_tokens = self.extract_audio_tokens(output, prompt_length)[0]

            score = None
            if len(audio_tokens) >= 14:
                audio = self.scorer.decode_tokens(audio_tokens)
                score = self.scorer.score_utmos(audio)

            if score is not None:
                scores.append(score)
                per_prompt[f"p{i:02d}"] = score

            status = f"UTMOS={score:.3f}" if score else "FAILED"
            logger.info(f"  [{i+1:2d}/{len(self.eval_prompts)}] "
                         f"{prompt_text[:45]:45s} → {len(audio_tokens):4d}tok {status}")

        if scores:
            return {
                "utmos_mean": float(np.mean(scores)),
                "utmos_std": float(np.std(scores)),
                "utmos_min": float(min(scores)),
                "utmos_max": float(max(scores)),
                "utmos_median": float(np.median(scores)),
                "n_evaluated": len(scores),
                "n_total": len(self.eval_prompts),
                "per_prompt": per_prompt,
            }
        return {"utmos_mean": 0.0, "n_evaluated": 0, "n_total": len(self.eval_prompts)}

    def save_checkpoint(self, tag: str = None):
        """Save LoRA checkpoint and tokenizer."""
        tag = tag or f"step-{self.global_step}"
        save_path = self.output_dir / f"checkpoint-{tag}"
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        logger.info(f"Checkpoint saved: {save_path}")

    def _save_metrics(self):
        """Save training metrics to JSON."""
        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump({
                "metrics": self.metrics_history,
                "best_utmos": self.best_utmos,
                "best_step": self.best_step,
                "global_step": self.global_step,
                "config": {
                    "model": self.config.model_name,
                    "voice": self.config.voice_name,
                    "group_size": self.config.group_size,
                    "grad_accum": self.config.grad_accum_steps,
                    "effective_completions_per_step": self.config.group_size * self.config.grad_accum_steps,
                    "lr": self.config.learning_rate,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "gen_temp": self.config.gen_temperature,
                    "gen_top_p": self.config.gen_top_p,
                    "gen_top_k": self.config.gen_top_k,
                    "gen_rep_penalty": self.config.gen_repetition_penalty,
                    "eval_temp": self.config.eval_temperature,
                    "eval_top_p": self.config.eval_top_p,
                    "eval_rep_penalty": self.config.eval_repetition_penalty,
                    "max_new_tokens": self.config.max_new_tokens,
                    "num_steps": self.config.num_steps,
                    "warmup_steps": self.config.warmup_steps,
                    "seed": self.config.seed,
                },
                "baselines": {
                    "production_gguf": 4.391,
                    "base_hf_maya": 4.206,
                    "base_hf_tara": 3.660,
                    "sft_v3_step50": 3.942,
                },
            }, f, indent=2)

    def train(self):
        """Full GRPO training loop."""
        cfg = self.config
        eff = cfg.group_size * cfg.grad_accum_steps

        logger.info("\n" + "=" * 80)
        logger.info("  GRPO TRAINING FOR ORPHEUS TTS")
        logger.info("=" * 80)
        logger.info(f"  Model: {cfg.model_name}")
        logger.info(f"  Voice: {cfg.voice_name}")
        logger.info(f"  Steps: {cfg.num_steps}")
        logger.info(f"  Group size (G): {cfg.group_size}")
        logger.info(f"  Grad accumulation: {cfg.grad_accum_steps} prompts")
        logger.info(f"  Effective completions/step: {eff}")
        logger.info(f"  LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}")
        logger.info(f"  LR: {cfg.learning_rate}")
        logger.info(f"  Gen params: temp={cfg.gen_temperature}, top_p={cfg.gen_top_p}, "
                     f"top_k={cfg.gen_top_k}, rep={cfg.gen_repetition_penalty}")
        logger.info(f"  Eval params: temp={cfg.eval_temperature}, top_p={cfg.eval_top_p}, "
                     f"rep={cfg.eval_repetition_penalty}")
        logger.info(f"  Training prompts: {len(self.prompts)}")
        logger.info(f"  Eval prompts: {len(self.eval_prompts)}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 80)

        # Initial evaluation (baseline with LoRA at init weights)
        logger.info("\n[Eval @ step 0] Baseline evaluation (fresh LoRA)...")
        eval_result = self.evaluate()
        utmos_0 = eval_result["utmos_mean"]
        logger.info(
            f"\n  Step 0 UTMOS: {utmos_0:.3f} ± {eval_result.get('utmos_std', 0):.3f} "
            f"({eval_result['n_evaluated']}/{eval_result['n_total']})"
        )
        logger.info(f"  Baselines: HF base=4.206, Production GGUF=4.391, SFT v3=3.942")

        self.best_utmos = utmos_0
        self.best_step = 0
        self.metrics_history.append({"step": 0, "type": "eval", **eval_result})
        self._save_metrics()

        # Save initial config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump({
                k: v if not isinstance(v, list) else v
                for k, v in vars(cfg).items()
            }, f, indent=2)

        t_start = time.time()

        for step in range(1, cfg.num_steps + 1):
            t_step = time.time()

            # Learning rate warmup
            if step <= cfg.warmup_steps:
                lr_scale = step / cfg.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = cfg.learning_rate * lr_scale

            # Training step
            metrics = self.train_step()
            step_time = time.time() - t_step
            metrics["step_time"] = step_time

            # Log
            if step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                steps_per_min = step / (elapsed / 60)
                eta_min = (cfg.num_steps - step) / steps_per_min if steps_per_min > 0 else 0

                logger.info(
                    f"  Step {step:4d}/{cfg.num_steps} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"UTMOS={metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f} "
                    f"[{metrics['reward_min']:.2f}-{metrics['reward_max']:.2f}] | "
                    f"tokens={metrics['audio_tokens_mean']:.0f} | "
                    f"grad={metrics['grad_norm']:.3f} | "
                    f"degen={metrics['n_degenerate']}/{metrics['n_completions']} | "
                    f"{step_time:.1f}s | "
                    f"ETA {eta_min:.0f}min"
                )
                self.metrics_history.append({"type": "train", **metrics})

            # Evaluate
            if step % cfg.eval_every == 0:
                logger.info(f"\n[Eval @ step {step}]")
                eval_result = self.evaluate()
                utmos = eval_result["utmos_mean"]

                logger.info(
                    f"\n  Step {step} UTMOS: {utmos:.3f} ± {eval_result.get('utmos_std', 0):.3f} "
                    f"({eval_result['n_evaluated']}/{eval_result['n_total']})"
                )

                if utmos > self.best_utmos + 0.01:
                    self.best_utmos = utmos
                    self.best_step = step
                    logger.info(f"  *** NEW BEST UTMOS: {utmos:.3f} at step {step} ***")
                    self.save_checkpoint("best")
                else:
                    logger.info(f"  Best remains: {self.best_utmos:.3f} @ step {self.best_step}")

                delta_from_base = utmos - 4.206
                delta_from_prod = utmos - 4.391
                logger.info(f"  vs HF base (4.206): {delta_from_base:+.3f}")
                logger.info(f"  vs Production (4.391): {delta_from_prod:+.3f}")

                self.metrics_history.append({
                    "step": step,
                    "type": "eval",
                    **eval_result,
                })
                self._save_metrics()

            # Save checkpoint
            if step % cfg.save_every == 0:
                self.save_checkpoint()
                gc.collect()
                torch.cuda.empty_cache()

        # Final save
        elapsed = time.time() - t_start
        self.save_checkpoint("final")
        self._save_metrics()

        logger.info("\n" + "=" * 80)
        logger.info("  GRPO TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
        logger.info(f"  Steps: {cfg.num_steps}")
        logger.info(f"  Best UTMOS: {self.best_utmos:.3f} at step {self.best_step}")
        logger.info(f"  Baselines:")
        logger.info(f"    Fresh LoRA (step 0):    {utmos_0:.3f}")
        logger.info(f"    HF base (maya):         4.206")
        logger.info(f"    Production (GGUF):      4.391")
        logger.info(f"    SFT v3 (step 50):       3.942")
        delta = self.best_utmos - 4.206
        logger.info(f"  Improvement over HF base: {delta:+.3f}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 80)

        if self.best_utmos > 4.206:
            logger.info("\n  SUCCESS: GRPO improved quality above HF baseline!")
            if self.best_utmos > 4.391:
                logger.info("  *** EXCEEDED PRODUCTION BASELINE (4.391)! ***")
            else:
                logger.info(f"  Next: merge best checkpoint + GGUF to test production quality")
                logger.info(f"  python 22_merge_and_convert.py --checkpoint "
                             f"{self.output_dir}/checkpoint-best")
        else:
            logger.info("\n  GRPO did not improve over baseline.")
            logger.info("  Consider: adjust gen_temp, lr, group_size, or add more training steps")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training for Orpheus TTS Quality Optimization"
    )
    parser.add_argument("--gpu", type=int, default=3,
                        help="GPU index")
    parser.add_argument("--steps", type=int, default=200,
                        help="Total optimizer steps")
    parser.add_argument("--group-size", type=int, default=8,
                        help="G: completions per prompt")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Prompts per optimizer step")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--gen-temp", type=float, default=0.9,
                        help="Generation temperature (higher for exploration)")
    parser.add_argument("--gen-top-k", type=int, default=50,
                        help="Generation top-k")
    parser.add_argument("--lora-r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max audio tokens per generation")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory override")
    parser.add_argument("--voice", default="maya",
                        help="Voice name for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    config = GRPOConfig(
        num_steps=args.steps,
        group_size=args.group_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        gen_temperature=args.gen_temp,
        gen_top_k=args.gen_top_k,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_new_tokens=args.max_new_tokens,
        voice_name=args.voice,
        seed=args.seed,
    )
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 80)
    logger.info("  Orpheus TTS GRPO Quality Optimization")
    logger.info("  Goal: Push UTMOS above 4.391 (production baseline)")
    logger.info("  Method: REINFORCE with group-relative baseline (GRPO)")
    logger.info("  Reward: UTMOS speech quality score")
    logger.info("=" * 80)

    trainer = OrpheusGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
