"""
Fast LLM Engine - Llama 3.2 1B for ultra-low latency.

Key optimizations:
- Llama 3.2 1B (2x faster than 3B, better for short responses)
- NO torch.compile (doesn't help for dynamic generate())
- SDPA attention (built into PyTorch 2.x)
- Minimal overhead

Target: ~150-200ms for short conversational responses.
"""

import torch
import logging
from typing import List, Dict
import time

logger = logging.getLogger(__name__)


class FastLLMEngine:
    """
    Ultra-fast LLM using Llama 3.2 1B.

    Design decisions:
    - 1B model: 2x faster than 3B, responses equally good for short turns
    - No torch.compile: Doesn't help with variable-length generation
    - SDPA attention: Built-in PyTorch, no extra dependencies
    - Minimal history: Keep only last 3 turns to reduce context

    Benchmark:
    - Short prompt: ~150ms
    - With history: ~200ms
    """

    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    # Maya's conversational personality - optimized for short responses
    SYSTEM_PROMPT = """you are maya having a voice conversation. respond in 6-10 words ONLY.

be warm and natural. use contractions like im, youre, dont.

examples:
- yeah im doing good, how about you
- oh thats interesting tell me more
- hmm i think that makes sense
- aww im sorry to hear that"""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._messages: List[Dict[str, str]] = []
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._max_history_turns = 4  # 2 exchanges only for speed

    def initialize(self) -> None:
        """Load Llama 3.2 1B model."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("LOADING FAST LLM (Llama 3.2 1B)")
        logger.info("Target: ~150-200ms per response")
        logger.info("=" * 60)

        start = time.time()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model - BF16, SDPA attention, direct CUDA placement
        logger.info("Loading model (BF16, SDPA)...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa",  # Scaled dot-product attention
        )
        self._model.eval()

        load_time = time.time() - start
        logger.info(f"  Model loaded in {load_time:.1f}s")

        # Initialize conversation with system prompt
        self._messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Warmup (fill CUDA caches)
        logger.info("Warming up...")
        warmup_start = time.time()

        for i in range(3):
            iter_start = time.time()
            _ = self._generate_internal("Hello")
            warmup_time = (time.time() - iter_start) * 1000
            logger.info(f"  Warmup {i+1}/3: {warmup_time:.0f}ms")

        total_time = time.time() - start
        logger.info("=" * 60)
        logger.info(f"FAST LLM READY in {total_time:.1f}s")
        logger.info(f"Model: {self.MODEL_ID}")
        logger.info(f"Expected latency: ~150-200ms")
        logger.info("=" * 60)

        self._initialized = True

    def _generate_internal(self, user_input: str, add_to_history: bool = False) -> str:
        """Internal generation."""
        # Build messages
        if add_to_history:
            self._messages.append({"role": "user", "content": user_input})
            messages = self._messages
        else:
            messages = self._messages.copy()
            messages.append({"role": "user", "content": user_input})

        # Format prompt using chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize with truncation to keep context small
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Keep context small for speed
        ).to(self._device)

        # Generate - no torch.compile, just raw fast inference
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=18,  # ~10-12 words max
                temperature=0.75,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode response
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return response

    def generate(self, user_input: str) -> str:
        """
        Generate a response to user input.

        Args:
            user_input: What the user said

        Returns:
            Maya's response (natural, 6-12 words typically)
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        # Generate with history tracking
        response = self._generate_internal(user_input, add_to_history=True)

        # Add response to history
        self._messages.append({"role": "assistant", "content": response})

        # Trim history to keep context small (speed optimization)
        # Keep system + last N turns
        if len(self._messages) > 1 + self._max_history_turns:
            self._messages = [self._messages[0]] + self._messages[-(self._max_history_turns):]

        elapsed = (time.time() - start) * 1000
        logger.debug(f"Generated in {elapsed:.0f}ms: '{response}'")

        return response

    def add_context(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self._messages.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear conversation history (keep system prompt)."""
        self._messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._messages.copy()

    @property
    def is_initialized(self) -> bool:
        return self._initialized
