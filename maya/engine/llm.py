"""
Language Model Engine - Llama 3.2 3B Instruct

Generates Maya's responses with real understanding and personality.

Features:
- Streaming token generation
- Conversation context
- Maya personality via system prompt
"""

import torch
import logging
from typing import Optional, List, Generator, Dict
import time
from threading import Thread
from queue import Queue

from ..config import LLM

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Llama 3.2 3B wrapper for response generation.

    Features:
    - Streaming output for low latency
    - Full conversation context
    - Maya's personality built-in
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._streamer = None

        # Conversation history
        self._messages: List[Dict[str, str]] = []

        # Performance tracking
        self._total_generations = 0
        self._total_tokens = 0
        self._total_time = 0.0

    def initialize(self) -> None:
        """Load Llama model and tokenizer."""
        if self._initialized:
            return

        logger.info(f"Loading {LLM.model_id}...")
        start = time.time()

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TextIteratorStreamer,
                BitsAndBytesConfig
            )

            # Quantization config for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                LLM.model_id,
                trust_remote_code=True
            )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model with 4-bit quantization
            self._model = AutoModelForCausalLM.from_pretrained(
                LLM.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            # Initialize conversation with system prompt
            self._messages = [
                {"role": "system", "content": LLM.system_prompt}
            ]

            elapsed = time.time() - start
            logger.info(f"Llama loaded in {elapsed:.1f}s")
            self._initialized = True

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Run: pip install transformers accelerate bitsandbytes")
            raise

    def generate(self, user_input: str) -> str:
        """
        Generate a response (non-streaming).

        Args:
            user_input: User's message

        Returns:
            Maya's response
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        # Add user message
        self._messages.append({"role": "user", "content": user_input})

        # Format for model
        prompt = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self._model.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=LLM.max_new_tokens,
                temperature=LLM.temperature,
                top_p=LLM.top_p,
                do_sample=LLM.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = response.strip()

        # Add to history
        self._messages.append({"role": "assistant", "content": response})

        # Trim history if too long (keep system + last 10 turns)
        if len(self._messages) > 21:
            self._messages = [self._messages[0]] + self._messages[-20:]

        # Track performance
        elapsed = time.time() - start
        self._total_generations += 1
        self._total_tokens += len(new_tokens)
        self._total_time += elapsed

        logger.debug(f"Generated in {elapsed*1000:.0f}ms: '{response[:50]}...'")

        return response

    def generate_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Generate a response with streaming.

        Args:
            user_input: User's message

        Yields:
            Response tokens as they're generated
        """
        if not self._initialized:
            self.initialize()

        from transformers import TextIteratorStreamer

        start = time.time()

        # Add user message
        self._messages.append({"role": "user", "content": user_input})

        # Format for model
        prompt = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self._model.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        # Generate in background thread
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=LLM.max_new_tokens,
            temperature=LLM.temperature,
            top_p=LLM.top_p,
            do_sample=LLM.do_sample,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they come
        full_response = ""
        for text in streamer:
            full_response += text
            yield text

        thread.join()

        # Add to history
        self._messages.append({"role": "assistant", "content": full_response.strip()})

        # Trim history
        if len(self._messages) > 21:
            self._messages = [self._messages[0]] + self._messages[-20:]

        # Track performance
        elapsed = time.time() - start
        self._total_generations += 1
        self._total_time += elapsed

        logger.debug(f"Streamed in {elapsed*1000:.0f}ms: '{full_response[:50]}...'")

    def generate_starter(self, user_input: str) -> str:
        """
        Generate a SHORT starter response (2-4 words).

        This is the FAST path - generates minimal tokens for quick TTS.
        Examples: "Oh interesting!", "Hmm, well...", "Yeah, so..."

        Args:
            user_input: User's message

        Returns:
            Short starter (2-4 words)
        """
        if not self._initialized:
            self.initialize()

        # Add user message temporarily
        messages = self._messages + [{"role": "user", "content": user_input}]

        # Use a special prompt for very short response
        starter_system = """Generate ONLY a 2-4 word natural conversation starter. Examples:
- "Oh, that's interesting..."
- "Hmm, let me think..."
- "Yeah, so basically..."
- "Oh wow, really?"
- "Well, you know..."
Just the starter, nothing more."""

        messages_for_starter = [
            {"role": "system", "content": starter_system},
            {"role": "user", "content": f"Generate a natural 2-4 word starter for responding to: '{user_input}'"}
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages_for_starter,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=10,  # Very short
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        starter = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean up - keep only first sentence fragment
        if '.' in starter:
            starter = starter.split('.')[0] + '...'
        if len(starter.split()) > 6:
            starter = ' '.join(starter.split()[:4]) + '...'

        logger.debug(f"Generated starter: '{starter}'")
        return starter

    def generate_continuation(self, user_input: str, starter: str) -> str:
        """
        Generate the continuation after the starter.

        Args:
            user_input: Original user message
            starter: The starter that was already generated

        Returns:
            The rest of the response
        """
        if not self._initialized:
            self.initialize()

        # Add user message to history
        self._messages.append({"role": "user", "content": user_input})

        # Create prompt that continues from starter
        continuation_prompt = f"Continue this response naturally (the conversation started with '{starter}'): "

        prompt = self._tokenizer.apply_chat_template(
            self._messages + [{"role": "assistant", "content": starter}],
            tokenize=False,
            add_generation_prompt=False
        )

        # Add continuation prompt
        prompt = prompt.rstrip() + " "

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=LLM.max_new_tokens,
                temperature=LLM.temperature,
                top_p=LLM.top_p,
                do_sample=LLM.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        continuation = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # The full response for history is starter + continuation
        full_response = starter + " " + continuation
        self._messages.append({"role": "assistant", "content": full_response})

        # Trim history if too long
        if len(self._messages) > 21:
            self._messages = [self._messages[0]] + self._messages[-20:]

        logger.debug(f"Generated continuation: '{continuation[:50]}...'")
        return continuation

    def add_context(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self._messages.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear conversation history (keep system prompt)."""
        self._messages = [self._messages[0]] if self._messages else []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._messages.copy()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "total_generations": self._total_generations,
            "total_tokens": self._total_tokens,
            "total_time": self._total_time,
            "history_length": len(self._messages),
        }
