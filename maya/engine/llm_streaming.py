"""
Streaming LLM Engine - Generate tokens and yield partial sentences.

The key to Sesame-level latency is NOT faster LLM inference.
It's STREAMING: start TTS before LLM finishes.

Architecture:
    User input → LLM starts generating → First phrase ready (~50-100ms)
                                       → TTS starts on partial text
                                       → First audio at ~150-200ms
                                       → LLM continues in parallel

This reduces perceived latency from:
    LLM (400ms) + TTS first (132ms) = 532ms

To:
    LLM partial (80ms) + TTS first (132ms) = ~212ms

That's the Sesame secret.
"""

import torch
import logging
import time
import re
from typing import Generator, Optional, List, Dict
from threading import Thread
from queue import Queue

logger = logging.getLogger(__name__)


class StreamingLLMEngine:
    """
    Streaming LLM that yields partial sentences for immediate TTS.

    Key features:
    - Yields text as soon as a complete phrase/sentence is ready
    - Uses sentence boundary detection (., !, ?, or natural pause)
    - Allows TTS to start generating while LLM continues
    - Dramatically reduces time to first audio

    Target: First phrase in ~80ms (vs 400ms for full response)
    """

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self._model_id = model_id
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._messages: List[Dict[str, str]] = []
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Streaming config
        self._min_phrase_tokens = 4  # Minimum tokens before yielding
        self._phrase_boundaries = {'.', '!', '?', ',', '...'}  # Natural pause points

    def initialize(self) -> None:
        """Load model optimized for streaming generation."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("LOADING STREAMING LLM")
        logger.info("Optimized for partial sentence generation")
        logger.info("=" * 60)

        start = time.time()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        logger.info(f"Loading {self._model_id}...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self._model.eval()

        # Compile for speed
        logger.info("Compiling model...")
        self._model = torch.compile(self._model, mode='reduce-overhead', fullgraph=False)

        # Initialize conversation
        self._messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]

        # Warmup
        logger.info("Warming up streaming generation...")
        for _ in self.generate_stream("hi"):
            pass

        total_time = time.time() - start
        logger.info(f"Streaming LLM ready in {total_time:.1f}s")
        self._initialized = True

    def _get_system_prompt(self) -> str:
        """System prompt for Maya."""
        return """you are maya, warm and expressive, on a voice call.

start EVERY response with one emotion tag in brackets:
- [happy] for positive, excited
- [sad] for sympathetic
- [confused] for uncertain
- [whisper] for gentle, intimate
- no tag for neutral

keep responses SHORT - 6-12 words max.

examples:
- "[happy] oh wow thats amazing!"
- "[sad] oh no im so sorry"
- "[confused] wait what do you mean?"
- "yeah that makes sense"
"""

    def generate_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Generate response and yield partial sentences as they become ready.

        This is the key to Sesame-level latency - we don't wait for the
        full response. As soon as we have a complete phrase, we yield it
        so TTS can start generating audio.

        Yields:
            Partial sentences/phrases as they become ready
        """
        if not self._initialized:
            self.initialize()

        # Add user message
        self._messages.append({"role": "user", "content": user_input})

        # Prepare input
        prompt = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self._device)

        # Generate with streaming
        generated_text = ""
        pending_text = ""
        token_count = 0
        first_yield_time = None
        start_time = time.time()

        with torch.no_grad():
            # Use generate with streamer
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Start generation in background thread
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 35,
                "temperature": 0.9,
                "top_p": 0.92,
                "do_sample": True,
                "streamer": streamer,
                "pad_token_id": self._tokenizer.eos_token_id,
            }

            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()

            # Process tokens as they arrive
            for new_text in streamer:
                token_count += 1
                pending_text += new_text
                generated_text += new_text

                # Check if we have a complete phrase to yield
                phrase = self._extract_complete_phrase(pending_text)

                if phrase:
                    if first_yield_time is None:
                        first_yield_time = time.time() - start_time
                        logger.info(f"First phrase in {first_yield_time*1000:.0f}ms: '{phrase}'")

                    yield phrase
                    pending_text = pending_text[len(phrase):].lstrip()

            # Yield any remaining text
            if pending_text.strip():
                yield pending_text.strip()

            thread.join()

        # Add assistant response to history
        self._messages.append({"role": "assistant", "content": generated_text.strip()})

        total_time = time.time() - start_time
        logger.info(f"Full response in {total_time*1000:.0f}ms ({token_count} tokens)")

    def _extract_complete_phrase(self, text: str) -> Optional[str]:
        """
        Extract a complete phrase if one is ready.

        A phrase is complete if:
        1. It ends with a sentence boundary (. ! ? ,)
        2. OR it has enough tokens and ends with a word boundary

        Returns:
            Complete phrase or None
        """
        text = text.strip()
        if not text:
            return None

        # Check for emotion tag at start - yield immediately
        if text.startswith('['):
            bracket_end = text.find(']')
            if bracket_end > 0:
                # Include tag plus first few words if available
                tag = text[:bracket_end + 1]
                rest = text[bracket_end + 1:].strip()

                # Find first phrase boundary after tag
                for i, char in enumerate(rest):
                    if char in self._phrase_boundaries:
                        return tag + " " + rest[:i + 1]

                # If we have tag + some words but no boundary, yield after ~4 words
                words = rest.split()
                if len(words) >= 3:
                    return tag + " " + " ".join(words[:4])

        # Check for sentence boundaries
        for boundary in ['. ', '! ', '? ', ', ']:
            idx = text.find(boundary)
            if idx > 0:
                return text[:idx + 1]

        # For longer text without boundaries, yield at word boundary
        words = text.split()
        if len(words) >= 5:
            return " ".join(words[:4])

        return None

    def generate(self, user_input: str) -> str:
        """Non-streaming generate for compatibility."""
        parts = list(self.generate_stream(user_input))
        return " ".join(parts)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages = [self._messages[0]]  # Keep system prompt
