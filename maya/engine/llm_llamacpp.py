"""
LLM Engine via llama.cpp - High-performance Llama 3.2 3B Instruct

Uses llama-server for optimized inference via GGUF quantization.
Benchmark: 109ms avg latency, 155 tok/s (Q4_K_M on A10G) - 6.7x faster than HF Transformers.

Architecture:
- llama-server runs Llama 3.2 3B Instruct GGUF on dedicated GPU
- OpenAI-compatible /v1/chat/completions API
- Automatic chat template detection from GGUF metadata
- Connection pooling via requests.Session

Drop-in replacement for OptimizedLLMEngine with same interface.
"""

import torch
import logging
import time
import os
import json
import signal
import subprocess
from typing import Optional, List, Dict, Generator
from pathlib import Path

from ..config import LLM, DEVICES

logger = logging.getLogger(__name__)

DEFAULT_GGUF_DIR = "/home/ec2-user/SageMaker/.cache/huggingface/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF"
DEFAULT_LLAMA_SERVER = "/home/ec2-user/SageMaker/llama.cpp/build/bin/llama-server"
LLAMA_LIB_PATH = "/home/ec2-user/SageMaker/llama.cpp/build/ggml/src:/home/ec2-user/SageMaker/llama.cpp/build/ggml/src/ggml-cuda"


def _find_gguf(quant: str = "Q4_K_M") -> Optional[str]:
    """Find a GGUF model file in the HF cache."""
    for snapshot_dir in Path(DEFAULT_GGUF_DIR).glob("snapshots/*"):
        for gguf in snapshot_dir.glob(f"*{quant}*"):
            resolved = gguf.resolve()
            if resolved.exists():
                return str(gguf)
    return None


class LlamaCppLLMEngine:
    """
    Llama 3.2 3B Instruct via llama.cpp for ultra-low-latency inference.

    Benchmark on A10G:
      HF Transformers BF16: ~732ms per response
      llama.cpp Q4_K_M:     ~109ms per response (6.7x faster)

    Features:
    - Manages llama-server lifecycle (start/stop)
    - OpenAI-compatible chat API with proper system prompts
    - Conversation history management
    - Connection pooling for minimal HTTP overhead
    """

    def __init__(
        self,
        gguf_quant: Optional[str] = None,
        server_port: Optional[int] = None,
    ):
        self._gguf_quant = gguf_quant or LLM.gguf_quant
        self._server_port = server_port or LLM.server_port
        self._server_url = f"http://127.0.0.1:{self._server_port}"
        self._server_process = None
        self._initialized = False
        self._messages: List[Dict[str, str]] = []
        self._session = None

    def initialize(self) -> None:
        """Start llama-server and set up session."""
        if self._initialized:
            return

        start = time.time()

        logger.info("=" * 60)
        logger.info("LOADING LLM (llama.cpp, ultra-fast)")
        logger.info(f"  Model: Llama 3.2 3B Instruct ({self._gguf_quant})")
        logger.info(f"  GPU: {DEVICES.llm_gpu}, Port: {self._server_port}")
        logger.info("=" * 60)

        self._start_server()

        import requests
        self._session = requests.Session()

        # Initialize conversation with system prompt
        self._messages = [
            {"role": "system", "content": LLM.system_prompt}
        ]

        # Warmup (triggers CUDA kernel compilation in llama.cpp)
        logger.info("Warming up LLM...")
        warmup_texts = [
            "Hello, how are you?",
            "Tell me about your day.",
            "What do you think about that?",
        ]
        for i, text in enumerate(warmup_texts):
            t0 = time.time()
            _ = self._generate_internal(text)
            ms = (time.time() - t0) * 1000
            logger.info(f"  Warmup {i+1}/{len(warmup_texts)}: {ms:.0f}ms")

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"LLM READY in {elapsed:.1f}s (llama.cpp, ~109ms/response)")
        logger.info("=" * 60)

        self._initialized = True

    def _start_server(self) -> None:
        """Start llama-server for LLM."""
        gguf_path = _find_gguf(self._gguf_quant)
        if not gguf_path:
            raise FileNotFoundError(
                f"No GGUF model found for {self._gguf_quant}. "
                f"Download with: python -c \"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('bartowski/Llama-3.2-3B-Instruct-GGUF', "
                f"'Llama-3.2-3B-Instruct-{self._gguf_quant}.gguf')\""
            )

        server_bin = DEFAULT_LLAMA_SERVER
        if not Path(server_bin).exists():
            raise FileNotFoundError(f"llama-server not found at {server_bin}")

        # Check if already running
        try:
            import requests
            r = requests.get(f"{self._server_url}/health", timeout=2)
            if r.status_code == 200:
                logger.info(f"llama-server already running on port {self._server_port}")
                return
        except Exception:
            pass

        logger.info(f"Starting llama-server for LLM: GPU {DEVICES.llm_gpu}, port {self._server_port}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(DEVICES.llm_gpu)
        env["LD_LIBRARY_PATH"] = LLAMA_LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

        cmd = [
            server_bin,
            "-m", gguf_path,
            "-c", "4096",       # 4K context (longer conversations, minimal VRAM: ~256MB)
            "-ngl", "99",       # All layers on GPU
            "--host", "127.0.0.1",
            "--port", str(self._server_port),
            "-fa", "on",        # Flash attention
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "-np", "1",         # Single slot for lowest latency
            "--mlock",          # Lock model in memory for stable inference
            "--no-mmap",        # Keep model in memory, prevent OS swapping
            "-t", "4",          # Thread count for CPU operations
        ]

        self._server_process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # Wait for server
        import requests
        for attempt in range(30):
            time.sleep(1)
            try:
                r = requests.get(f"{self._server_url}/health", timeout=2)
                if r.status_code == 200:
                    logger.info(f"llama-server for LLM ready (PID {self._server_process.pid})")
                    return
            except Exception:
                pass

            if self._server_process.poll() is not None:
                stdout = self._server_process.stdout.read().decode() if self._server_process.stdout else ""
                raise RuntimeError(f"llama-server exited: {stdout[-500:]}")

        raise TimeoutError("llama-server for LLM failed to start")

    def _stop_server(self) -> None:
        """Stop llama-server."""
        if self._server_process and self._server_process.poll() is None:
            logger.info("Stopping LLM llama-server...")
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                self._server_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._server_process = None

    def __del__(self):
        self._stop_server()

    @staticmethod
    def _truncate_at_boundary(text: str, max_words: int = 50) -> str:
        """Truncate response at a natural sentence boundary.

        Safety net only - the system prompt controls actual response length.
        SOTA research: Sesame/OpenAI/ElevenLabs use 2-3 sentences (~20-45 words).
        This catches the rare case where the LLM runs long despite prompting.
        """
        words = text.split()
        if len(words) <= max_words:
            return text

        # Find a natural sentence boundary within the word limit
        truncated = " ".join(words[:max_words])

        # Try to cut at last sentence boundary
        for sep in [". ", "? ", "! "]:
            idx = truncated.rfind(sep)
            if idx > len(truncated) // 3:
                return truncated[:idx + 1].strip()

        # Fall back to comma only if it's in the latter half
        idx = truncated.rfind(", ")
        if idx > len(truncated) // 2:
            return truncated[:idx].strip()

        return truncated

    def _generate_internal(self, user_input: str, add_to_history: bool = False) -> str:
        """Generate response via llama-server chat API."""
        if add_to_history:
            self._messages.append({"role": "user", "content": user_input})
            messages = self._messages
        else:
            messages = self._messages + [{"role": "user", "content": user_input}]

        payload = {
            "messages": messages,
            "max_tokens": LLM.max_new_tokens,
            "temperature": LLM.temperature,
            "top_p": LLM.top_p,
            "frequency_penalty": 0.4,   # Prevent falling into response patterns
            "presence_penalty": 0.3,    # Encourage topic diversity
        }

        resp = self._session.post(
            f"{self._server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Truncate at natural boundary to keep responses concise
        content = self._truncate_at_boundary(content)

        return content

    def generate_stream(self, user_input: str) -> Generator[str, None, None]:
        """Stream LLM response sentence-by-sentence via SSE.

        Yields complete sentences as they arrive, enabling TTS to start
        on the first sentence while LLM generates the rest.

        SOTA pattern: ElevenLabs starts TTS after first sentence + comma.
        Hume EVI delivers audio sentence-by-sentence.

        Args:
            user_input: User's message

        Yields:
            Complete sentences as strings
        """
        if not self._initialized:
            self.initialize()

        self._messages.append({"role": "user", "content": user_input})

        payload = {
            "messages": self._messages,
            "max_tokens": LLM.max_new_tokens,
            "temperature": LLM.temperature,
            "top_p": LLM.top_p,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.3,
            "stream": True,
        }

        resp = self._session.post(
            f"{self._server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
            stream=True,
        )
        resp.raise_for_status()

        buffer = ""
        full_response = ""

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]  # Strip "data: " prefix
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if not token:
                    continue
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

            buffer += token
            full_response += token

            # Check for sentence boundary in buffer
            # Priority: sentence-ending punctuation (.?!) > comma (for longer buffers)
            # ElevenLabs pattern: start TTS after first sentence OR after enough words + comma
            split_idx = -1

            for i, ch in enumerate(buffer):
                if ch in ".?!" and (i == len(buffer) - 1 or i + 1 >= len(buffer) or buffer[i + 1] == " "):
                    split_idx = i
                    break

            # Comma-based split: if buffer is long enough (50+ chars) and has a comma
            # This handles casual speech without periods
            if split_idx < 0 and len(buffer) > 50:
                comma_idx = buffer.rfind(", ")
                if comma_idx > 20:  # At least 20 chars before comma
                    split_idx = comma_idx

            if split_idx >= 0:
                sentence = buffer[:split_idx + 1].strip()
                buffer = buffer[split_idx + 1:].lstrip()
                if sentence:
                    yield sentence

        # Yield any remaining text in the buffer
        remaining = buffer.strip()
        if remaining:
            yield remaining

        # Update conversation history with full response
        full_response = self._truncate_at_boundary(full_response.strip())
        self._messages.append({"role": "assistant", "content": full_response})

        # Trim history
        if len(self._messages) > 31:
            self._messages = [self._messages[0]] + self._messages[-30:]

    def generate(self, user_input: str) -> str:
        """Generate a conversational response.

        Args:
            user_input: User's message

        Returns:
            Maya's response (1-3 sentences, prompt-controlled length)
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        response = self._generate_internal(user_input, add_to_history=True)
        self._messages.append({"role": "assistant", "content": response})

        # Trim history (keep system prompt + last 30 messages = ~15 exchanges)
        # With 4096-token context and ~30 words per turn, this fits comfortably
        if len(self._messages) > 31:
            self._messages = [self._messages[0]] + self._messages[-30:]

        elapsed = (time.time() - start) * 1000
        logger.debug(f"LLM generated in {elapsed:.0f}ms: '{response}'")

        return response

    def add_context(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self._messages.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear conversation history (keep system prompt)."""
        if self._messages and self._messages[0].get("role") == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = [{"role": "system", "content": LLM.system_prompt}]
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._messages.copy()

    @property
    def is_initialized(self) -> bool:
        return self._initialized
