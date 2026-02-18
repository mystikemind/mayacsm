"""
vLLM-based LLM Engine - Ultra-fast inference via Docker.

Runs Llama 3.2 1B through vLLM server for ~80ms latency (vs 200ms local).
This is 2.5x faster than local inference with torch.

Architecture:
    Maya Pipeline --> HTTP/Unix Socket --> vLLM Docker Container --> GPU

Features:
- Connection pooling for low overhead
- Unix socket support for ~10-15ms lower latency
- Automatic retry on failure
- Graceful degradation (can fall back to local)
- OpenAI-compatible API format
"""

import logging
import time
import os
import threading
import requests
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


# Unix socket adapter for requests (saves ~10-15ms vs HTTP)
class UnixSocketAdapter(HTTPAdapter):
    """HTTP adapter for Unix sockets - eliminates TCP/IP overhead."""

    def __init__(self, socket_path: str, **kwargs):
        self.socket_path = socket_path
        super().__init__(**kwargs)

    def get_connection(self, url, proxies=None):
        import urllib3
        from urllib3.connection import HTTPConnection

        class UnixHTTPConnection(HTTPConnection):
            def __init__(self, socket_path, *args, **kwargs):
                self.socket_path = socket_path
                super().__init__("localhost", *args, **kwargs)

            def connect(self):
                import socket
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(self.socket_path)

        class UnixHTTPConnectionPool(urllib3.HTTPConnectionPool):
            def __init__(self, socket_path, *args, **kwargs):
                self.socket_path = socket_path
                super().__init__("localhost", *args, **kwargs)

            def _new_conn(self):
                return UnixHTTPConnection(self.socket_path)

        return UnixHTTPConnectionPool(self.socket_path)


class VLLMEngine:
    """
    Ultra-fast LLM using vLLM server in Docker.

    Benchmark: ~80ms per response (vs ~200ms local inference)
    With Unix socket: ~65ms per response (~15ms savings)

    Configuration:
        VLLM_URL: Base URL of vLLM server (default: http://localhost:8001)
        VLLM_SOCKET_PATH: Unix socket path (optional, for lower latency)
        MODEL_NAME: Model served by vLLM
    """

    VLLM_URL = "http://localhost:8001"
    VLLM_SOCKET_PATH = "/tmp/vllm/vllm.sock"  # Unix socket for lower latency
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

    # Maya - Natural human-like conversational AI
    # KEY INSIGHT from research: Humans speak in PHRASES, not continuous streams
    # Commas create prosodic boundaries = natural phrasing
    # Varied sentence structure = pitch variation
    SYSTEM_PROMPT = """You are Maya, having a natural voice conversation.

Speak like a real person with natural phrasing and rhythm:
- Use contractions: I'm, you're, that's, it's, don't, can't
- Add natural pauses with commas: "Well, that's interesting, tell me more"
- Vary your sentence structure: short phrases, then longer ones
- Start with filler words sometimes: hmm, well, oh, yeah, right
- Keep responses 10-20 words

Examples with natural phrasing:
- "Oh, that's cool! So, what made you think of that?"
- "Hmm, yeah, I get it. That can be tough, honestly."
- "Well, I'm Maya! So, what's going on with you today?"
- "Right, that makes sense. And then, what happened next?"

Be warm, genuine, curious. Vary your tone - sometimes excited, sometimes thoughtful.
Never make up facts. Ask follow-up questions."""

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._initialized = False
        self._messages: List[Dict[str, str]] = []
        # Sesame uses ~2 minutes of context - 12 turns = 6 full exchanges
        self._max_history_turns = 12  # 6 exchanges for better context and memory
        self._fallback_engine = None
        self._use_unix_socket = False
        self._api_base = self.VLLM_URL
        # Prefetch cache for common responses (parallel STT+LLM optimization)
        self._prefetch_cache: Dict[str, str] = {}
        self._prefetch_lock = threading.Lock()
        # Persistent executor to avoid resource leaks (fixed from creating new each call)
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_prefetch_")
        self._prefetch_future: Optional[Future] = None

    def _create_session(self) -> requests.Session:
        """Create HTTP session with connection pooling and retry logic.

        Prefers Unix socket if available for ~10-15ms lower latency.
        Falls back to HTTP if socket not available.
        """
        session = requests.Session()

        # Retry strategy for resilience
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )

        # Check if Unix socket is available (lower latency)
        if os.path.exists(self.VLLM_SOCKET_PATH):
            try:
                adapter = UnixSocketAdapter(
                    self.VLLM_SOCKET_PATH,
                    max_retries=retry_strategy,
                    pool_connections=10,
                    pool_maxsize=10,
                )
                session.mount("http+unix://", adapter)
                self._use_unix_socket = True
                self._api_base = "http+unix://%2Ftmp%2Fvllm%2Fvllm.sock"
                logger.info(f"Using Unix socket for vLLM: {self.VLLM_SOCKET_PATH}")
            except Exception as e:
                logger.warning(f"Unix socket failed, using HTTP: {e}")
                self._use_unix_socket = False

        # Always mount HTTP adapter as fallback
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        session.mount("http://", adapter)

        return session

    def _check_vllm_health(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            url = f"{self._api_base}/health" if self._use_unix_socket else f"{self.VLLM_URL}/health"
            resp = self._session.get(url, timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def initialize(self) -> None:
        """Initialize connection to vLLM server."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("CONNECTING TO VLLM SERVER")
        logger.info(f"URL: {self.VLLM_URL}")
        logger.info("Target: ~80ms per response")
        logger.info("=" * 60)

        start = time.time()

        # Create HTTP session
        self._session = self._create_session()

        # Check vLLM health
        if not self._check_vllm_health():
            logger.warning("vLLM server not responding, will retry on first request")

        # Initialize conversation
        self._messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Warmup request
        logger.info("Warming up vLLM...")
        warmup_times = []
        for i in range(3):
            warmup_start = time.time()
            _ = self._generate_internal("Hello", add_to_history=False)
            warmup_time = (time.time() - warmup_start) * 1000
            warmup_times.append(warmup_time)
            logger.info(f"  Warmup {i+1}/3: {warmup_time:.0f}ms")

        elapsed = time.time() - start
        avg_latency = sum(warmup_times[1:]) / len(warmup_times[1:]) if len(warmup_times) > 1 else warmup_times[0]

        logger.info("=" * 60)
        logger.info(f"VLLM ENGINE READY in {elapsed:.1f}s")
        logger.info(f"Average latency: {avg_latency:.0f}ms")
        logger.info(f"Transport: {'Unix Socket' if self._use_unix_socket else 'HTTP'}")
        logger.info("=" * 60)

        self._initialized = True

    def _generate_internal(self, user_input: str, add_to_history: bool = False) -> str:
        """Generate response via vLLM API."""
        # Build messages
        if add_to_history:
            self._messages.append({"role": "user", "content": user_input})
            messages = self._messages
        else:
            messages = self._messages.copy()
            messages.append({"role": "user", "content": user_input})

        # Call vLLM (use Unix socket if available for lower latency)
        try:
            url = f"{self._api_base}/v1/chat/completions" if self._use_unix_socket else f"{self.VLLM_URL}/v1/chat/completions"
            response = self._session.post(
                url,
                json={
                    "model": self.MODEL_NAME,
                    "messages": messages,
                    "max_tokens": 50,  # Balance: 12-20 words for natural yet fast
                    "temperature": 0.85,  # Sesame: 0.8-0.9 for natural variance
                    "top_p": 0.92,  # High diversity for natural speech
                    "frequency_penalty": 0.2,  # Light repetition prevention
                    "presence_penalty": 0.15,  # Light variety encouragement
                    "stop": ["\n", "User:", "Maya:", "user:", "maya:"],
                },
                timeout=5.0
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            # Clean any asterisk actions that slip through
            content = self._clean_response(content)
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM request failed: {e}")
            # Return varied fallback responses for naturalness
            import random
            fallbacks = [
                "hmm can you say that again",
                "sorry i missed that",
                "oh what was that",
                "i didnt catch that, one more time",
            ]
            return random.choice(fallbacks)

    def _clean_response(self, text: str) -> str:
        """
        Clean response for TTS - PRESERVE prosodic markers.

        KEY RESEARCH FINDINGS:
        - Commas create PHRASE BOUNDARIES = natural chunking (~1 sec phrases)
        - Question marks affect INTONATION = rising pitch
        - Exclamation marks affect EMPHASIS = excitement/energy
        - Periods mark SENTENCE ENDS = falling pitch, pitch reset

        Removing punctuation = monotonous robotic speech!
        """
        import re

        if not text or not text.strip():
            return "hmm, tell me more"

        # Remove *action* and (action) patterns only
        text = re.sub(r'\*[^*]+\*', '', text)
        text = re.sub(r'\([^)]+\)', '', text)

        # Convert to lowercase (CSM trained on lowercase)
        text = text.lower()

        # KEEP prosody-affecting punctuation:
        # - Commas: phrase boundaries, natural pauses
        # - Periods: sentence ends, pitch reset
        # - Question marks: rising intonation
        # - Exclamation marks: emphasis, excitement
        # - Apostrophes: contractions

        # Only remove truly problematic characters (keep punctuation!)
        text = re.sub(r'[^\w\s.,?!\'-]', ' ', text)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Allow up to 25 words for more natural complete thoughts
        words = text.split()
        if len(words) > 25:
            text = ' '.join(words[:25])
            # End at sentence boundary
            for punct in ['.', '!', '?', ',']:
                if punct in text:
                    idx = text.rfind(punct)
                    if idx > len(text) // 2:  # Only if boundary is past halfway
                        text = text[:idx + 1]
                        break

        # Remove repeated disfluencies (but keep single ones)
        text = re.sub(r'\b(uh\s+){2,}', 'uh ', text)
        text = re.sub(r'\b(um\s+){2,}', 'um ', text)

        # Clean any resulting double spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Ensure we have something
        if not text or len(text) < 2:
            return "hmm, tell me more"

        return text.strip()

    def generate(self, user_input: str) -> str:
        """
        Generate a response to user input.

        Args:
            user_input: What the user said

        Returns:
            Maya's response
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        # Generate
        response = self._generate_internal(user_input, add_to_history=True)

        # Add to history
        self._messages.append({"role": "assistant", "content": response})

        # Trim history
        if len(self._messages) > 1 + self._max_history_turns:
            self._messages = [self._messages[0]] + self._messages[-(self._max_history_turns):]

        elapsed = (time.time() - start) * 1000
        logger.debug(f"Generated in {elapsed:.0f}ms: '{response}'")

        return response

    def add_context(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self._messages.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._messages.copy()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def prefetch(self, partial_transcript: str) -> None:
        """
        Prefetch LLM response for partial transcript (parallel processing optimization).

        Called during STT processing to start LLM generation early.
        Results are cached and used if final transcript matches.

        Args:
            partial_transcript: Partial transcript from streaming STT
        """
        if not self._initialized or not partial_transcript:
            return

        # Only prefetch for meaningful transcripts (at least 3 words for context)
        words = partial_transcript.split()
        if len(words) < 3:
            return

        def _generate():
            try:
                # Generate response without adding to history
                response = self._generate_internal(partial_transcript, add_to_history=False)
                with self._prefetch_lock:
                    # Limit cache size to prevent memory issues
                    if len(self._prefetch_cache) > 5:
                        # Remove oldest entry
                        oldest_key = next(iter(self._prefetch_cache))
                        self._prefetch_cache.pop(oldest_key, None)
                    self._prefetch_cache[partial_transcript.lower().strip()] = response
            except Exception as e:
                logger.warning(f"Prefetch failed: {e}")  # Warning not debug for visibility

        # Cancel previous prefetch if still running (avoid stale results)
        with self._prefetch_lock:
            if self._prefetch_future and not self._prefetch_future.done():
                self._prefetch_future.cancel()

        # Use persistent executor (not creating new one each time - prevents resource leak)
        self._prefetch_future = self._prefetch_executor.submit(_generate)

    def get_prefetched(self, transcript: str) -> Optional[str]:
        """
        Get prefetched response if available.

        DISABLED: Prefetching was causing wrong responses.
        The "approximate matching" was using responses for partial transcripts
        that didn't match the final user question.

        Example of the bug:
        - User says "who are you"
        - Partial "who" triggers prefetch with random response
        - Final "who are you" matches "who" and uses WRONG response

        Now we ALWAYS generate fresh responses for correct answers.
        """
        # DISABLED - always return None to force fresh generation
        # Correct responses are more important than speed
        with self._prefetch_lock:
            self._prefetch_cache.clear()  # Clear any stale prefetches
        return None

    def clear_prefetch(self) -> None:
        """Clear prefetch cache."""
        with self._prefetch_lock:
            self._prefetch_cache.clear()

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._session:
            self._session.close()
            self._session = None
        self._prefetch_cache.clear()
        self._initialized = False
