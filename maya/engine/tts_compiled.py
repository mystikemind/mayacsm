"""
Compiled Official CSM Generator - Best of both worlds.

Uses the OFFICIAL Generator for correct audio quality,
but applies torch.compile to the model for speed.

This keeps the official generation logic while benefiting from compilation.
"""

import torch
import logging
import time
import sys
import os

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from typing import Optional, List
from generator import Generator, Segment, load_llama3_tokenizer
from models import Model
from moshi.models import loaders
from huggingface_hub import hf_hub_download

from ..config import TTS

logger = logging.getLogger(__name__)


# Enable TF32 for faster matmuls + cuDNN benchmark for Mimi codec convolutions
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class CompiledTTSEngine:
    """
    CSM Generator with torch.compile + TRUE streaming generation.

    Architecture matches Sesame AI's approach:
    - torch.compile on backbone + decoder (max-autotune)
    - True streaming: yields audio chunks DURING generation
    - No watermarking overhead (~20% faster)
    - Voice prompt for consistent identity

    Streaming: First chunk in ~4 frames (320ms audio), subsequent ~12 frames.
    """

    SAMPLE_RATE = 24000
    VOICE_PROMPT_PATH = TTS.voice_prompt_path
    VOICE_PROMPT_TEXT = "oh hey! yeah im doing pretty good"

    # Generation parameters from config
    TEMPERATURE = TTS.temperature
    TOPK = TTS.topk

    # Streaming generation constants (frames = 80ms each)
    # Smaller first chunk = lower latency to first audio
    FIRST_CHUNK_FRAMES = 2    # ~160ms audio - ultra-low latency first chunk
    CHUNK_FRAMES = 8           # ~640ms audio - balance latency vs decode overhead

    def __init__(self):
        self._generator: Optional[Generator] = None
        self._initialized = False
        self._initializing = False  # Prevent recursive init during warmup
        self._context: List[Segment] = []
        self._voice_prompt: Optional[Segment] = None

    def _preprocess_for_speech(self, text: str) -> str:
        """
        Preprocess text for CSM speech output.

        Keep prosody-relevant punctuation (periods, commas, question marks).
        The fine-tuned model was trained with these in the text.
        """
        import re

        text = text.lower()
        # Keep punctuation that affects prosody: . , ? '
        text = re.sub(r"[^\w\s.,?']", " ", text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _build_context(self, use_context: bool) -> List[Segment]:
        """Build context segments for generation."""
        context = []
        if self._voice_prompt:
            context.append(self._voice_prompt)
        if use_context and self._context:
            context.extend(self._context[-2:])
        return context

    def _estimate_max_audio_ms(self, text: str) -> int:
        """Estimate max audio duration from word count."""
        word_count = len(text.split())
        estimated_ms = int((word_count / 3.0 + 0.3) * 1000)
        return min(max(estimated_ms, 1500), 2500)  # 1.5s min, 2.5s max

    @torch.inference_mode()
    def _run_frame_loop(self, text: str, context: List[Segment], max_audio_ms: int):
        """Run the CSM token generation loop, yielding individual frame samples.

        Each frame = 32 codebook tokens = 80ms of audio.
        Bypasses Generator.generate() to skip watermarking (~20% speedup).

        Yields:
            Frame tensors (1, 32) - one per 80ms of audio
        """
        gen = self._generator
        model = gen._model
        device = gen.device

        model.reset_caches()
        max_generation_len = int(max_audio_ms / 80)

        # Tokenize context and text (same as Generator.generate)
        tokens, tokens_mask = [], []
        for segment in context:
            seg_tokens, seg_mask = gen._tokenize_segment(segment)
            tokens.append(seg_tokens)
            tokens_mask.append(seg_mask)

        gen_tokens, gen_mask = gen._tokenize_text_segment(text, 0)
        tokens.append(gen_tokens)
        tokens_mask.append(gen_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(device)

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(device)

        # Validate context length
        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            logger.warning(f"Context too long ({curr_tokens.size(1)} >= {max_context_len})")
            return

        for _ in range(max_generation_len):
            sample = model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos, self.TEMPERATURE, self.TOPK
            )

            if torch.all(sample == 0):
                break

            yield sample

            curr_tokens = torch.cat(
                [sample, torch.zeros(1, 1).long().to(device)], dim=1
            ).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

    def _decode_frames(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """Decode a list of frame tensors to audio using Mimi codec."""
        if not frames:
            return torch.tensor([], device=self._generator.device)
        stacked = torch.stack(frames).permute(1, 2, 0)
        audio = self._generator._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)
        return audio

    def initialize(self) -> None:
        """Load and compile the official CSM generator."""
        if self._initialized or self._initializing:
            return

        self._initializing = True  # Prevent recursive init during warmup

        logger.info("=" * 60)
        logger.info("LOADING COMPILED OFFICIAL CSM GENERATOR")
        logger.info("Official quality + torch.compile speed + Voice Prompt")
        logger.info("=" * 60)

        start_total = time.time()

        # Load model FIRST, then compile, then create Generator
        logger.info("Loading CSM-1B model...")
        start = time.time()
        model = Model.from_pretrained("sesame/csm-1b")

        # Load fine-tuned weights if available (LoRA v3 merged checkpoint)
        if os.path.exists(TTS.custom_model_path):
            logger.info(f"Loading fine-tuned weights from {TTS.custom_model_path}...")
            sd = torch.load(TTS.custom_model_path, map_location="cuda")
            valid_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in sd.items() if k in valid_keys}
            model.load_state_dict(filtered, strict=True)
            logger.info(f"  Fine-tuned model loaded ({len(filtered)} keys)")
            del sd, filtered
        else:
            logger.warning(f"  Custom model not found at {TTS.custom_model_path}, using base CSM-1B")

        model.to(device="cuda", dtype=torch.bfloat16)
        model.eval()
        logger.info(f"  Model loaded in {time.time()-start:.1f}s")

        # Apply torch.compile to both backbone and decoder for maximum speed
        logger.info("Compiling backbone (max-autotune mode)...")
        start = time.time()
        model.backbone = torch.compile(
            model.backbone,
            mode='max-autotune',
            fullgraph=False,
        )
        logger.info(f"  Backbone compiled in {time.time()-start:.1f}s")

        logger.info("Compiling decoder (max-autotune mode)...")
        start = time.time()
        model.decoder = torch.compile(
            model.decoder,
            mode='max-autotune',
            fullgraph=False,
        )
        logger.info(f"  Decoder compiled in {time.time()-start:.1f}s")

        # Create Generator with compiled model
        logger.info("Creating Generator...")
        self._generator = Generator(model)

        # Load voice prompt for consistent voice identity
        logger.info("Loading voice prompt...")
        try:
            if os.path.exists(self.VOICE_PROMPT_PATH):
                voice_data = torch.load(self.VOICE_PROMPT_PATH)
                voice_audio = voice_data['audio'].to(self._generator.device)

                # Create permanent voice prompt segment
                # Text MUST match the audio for best quality
                voice_text = voice_data.get('text', self.VOICE_PROMPT_TEXT)
                self._voice_prompt = Segment(
                    text=voice_text,
                    speaker=0,
                    audio=voice_audio
                )
                logger.info(f"  Voice prompt loaded: {len(voice_audio)/self.SAMPLE_RATE:.1f}s")
            else:
                logger.warning(f"  Voice prompt not found at {self.VOICE_PROMPT_PATH}")
                self._voice_prompt = None
        except Exception as e:
            logger.warning(f"  Failed to load voice prompt: {e}")
            self._voice_prompt = None

        # AGGRESSIVE warmup - compile ALL code paths before first real use
        # Must warmup both full generation AND streaming paths
        logger.info("Warming up (ensures fast inference)...")
        warmup_start = time.time()

        warmup_context = [self._voice_prompt] if self._voice_prompt else []

        # Warmup phrases - diverse lengths to compile all kernels
        warmup_phrases = [
            "hi",                           # 1 word (short)
            "how are you",                  # 3 words (medium)
            "i hope you are well today",    # 6 words (typical response)
            "thats really interesting tell me more",  # 6 words (another pattern)
        ]

        # Warmup full generation
        logger.info("  Warming up full generation...")
        for phrase in warmup_phrases:
            for i in range(2):
                start = time.time()
                _ = self._generator.generate(
                    text=phrase,
                    speaker=0,
                    context=warmup_context,
                    max_audio_length_ms=3000,
                    temperature=self.TEMPERATURE,
                    topk=self.TOPK
                )
                elapsed = (time.time() - start) * 1000
                if i == 0:
                    logger.info(f"    Full '{phrase[:15]}': {elapsed:.0f}ms")
                if elapsed < 500:
                    break

        # Warmup streaming generation (different code path)
        logger.info("  Warming up streaming generation...")
        for phrase in warmup_phrases[:2]:  # Only first 2 for speed
            for i in range(2):
                start = time.time()
                chunks = list(self.generate_stream(phrase, use_context=True))
                elapsed = (time.time() - start) * 1000
                if i == 0:
                    logger.info(f"    Stream '{phrase[:15]}': {elapsed:.0f}ms ({len(chunks)} chunks)")
                if elapsed < 500:
                    break

        logger.info("  Compilation complete!")

        warmup_time = time.time() - warmup_start
        total_time = time.time() - start_total

        logger.info("=" * 60)
        logger.info(f"COMPILED CSM READY")
        logger.info(f"  Total init time: {total_time:.1f}s")
        logger.info(f"  Warmup time: {warmup_time:.1f}s")
        logger.info(f"  Voice prompt: {'LOADED' if self._voice_prompt else 'NOT FOUND'}")
        logger.info("=" * 60)

        self._initializing = False
        self._initialized = True

    @torch.inference_mode()
    def generate(self, text: str, use_context: bool = True) -> torch.Tensor:
        """
        Generate complete audio (non-streaming, with quality retry).

        Bypasses watermarking for ~20% speedup.
        Uses direct frame loop instead of Generator.generate().

        Args:
            text: Text to synthesize (must not be empty)
            use_context: Whether to use conversation context

        Returns:
            Audio tensor at 24kHz
        """
        if not self._initialized:
            self.initialize()

        if not text or not text.strip():
            logger.warning("Empty text provided to TTS, returning silence")
            return torch.zeros(int(self.SAMPLE_RATE * 0.5), device=self._generator.device)

        text = self._preprocess_for_speech(text.strip())
        start = time.time()

        context = self._build_context(use_context)
        max_audio_ms = self._estimate_max_audio_ms(text)

        # Quality-based retry (max 2 attempts)
        min_rms = 0.015
        max_attempts = 2
        audio = None

        for attempt in range(max_attempts):
            # Collect all frames (bypasses watermark)
            frames = list(self._run_frame_loop(text, context, max_audio_ms))

            if not frames:
                logger.warning(f"TTS attempt {attempt+1}: no frames generated")
                continue

            candidate = self._decode_frames(frames)

            rms = torch.sqrt(torch.mean(candidate ** 2))
            if rms >= min_rms:
                audio = candidate
                break
            elif attempt < max_attempts - 1:
                logger.warning(f"TTS attempt {attempt+1}: RMS={rms:.4f} too low, retrying")
            else:
                logger.warning(f"TTS attempt {attempt+1}: RMS={rms:.4f} still low, using anyway")
                audio = candidate

        if audio is None:
            logger.warning("TTS failed all attempts, returning silence")
            return torch.zeros(int(self.SAMPLE_RATE * 0.5), device=self._generator.device)

        # Minimal processing (normalization done in orchestrator)
        audio = audio.clone()
        audio = audio - audio.mean()
        peak = audio.abs().max()
        if peak > 1.0:
            audio = audio / peak

        # Gentle fade (10ms) to prevent clicks
        fade_samples = int(self.SAMPLE_RATE * 0.010)
        if len(audio) > fade_samples * 2:
            fade_in = torch.linspace(0, 1, fade_samples, device=audio.device)
            fade_out = torch.linspace(1, 0, fade_samples, device=audio.device)
            audio[:fade_samples] = audio[:fade_samples] * fade_in
            audio[-fade_samples:] = audio[-fade_samples:] * fade_out

        elapsed = time.time() - start
        duration = len(audio) / self.SAMPLE_RATE
        rtf = elapsed / duration if duration > 0 else 0
        logger.info(f"TTS: {duration:.1f}s audio in {elapsed*1000:.0f}ms (RTF: {rtf:.2f}x)")

        return audio

    @torch.inference_mode()
    def generate_stream(self, text: str, use_context: bool = True):
        """
        TRUE streaming TTS - yields audio chunks DURING generation.

        Instead of generating all audio then sending, this yields chunks
        as frames are generated:
        - First chunk: 4 frames = ~320ms audio (arrives fast)
        - Subsequent chunks: 12 frames = ~960ms audio

        No watermarking, no retry (streaming can't restart).
        Per-chunk DC removal and clipping prevention.

        Args:
            text: Text to synthesize
            use_context: Whether to use conversation context

        Yields:
            Audio chunks as torch tensors at 24kHz
        """
        if not self._initialized and not self._initializing:
            self.initialize()

        if not text or not text.strip():
            yield torch.zeros(int(self.SAMPLE_RATE * 0.1), device=self._generator.device)
            return

        text = self._preprocess_for_speech(text.strip())

        context = self._build_context(use_context)
        max_audio_ms = self._estimate_max_audio_ms(text)

        frame_buffer = []
        is_first_chunk = True
        start = time.time()

        for sample in self._run_frame_loop(text, context, max_audio_ms):
            frame_buffer.append(sample.clone())

            chunk_size = self.FIRST_CHUNK_FRAMES if is_first_chunk else self.CHUNK_FRAMES

            if len(frame_buffer) >= chunk_size:
                audio_chunk = self._decode_frames(frame_buffer)

                # Per-chunk processing
                audio_chunk = audio_chunk - audio_chunk.mean()
                peak = audio_chunk.abs().max()
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                if is_first_chunk:
                    elapsed = (time.time() - start) * 1000
                    chunk_dur = len(audio_chunk) / self.SAMPLE_RATE
                    logger.info(f"TTS first chunk: {chunk_dur:.2f}s audio in {elapsed:.0f}ms")

                yield audio_chunk
                frame_buffer = []
                is_first_chunk = False

        # Yield remaining frames
        if frame_buffer:
            audio_chunk = self._decode_frames(frame_buffer)
            audio_chunk = audio_chunk - audio_chunk.mean()
            peak = audio_chunk.abs().max()
            if peak > 1.0:
                audio_chunk = audio_chunk / peak
            yield audio_chunk

        total = (time.time() - start) * 1000
        logger.info(f"TTS streaming complete: {total:.0f}ms total")

    def add_context(self, text: str, audio: torch.Tensor, is_user: bool = True) -> None:
        """Add a turn to conversation context."""
        if not self._initialized:
            self.initialize()

        if audio.dim() > 1:
            audio = audio.squeeze()

        speaker_id = 1 if is_user else 0

        segment = Segment(
            text=text,
            speaker=speaker_id,
            audio=audio.cpu()  # Keep on CPU for memory
        )

        self._context.append(segment)

        # Keep only last 4 turns for quality (increased from 2)
        if len(self._context) > 4:
            self._context = self._context[-4:]

        logger.debug(f"Added context: speaker={speaker_id}, text='{text[:30]}...', context_size={len(self._context)}")

    def clear_context(self) -> None:
        """Clear conversation context."""
        self._context.clear()

    def get_context_size(self) -> int:
        """Get current context size."""
        return len(self._context)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE
