"""
Streaming TTS Engine - CSM with Real-Time Audio Chunks

THE KEY TO LIGHTNING FAST RESPONSE:
- Generate audio in chunks as the model produces them
- Start streaming IMMEDIATELY - don't wait for full generation
- Achieve sub-200ms latency to first audio chunk

Based on research from csm-streaming repo achieving 0.28x RTF.
"""

import torch
import torchaudio
import asyncio
import logging
import time
from typing import Optional, List, AsyncIterator, Tuple, Generator
from dataclasses import dataclass
import sys

# Add CSM to path
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from ..config import TTS, AUDIO
import os

logger = logging.getLogger(__name__)


# CRITICAL OPTIMIZATION: Patch KVCache to avoid .item() calls
# This enables potential CUDA graph optimizations and reduces CPU-GPU sync
def _patch_kv_cache():
    """Patch torchtune KVCache.reset to avoid .item() call that blocks CUDA graphs."""
    try:
        import torchtune.modules.kv_cache as kv_module

        def patched_reset(self):
            """Reset without calling .size (avoids .item() CPU-GPU sync)."""
            self.k_cache.zero_()
            self.v_cache.zero_()
            # Reset cache_pos to initial state without using .size
            max_seq_len = self.cache_pos.size(0)
            self.cache_pos.copy_(torch.arange(0, max_seq_len, device=self.cache_pos.device))

        kv_module.KVCache.reset = patched_reset
        logger.info("KVCache patched to avoid .item() calls")
    except Exception as e:
        logger.warning(f"Failed to patch KVCache: {e}")

_patch_kv_cache()


@dataclass
class StreamingConfig:
    """Streaming generation configuration.

    OPTIMIZED FOR RTF ~3.5x (A10G GPU with 32 codebooks):
    - Larger chunks reduce decode overhead
    - First chunk balances latency vs buffer
    - Subsequent chunks are larger for smoother playback

    At CSM's 12.5Hz frame rate with RTF 3.5x:
    - 4 frames = 320ms audio, ~1.1s to generate
    - 8 frames = 640ms audio, ~2.2s to generate
    """
    initial_batch_size: int = 4    # 320ms audio, ~1.1s to generate
    batch_size: int = 8            # Larger chunks for smoother streaming
    buffer_size: int = 8           # Match batch size
    max_audio_length_ms: int = 8000   # Allow longer responses
    temperature: float = 0.9       # Match official generator
    topk: int = 50                 # Sesame standard


# FAST config - smaller first chunk for quicker first audio
# OPTIMIZED: 2 frames = ~160ms first chunk for Sesame-level latency
FAST_CONFIG = StreamingConfig(
    initial_batch_size=2,          # 160ms audio - FASTEST possible first chunk
    batch_size=6,                  # Medium chunks for smooth streaming
    buffer_size=6,
    max_audio_length_ms=5000,
    temperature=0.9,
    topk=50
)


class StreamingTTSEngine:
    """
    Streaming CSM-1B with real-time audio chunk generation.

    KEY INNOVATION:
    - Don't wait for full audio generation
    - Yield chunks as they're produced
    - User hears audio starting in ~200ms instead of 8+ seconds

    RESEARCH FINDINGS APPLIED:
    - Text preprocessing (lowercase, remove punctuation) improves quality
    - temperature=0.8, topk=50 for natural speech
    - Warmup with 2 generations before first real use
    - CONVERSATION CONTEXT is key to CSM quality (pass previous turns)
    """

    SAMPLE_RATE = 24000
    VOICE_PROMPT_PATH = TTS.voice_prompt_path
    VOICE_PROMPT_TEXT = "oh hey! yeah im doing pretty good"

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Text preprocessing for CSM - COMMUNITY OPTIMIZED.

        Research finding: Removing punctuation and using lowercase
        improves CSM-1B output quality. Keep apostrophes for contractions
        (I'm, don't, etc.) and $ for money values.

        From Sesame research docs:
        "Community workarounds discovered: Removing punctuation (except
        apostrophes, $ signs) and using lowercase improves quality"
        """
        import re

        # Strip whitespace first
        text = text.strip()

        # Convert to lowercase (research shows this improves quality)
        text = text.lower()

        # Remove punctuation EXCEPT apostrophes and $
        # Keep: a-z, 0-9, spaces, apostrophes, $
        # Remove: . , ! ? ; : " - ( ) [ ] { } etc.
        text = re.sub(r"[^\w\s'$]", "", text)

        # Normalize multiple spaces to single space
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def __init__(self):
        self._model = None
        self._text_tokenizer = None
        self._audio_tokenizer = None
        self._initialized = False
        self._device = "cuda"

        # Voice prompt for consistent voice
        self._voice_prompt = None

        # Conversation context
        self._context: List = []

        # Performance tracking
        self._total_generations = 0
        self._total_first_chunk_time = 0.0
        self._total_audio_seconds = 0.0

        # Enable optimizations
        self._setup_optimizations()

    def _setup_optimizations(self):
        """Configure GPU optimizations for maximum speed."""
        # Enable TF32 for faster computation on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True

        # Set memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)

    def _load_voice_prompt(self):
        """Load Maya's voice identity prompt from disk."""
        import os
        from generator import Segment

        logger.info("Loading voice prompt...")
        try:
            if os.path.exists(self.VOICE_PROMPT_PATH):
                voice_data = torch.load(self.VOICE_PROMPT_PATH)
                voice_audio = voice_data['audio'].to(self._device)

                if voice_audio.dim() > 1:
                    voice_audio = voice_audio.squeeze()

                voice_text = voice_data.get('text', self.VOICE_PROMPT_TEXT)
                self._voice_prompt = Segment(
                    text=voice_text,
                    speaker=TTS.speaker_id,
                    audio=voice_audio
                )
                logger.info(f"  Voice prompt loaded: {len(voice_audio)/self.SAMPLE_RATE:.1f}s")
            else:
                logger.warning(f"  Voice prompt not found at {self.VOICE_PROMPT_PATH}")
                self._voice_prompt = None
        except Exception as e:
            logger.warning(f"  Failed to load voice prompt: {e}")
            self._voice_prompt = None

    def _crossfade_chunks(
        self,
        previous_tail: Optional[torch.Tensor],
        current_chunk: torch.Tensor,
        crossfade_samples: int = 240  # 10ms at 24kHz
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply proper crossfade between consecutive audio chunks.

        This is the professional audio engineering solution for seamless
        chunk transitions. Instead of simple fade-in/fade-out that attenuates
        the signal, we crossfade overlapping regions to maintain energy.

        How it works:
        1. Keep the last N samples from previous chunk (these are NOT output)
        2. Prepend crossfaded version to current chunk
        3. The tail of current chunk is saved for next crossfade (not output yet)

        Args:
            previous_tail: Last crossfade_samples from previous chunk (None for first)
            current_chunk: Current audio chunk
            crossfade_samples: Number of samples to crossfade (10ms = 240 @ 24kHz)

        Returns:
            (processed_audio_to_output, new_tail_for_next_chunk)
        """
        if len(current_chunk) < crossfade_samples * 2:
            # Chunk too short for crossfade, just return as-is
            return current_chunk, None

        # Remove DC offset
        current_chunk = current_chunk - current_chunk.mean()

        # Extract tail for next crossfade (will NOT be output in this chunk)
        new_tail = current_chunk[-crossfade_samples:].clone()

        # Trim tail off this chunk (it will be crossfaded with next chunk)
        output_chunk = current_chunk[:-crossfade_samples].clone()

        if previous_tail is not None and len(previous_tail) == crossfade_samples:
            # Create crossfade curves (equal power crossfade for smooth energy)
            t = torch.linspace(0, 1, crossfade_samples, device=current_chunk.device)
            # Equal power: fade_out = cos(t * pi/2), fade_in = sin(t * pi/2)
            fade_out = torch.cos(t * 3.14159 / 2)
            fade_in = torch.sin(t * 3.14159 / 2)

            # Crossfade: previous tail fades out, current start fades in
            crossfaded = previous_tail * fade_out + current_chunk[:crossfade_samples] * fade_in

            # Prepend crossfaded region to output (after the crossfade point)
            output_chunk = torch.cat([crossfaded, output_chunk])

        return output_chunk, new_tail

    def _apply_final_fade(self, audio: torch.Tensor, fade_samples: int = 120) -> torch.Tensor:
        """Apply fade-out to final chunk to prevent click at end."""
        if len(audio) < fade_samples:
            return audio

        fade_out = torch.linspace(1.0, 0.0, fade_samples, device=audio.device)
        audio = audio.clone()
        audio[-fade_samples:] = audio[-fade_samples:] * fade_out
        return audio

    def _apply_noise_reduction(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Simple audio cleanup - remove DC offset.
        Heavy filtering can degrade speech quality.
        """
        if len(audio) < 100:
            return audio

        # Just remove DC offset - this is safe and effective
        audio = audio - audio.mean()

        return audio

    def _generate_frame_fast(
        self,
        model,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        num_codebooks: int = 16
    ) -> torch.Tensor:
        """
        Fast generate_frame that stops depth decoder early.

        This is the KEY OPTIMIZATION from i-LAVA paper:
        - Instead of 32 decoder iterations, only do 16 (or fewer)
        - Pad remaining codebooks with zeros
        - 50% fewer decoder iterations = ~50% faster
        """
        from models import sample_topk, _index_causal_mask

        # Get the underlying model (may be wrapped by torch.compile)
        m = model._orig_mod if hasattr(model, '_orig_mod') else model

        dtype = next(m.parameters()).dtype
        b, s, _ = tokens.size()

        # Run backbone (same as original)
        assert m.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(m.backbone_causal_mask, input_pos)
        embeds = m._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = m.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        # Generate codebook 0 (semantic)
        last_h = h[:, -1, :]
        c0_logits = m.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = m._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Depth decoder - STOP EARLY at num_codebooks
        m.decoder.reset_caches()
        for i in range(1, num_codebooks):  # Only generate first num_codebooks (e.g., 16 instead of 32)
            curr_decoder_mask = _index_causal_mask(m.decoder_causal_mask, curr_pos)
            decoder_h = m.decoder(m.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], m.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = m._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        # Pad remaining codebooks with zeros for full 32-codebook compatibility
        if num_codebooks < 32:
            padding = torch.zeros(b, 32 - num_codebooks, dtype=curr_sample.dtype, device=curr_sample.device)
            curr_sample = torch.cat([curr_sample, padding], dim=1)

        return curr_sample

    def initialize(self) -> None:
        """Load CSM-1B model with optimizations."""
        if self._initialized:
            return

        logger.info("Loading Streaming CSM-1B...")
        start = time.time()

        try:
            # Clear GPU cache first
            torch.cuda.empty_cache()

            from huggingface_hub import hf_hub_download
            from models import Model
            from moshi.models import loaders
            from transformers import AutoTokenizer
            from tokenizers.processors import TemplateProcessing

            # Load model - NO torch.compile (causes issues, minimal RTF improvement)
            model = Model.from_pretrained("sesame/csm-1b")

            # Load fine-tuned weights if available (LoRA v3 merged checkpoint)
            if os.path.exists(TTS.custom_model_path):
                logger.info(f"Loading fine-tuned weights from {TTS.custom_model_path}...")
                sd = torch.load(TTS.custom_model_path, map_location=self._device)
                valid_keys = set(model.state_dict().keys())
                filtered = {k: v for k, v in sd.items() if k in valid_keys}
                model.load_state_dict(filtered, strict=True)
                logger.info(f"  Fine-tuned model loaded ({len(filtered)} keys)")
                del sd, filtered
            else:
                logger.warning(f"  Custom model not found at {TTS.custom_model_path}, using base CSM-1B")

            model.to(device=self._device, dtype=torch.bfloat16)
            model.setup_caches(1)
            self._model = model
            logger.info("Model loaded (no torch.compile for stability)")

            # Load tokenizers
            tokenizer_name = "meta-llama/Llama-3.2-1B"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            bos = tokenizer.bos_token
            eos = tokenizer.eos_token
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
            )
            self._text_tokenizer = tokenizer

            # Load audio tokenizer (Mimi)
            # MUST use 32 codebooks for quality - reduced codebooks corrupts audio!
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            mimi = loaders.get_mimi(mimi_weight, device=self._device)
            self._num_codebooks = 32  # Full quality - reduced codebooks breaks audio
            mimi.set_num_codebooks(self._num_codebooks)
            self._audio_tokenizer = mimi
            logger.info(f"Using {self._num_codebooks} codebooks (full quality)")

            # Store original generate_frame for custom fast version
            self._original_generate_frame = model._orig_mod.generate_frame if hasattr(model, '_orig_mod') else None
            self._csm_full_codebooks = 32  # CSM always uses 32 internally

            elapsed = time.time() - start
            logger.info(f"Streaming CSM-1B loaded in {elapsed:.1f}s")

            # Load voice prompt for consistent voice identity
            self._load_voice_prompt()

            # Warmup
            self._warmup()

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load streaming CSM: {e}")
            raise

    def _warmup(self):
        """Warmup the model for faster first inference.

        AGGRESSIVE WARMUP:
        - 3+ generations to fully compile all code paths
        - Varying lengths to compile different buffer states
        - Forces CUDA kernel compilation and memory allocation
        """
        logger.info("Warming up streaming TTS (aggressive warmup)...")
        try:
            # Sync before warmup
            torch.cuda.synchronize()
            warmup_start = time.time()

            # Warmup 1: Very short (triggers first chunk path)
            test_text1 = "hmm"
            chunk_count = 0
            for _ in self._generate_frames_sync(test_text1, speaker=0, context=[]):
                chunk_count += 1
            logger.debug(f"Warmup 1: {chunk_count} chunks")

            # Warmup 2: Medium (triggers subsequent chunk path)
            test_text2 = "well let me think about that"
            chunk_count = 0
            for _ in self._generate_frames_sync(test_text2, speaker=0, context=[]):
                chunk_count += 1
            logger.debug(f"Warmup 2: {chunk_count} chunks")

            # Warmup 3: Longer (ensures full path compilation)
            test_text3 = "oh thats interesting i think we should explore that further"
            chunk_count = 0
            for _ in self._generate_frames_sync(test_text3, speaker=0, context=[]):
                chunk_count += 1
            logger.debug(f"Warmup 3: {chunk_count} chunks")

            torch.cuda.synchronize()
            warmup_time = time.time() - warmup_start
            logger.info(f"Warmup complete in {warmup_time:.1f}s (3 generations)")

        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def _tokenize_text(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text segment."""
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        return text_frame.to(self._device), text_frame_mask.to(self._device)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio segment."""
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(0)

        audio = audio.to(self._device)
        audio_tokens = self._audio_tokenizer.encode(audio)[0]

        # Add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self._device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self._device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self._device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a full segment (text + audio)."""
        text_tokens, text_masks = self._tokenize_text(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return (
            torch.cat([text_tokens, audio_tokens], dim=0),
            torch.cat([text_masks, audio_masks], dim=0)
        )

    @torch.inference_mode()
    def _generate_frames_sync(
        self,
        text: str,
        speaker: int,
        context: List,
        config: Optional[StreamingConfig] = None
    ):
        """
        Generate audio frames synchronously, yielding chunks.

        This is the core streaming implementation.
        """
        if config is None:
            config = StreamingConfig()

        self._model.reset_caches()

        max_generation_len = int(config.max_audio_length_ms / 80)

        # Build prompt from context
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        # Add generation text
        gen_tokens, gen_mask = self._tokenize_text(text, speaker)
        tokens.append(gen_tokens)
        tokens_mask.append(gen_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self._device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self._device)

        # Check context length
        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if prompt_tokens.size(0) >= max_context_len:
            # Truncate context if too long
            logger.warning(f"Context too long ({prompt_tokens.size(0)}), truncating")
            prompt_tokens = prompt_tokens[-max_context_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_context_len:]

        # Initialize generation state
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self._device)

        # PRE-ALLOCATE reusable tensors (optimization: avoid allocation in loop)
        zero_token = torch.zeros(1, 1, dtype=torch.long, device=self._device)
        zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=self._device)

        # Buffer configuration
        first_chunk = True
        buffer_size = config.initial_batch_size
        frame_buffer = []
        previous_tail = None  # For crossfading between chunks

        for i in range(max_generation_len):
            # Generate one frame using STANDARD generate_frame (full 32 codebooks)
            sample = self._model.generate_frame(
                curr_tokens,
                curr_tokens_mask,
                curr_pos,
                config.temperature,
                config.topk
            )

            if torch.all(sample == 0):
                break  # EOS

            frame_buffer.append(sample)

            # Update state for next frame - match official generator exactly
            curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), zero_mask], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

            # Yield chunk when buffer is full
            if len(frame_buffer) >= buffer_size:
                # Decode frames to audio (full 32 codebooks)
                stacked = torch.stack(frame_buffer).permute(1, 2, 0)
                audio_chunk = self._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)

                # Apply proper crossfade between chunks (professional audio approach)
                audio_chunk, previous_tail = self._crossfade_chunks(previous_tail, audio_chunk)

                yield audio_chunk

                frame_buffer = []
                first_chunk = False
                buffer_size = config.buffer_size

        # Yield remaining frames (final chunk)
        if frame_buffer:
            stacked = torch.stack(frame_buffer).permute(1, 2, 0)
            # Decode with full 32 codebooks
            audio_chunk = self._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)

            # Remove DC offset
            audio_chunk = audio_chunk - audio_chunk.mean()

            # For final chunk, we need to include the previous tail and crossfade
            if previous_tail is not None:
                # Create crossfade
                crossfade_samples = len(previous_tail)
                if len(audio_chunk) >= crossfade_samples:
                    t = torch.linspace(0, 1, crossfade_samples, device=audio_chunk.device)
                    fade_out = torch.cos(t * 3.14159 / 2)
                    fade_in = torch.sin(t * 3.14159 / 2)
                    crossfaded = previous_tail * fade_out + audio_chunk[:crossfade_samples] * fade_in
                    audio_chunk = torch.cat([crossfaded, audio_chunk[crossfade_samples:]])
                else:
                    # Very short final chunk, just prepend the tail
                    audio_chunk = torch.cat([previous_tail, audio_chunk])

            # Apply fade-out to final chunk to prevent click at end
            audio_chunk = self._apply_final_fade(audio_chunk)
            yield audio_chunk
        elif previous_tail is not None:
            # No remaining frames but we have a tail to output
            audio_chunk = self._apply_final_fade(previous_tail)
            yield audio_chunk

    async def generate_stream(
        self,
        text: str,
        use_context: bool = True,
        config: Optional[StreamingConfig] = None
    ) -> AsyncIterator[torch.Tensor]:
        """
        Async streaming audio generation.

        Yields audio chunks as they're generated.
        First chunk arrives in ~200ms instead of waiting 8+ seconds.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        first_chunk_time = None
        total_audio_samples = 0

        # Build context
        from generator import Segment
        context = []
        if self._voice_prompt is not None:
            context.append(self._voice_prompt)
        if use_context:
            context.extend(self._context)

        # Apply text preprocessing for better quality
        original_text = text
        text = self.preprocess_text(text)
        logger.debug(f"Streaming generation: '{original_text[:50]}...' -> '{text[:50]}...' with {len(context)} context segments")

        # Run generation in executor to not block event loop
        loop = asyncio.get_event_loop()

        # Use a queue to pass chunks between threads
        chunk_queue = asyncio.Queue()
        generation_done = asyncio.Event()

        def generate_sync():
            try:
                for chunk in self._generate_frames_sync(
                    text=text,
                    speaker=TTS.speaker_id,
                    context=context,
                    config=config
                ):
                    # Put chunk in queue
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(generation_done.set)

        # Start generation in thread
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(generate_sync)

        # Yield chunks as they arrive
        # OPTIMIZED: Reduced timeout from 100ms to 20ms for faster first chunk delivery
        chunk_count = 0
        while not generation_done.is_set() or not chunk_queue.empty():
            try:
                chunk = await asyncio.wait_for(
                    chunk_queue.get(),
                    timeout=0.02  # 20ms timeout (was 100ms - too much context switch overhead)
                )

                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    logger.info(f"First chunk in {first_chunk_time*1000:.0f}ms")

                total_audio_samples += len(chunk)
                chunk_count += 1

                yield chunk

            except asyncio.TimeoutError:
                continue

        # Wait for thread to complete
        future.result()
        executor.shutdown(wait=False)

        # Track metrics
        total_time = time.time() - start_time
        audio_duration = total_audio_samples / self.SAMPLE_RATE
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        self._total_generations += 1
        if first_chunk_time:
            self._total_first_chunk_time += first_chunk_time
        self._total_audio_seconds += audio_duration

        logger.info(
            f"Streaming complete: {chunk_count} chunks, "
            f"{audio_duration:.1f}s audio, RTF={rtf:.2f}x, "
            f"first_chunk={first_chunk_time*1000:.0f}ms"
        )

    def generate(self, text: str, use_context: bool = True) -> torch.Tensor:
        """
        Non-streaming generation (for compatibility).

        Collects all chunks and returns full audio.
        """
        if not self._initialized:
            self.initialize()

        # Apply text preprocessing for better quality
        text = self.preprocess_text(text)

        from generator import Segment
        context = []
        if self._voice_prompt is not None:
            context.append(self._voice_prompt)
        if use_context:
            context.extend(self._context)

        chunks = []
        for chunk in self._generate_frames_sync(
            text=text,
            speaker=TTS.speaker_id,
            context=context
        ):
            chunks.append(chunk)

        if chunks:
            return torch.cat(chunks, dim=0)
        else:
            return torch.zeros(self.SAMPLE_RATE)  # 1 second silence fallback

    def generate_fast(self, text: str, use_context: bool = True) -> torch.Tensor:
        """
        FAST generation for short utterances (starters, 2-4 words).

        Uses FAST_CONFIG with:
        - Smaller batch size (4 frames = ~320ms first chunk)
        - Shorter max length (3 seconds)
        - Optimized for minimum latency
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Apply text preprocessing
        text = self.preprocess_text(text)

        from generator import Segment
        context = []
        if self._voice_prompt is not None:
            context.append(self._voice_prompt)
        if use_context:
            context.extend(self._context)

        chunks = []
        for chunk in self._generate_frames_sync(
            text=text,
            speaker=TTS.speaker_id,
            context=context,
            config=FAST_CONFIG  # Use fast config!
        ):
            chunks.append(chunk)

        if chunks:
            audio = torch.cat(chunks, dim=0)
        else:
            audio = torch.zeros(self.SAMPLE_RATE)

        elapsed = time.time() - start_time
        duration = len(audio) / self.SAMPLE_RATE
        logger.info(f"FAST generate: '{text[:30]}' -> {duration:.1f}s audio in {elapsed*1000:.0f}ms")

        return audio

    def set_voice_prompt(self, text: str, audio: torch.Tensor) -> None:
        """Set Maya's voice identity prompt."""
        if not self._initialized:
            self.initialize()

        from generator import Segment

        if audio.dim() > 1:
            audio = audio.squeeze()

        self._voice_prompt = Segment(
            text=text,
            speaker=TTS.speaker_id,
            audio=audio
        )
        logger.info("Voice prompt set")

    def add_context(self, text: str, audio: torch.Tensor, is_user: bool = True) -> None:
        """Add turn to conversation context."""
        if not self._initialized:
            self.initialize()

        from generator import Segment

        if audio.dim() > 1:
            audio = audio.squeeze()

        speaker_id = 1 if is_user else TTS.speaker_id

        segment = Segment(
            text=text,
            speaker=speaker_id,
            audio=audio
        )

        self._context.append(segment)

        # Keep only last N turns
        if len(self._context) > TTS.context_turns:
            self._context = self._context[-TTS.context_turns:]

    def clear_context(self) -> None:
        """Clear conversation context (keep voice prompt)."""
        self._context.clear()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def get_stats(self) -> dict:
        """Get performance statistics."""
        avg_first_chunk = 0.0
        if self._total_generations > 0:
            avg_first_chunk = self._total_first_chunk_time / self._total_generations

        return {
            "total_generations": self._total_generations,
            "avg_first_chunk_ms": avg_first_chunk * 1000,
            "total_audio_seconds": self._total_audio_seconds,
            "context_turns": len(self._context),
        }
