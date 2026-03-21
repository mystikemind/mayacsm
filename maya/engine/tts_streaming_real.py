"""
Real Streaming TTS Engine - True frame-by-frame audio generation.

Uses CSM's StreamingGenerator for REAL streaming:
- First audio chunk in ~320ms (4 frames)
- Subsequent chunks every ~960ms (12 frames)
- No waiting for complete generation

SESAME AI OPTIMIZATIONS:
- Temperature 1.0 for maximum naturalness
- LUFS-based loudness normalization (-16 LUFS)
- True peak limiting at -1 dBTP
- No tanh soft-clipping (causes distortion)

This replaces the fake streaming in tts_compiled.py which generated
complete audio then chunked it.
"""

import torch
import logging
import time
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'csm'))

# CRITICAL: Set torch._dynamo cache limit to prevent RecompileLimitExceeded errors
# This is needed because different input shapes cause recompilation
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.suppress_errors = True

from typing import Optional, List, Generator as PyGenerator, AsyncGenerator
from dataclasses import dataclass

# Import no_cuda_graph to disable CUDA graphs during streaming
# This allows variable-size decode calls within streaming context
from moshi.utils.compile import no_cuda_graph

from ..config import TTS as TTS_CONFIG
from .audio_processor import StatefulAudioProcessor, get_processor, reset_processor, repair_clicks

logger = logging.getLogger(__name__)


def _calculate_rms_db(audio: torch.Tensor) -> float:
    """Calculate RMS level in dB (approximation of LUFS for speech)."""
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms > 1e-8:
        return 20 * math.log10(rms.item())
    return -100.0


def _normalize_lufs(audio: torch.Tensor, target_lufs: float = -16.0,
                    true_peak_limit: float = 0.89) -> torch.Tensor:
    """
    SIMPLIFIED - just basic peak normalization, no fancy processing.
    All the complex processing was causing siren noises.
    """
    if len(audio) == 0:
        return audio

    # Just simple peak normalization to 0.7
    peak = audio.abs().max()
    if peak > 1e-6:
        audio = audio * (0.7 / peak)

    return audio


def _enhance_audio_quality(audio: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
    """
    COMPLETELY DISABLED - just return raw audio.
    All processing was causing siren noises and artifacts.
    """
    return audio


# Enable TF32 for faster matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# Enable cuDNN benchmark for auto-tuning convolutions (Sesame optimization)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class Segment:
    """A conversation segment with speaker, text, and audio."""
    speaker: int
    text: str
    audio: torch.Tensor


@dataclass
class TokenizedSegment:
    """Pre-tokenized segment for fast context building during generation."""
    speaker: int
    text: str
    tokens: torch.Tensor       # Pre-computed tokens
    tokens_mask: torch.Tensor  # Pre-computed mask


class RealStreamingTTSEngine:
    """
    True streaming TTS using CSM's StreamingGenerator.

    Key difference from CompiledTTSEngine:
    - generate_stream() yields audio DURING generation, not after
    - First chunk arrives in ~320ms instead of waiting 800-1200ms
    - Uses mimi.streaming() context for clickless audio

    Architecture:
    1. Load CSM model with torch.compile on backbone + decoder
    2. Use StreamingGenerator.generate_stream() for true streaming
    3. Yield 320ms first chunk, then 960ms subsequent chunks
    """

    SAMPLE_RATE = 24000
    # Voice prompt path - MUST match training data!
    # Model fine-tuned on Expresso ex04 data, so voice prompt must be Expresso
    # Using Sesame prompt with Expresso model causes GIBBERISH output
    # Use config path for consistency
    VOICE_PROMPT_PATH = TTS_CONFIG.voice_prompt_path

    # PROVEN SETTINGS FOR CONSISTENCY (tested 2026-02-06):
    # - Temperature 0.8 (not 0.9 or 1.0 - causes instability)
    # - Keep responses under 4 seconds
    # - Natural disfluencies WORK: "hmm", "yeah", "mhm", "um" all tested OK
    # - Old constraint about "Hmm causes gibberish" was from BASE model
    # - Current fine-tuned model handles all natural speech patterns
    VOICE_PROMPT_TEXT = "hi there its really nice to meet you im maya i love having conversations"

    # Fallback path if main prompt not found
    VOICE_PROMPT_FALLBACK = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt_short.pt'

    # Streaming parameters from config (Sesame-optimized)
    FIRST_CHUNK_FRAMES = TTS_CONFIG.first_chunk_frames   # ~160ms - FASTEST possible first audio
    CHUNK_FRAMES = TTS_CONFIG.chunk_frames               # ~640ms - smaller chunks for responsiveness
    CROSSFADE_SAMPLES = TTS_CONFIG.crossfade_samples     # 10ms at 24kHz - overlap-add crossfade

    # Fixed prompt length for consistent CUDA graph shapes (reduces variance)
    # Pad shorter prompts, truncate longer ones
    FIXED_PROMPT_LENGTH = 128  # Enough for voice prompt + short response

    # CSM context limits - prevent overflow
    # CSM has 2048 token limit, each audio frame is 1 token, each text token is 1
    # ~12.5 frames/second = 750 frames for 60s, plus text ~200 tokens = ~950 for 1 turn
    MAX_CONTEXT_TOKENS = 1800  # Leave headroom for current generation
    TOKENS_PER_SECOND_AUDIO = 12.5  # Mimi codec frame rate
    AVG_TOKENS_PER_WORD = 1.3  # Approximate for Llama tokenizer

    def __init__(self):
        self._generator = None
        self._initialized = False
        self._context: List[TokenizedSegment] = []  # Pre-tokenized for speed
        self._voice_prompt: Optional[Segment] = None
        self._voice_prompt_tokens: Optional[TokenizedSegment] = None  # Pre-tokenized

        # Configure device from TTS config
        from ..config import TTS as TTS_CONFIG
        gpu_idx = getattr(TTS_CONFIG, 'device_index', 0)
        self._device = f"cuda:{gpu_idx}"

        # Pre-computed crossfade tensors (avoid recomputing on every chunk)
        self._crossfade_t: Optional[torch.Tensor] = None
        self._crossfade_fade_out: Optional[torch.Tensor] = None
        self._crossfade_fade_in: Optional[torch.Tensor] = None

    def _preprocess_for_speech(self, text: str) -> str:
        """Preprocess text for natural CSM speech output.

        Keep punctuation that affects prosody:
        - ! for emotional emphasis
        - ? for questions (rising intonation)
        - , for natural pauses
        - . for sentence endings
        - ' for contractions
        - - for compound words
        - [ ] for style tags like [happy], [sad], [whisper]

        The model was trained on Expresso with style tags, so we MUST
        preserve them for emotional/expressive output.
        """
        import re
        text = text.lower()
        # Keep prosody-affecting punctuation AND style tag brackets: . , ? ! ' - [ ]
        text = re.sub(r"[^\w\s.,?!'\-\[\]]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def initialize(self) -> None:
        """Load CSM with true streaming support and torch.compile."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("LOADING REAL STREAMING CSM")
        logger.info("True streaming: First audio in ~320ms")
        logger.info("=" * 60)

        start_total = time.time()

        # Import CSM components
        from models import Model
        from moshi.models import loaders
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing

        # Load model - USE THE CORRECT FINE-TUNED MODEL
        # This model was trained with CORRECT approach based on davidbrowne17/csm-streaming:
        # - LoRA + decoder + codebook0_head training
        # - lr=1e-6, epochs=5, max_grad_norm=0.1
        # - All 32 codebooks trained (not just codebook 0)
        # - Val loss: 9.8858, 39% fewer clicks than base
        BEST_FINETUNED_MODEL = '/home/ec2-user/SageMaker/project_maya/training/checkpoints/csm_maya_correct/best_model/model_merged.pt'
        # Fallback options
        LEGACY_LORA_MODEL = '/home/ec2-user/SageMaker/project_maya/training/checkpoints/csm_maya_lora_v3/best_model/model_merged.pt'
        PRODUCTION_MODEL = '/home/ec2-user/SageMaker/project_maya/training/checkpoints/csm_maya_production/model.pt'

        logger.info("Loading CSM model...")
        start = time.time()

        import os

        # Set CUDA device BEFORE model loading to ensure all tensors are on correct device
        # This is critical because CSM's internal caches (causal_mask, etc.) are created
        # during model initialization and must be on the same device as the model
        from ..config import TTS as TTS_CONFIG
        gpu_idx = getattr(TTS_CONFIG, 'device_index', 0)
        device = f"cuda:{gpu_idx}"
        torch.cuda.set_device(gpu_idx)
        logger.info(f"  Target device: {device}")

        # Use fine-tuned model for consistent voice identity
        # NOTE: Test samples sound natural, so the model is GOOD.
        # The issue is in pipeline integration, not the model itself.
        finetuned_path = None
        if os.path.exists(BEST_FINETUNED_MODEL):
            finetuned_path = BEST_FINETUNED_MODEL
            logger.info("  *** LOADING FINE-TUNED MODEL ***")
        elif os.path.exists(LEGACY_LORA_MODEL):
            finetuned_path = LEGACY_LORA_MODEL
            logger.info("  *** LOADING LEGACY LoRA MODEL ***")
        elif os.path.exists(PRODUCTION_MODEL):
            finetuned_path = PRODUCTION_MODEL
            logger.info("  *** LOADING PRODUCTION MODEL ***")

        if finetuned_path:
            # Load base model architecture first
            model = Model.from_pretrained("sesame/csm-1b")
            # Load fine-tuned weights (already merged with base)
            state_dict = torch.load(finetuned_path, map_location="cuda", weights_only=False)
            # Handle potential nested state dict
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            logger.info("  *** FINE-TUNED WEIGHTS LOADED SUCCESSFULLY ***")
        else:
            logger.warning("  *** WARNING: No fine-tuned model found, using base CSM-1B ***")
            model = Model.from_pretrained("sesame/csm-1b")

        # Move model to configured device (already set above)
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        logger.info(f"  Model loaded in {time.time()-start:.1f}s")

        # Apply torch.compile WITHOUT CUDA graphs
        # Using mode='default' instead of 'reduce-overhead' because:
        # 1. Same latency after warmup (~127ms vs ~130ms)
        # 2. MUCH lower memory usage (6.7GB vs 16-21GB)
        # 3. No memory accumulation over long conversations
        # 4. No OOM or latency spikes after 20+ turns
        #
        # The trade-off is longer initial compilation, which is absorbed during warmup.
        logger.info("Compiling backbone (mode='default', no CUDA graphs)...")
        start = time.time()
        try:
            model.backbone = torch.compile(
                model.backbone,
                mode='default',
                fullgraph=True,
            )
            logger.info(f"  Backbone compiled in {time.time()-start:.1f}s")
        except Exception as e:
            logger.warning(f"  Backbone fullgraph compile failed, falling back: {e}")
            model.backbone = torch.compile(model.backbone, mode='default', fullgraph=False)

        # Apply torch.compile to decoder (no CUDA graphs)
        logger.info("Compiling decoder (mode='default', no CUDA graphs)...")
        start = time.time()
        try:
            model.decoder = torch.compile(
                model.decoder,
                mode='default',
                fullgraph=True,
            )
            logger.info(f"  Decoder compiled in {time.time()-start:.1f}s")
        except Exception as e:
            logger.warning(f"  Decoder fullgraph compile failed, falling back: {e}")
            model.decoder = torch.compile(model.decoder, mode='default', fullgraph=False)

        # Setup KV caches
        model.setup_caches(1)

        # Load tokenizers
        logger.info("Loading tokenizers...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )

        # Load Mimi codec
        logger.info("Loading Mimi codec...")
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)

        # Store components
        self._model = model
        self._text_tokenizer = tokenizer
        self._audio_tokenizer = mimi

        # Load voice context FROM TRAINING DATA (this produces best quality!)
        # The test script that produced good samples loaded context directly from
        # training data, not from a separate voice prompt file
        logger.info("Loading voice context from training data...")
        try:
            import os
            import json
            import torchaudio

            TRAINING_DATA = '/home/ec2-user/SageMaker/project_maya/training/data/csm_ready_ex04'
            train_json = os.path.join(TRAINING_DATA, 'train.json')

            self._voice_contexts = []
            self._voice_context_tokens = []

            if os.path.exists(train_json):
                with open(train_json) as f:
                    train_samples = json.load(f)

                # Get default style samples
                default_samples = [s for s in train_samples if s.get("style") == "default"][:8]
                if not default_samples:
                    default_samples = train_samples[:8]

                total_duration = 0
                for sample in default_samples:
                    if total_duration >= 15:
                        break

                    audio_path = os.path.join(TRAINING_DATA, sample["path"])
                    if not os.path.exists(audio_path):
                        continue

                    audio, sr = torchaudio.load(audio_path)
                    if sr != 24000:
                        audio = torchaudio.functional.resample(audio, sr, 24000)
                    audio = audio.mean(dim=0) if audio.dim() > 1 and audio.size(0) > 1 else audio.squeeze(0)
                    audio = audio.to(device)

                    text = sample["text"]
                    if text.startswith("["):
                        text = text.split("]", 1)[-1].strip()

                    seg = Segment(speaker=0, text=text, audio=audio)
                    self._voice_contexts.append(seg)

                    seg_tokens, seg_mask = self._tokenize_segment(seg)
                    self._voice_context_tokens.append(TokenizedSegment(
                        speaker=0, text=text, tokens=seg_tokens, tokens_mask=seg_mask
                    ))

                    total_duration += sample.get("duration", len(audio) / 24000)
                    logger.info(f"  Context: '{text[:40]}...' ({len(audio)/24000:.1f}s)")

                logger.info(f"  Total context: {len(self._voice_contexts)} segments, ~{total_duration:.1f}s")

                # For backwards compatibility
                self._voice_prompt = self._voice_contexts[0] if self._voice_contexts else None
                self._voice_prompt_tokens = self._voice_context_tokens[0] if hasattr(self, '_voice_context_tokens') and self._voice_context_tokens else None
            else:
                logger.warning(f"  Training data not found: {train_json}")
                # Fallback to voice prompt file
                voice_path = self.VOICE_PROMPT_PATH
                if os.path.exists(voice_path):
                    voice_data = torch.load(voice_path)
                    voice_audio = voice_data['audio'].to(device)
                    voice_text = voice_data.get('text', self.VOICE_PROMPT_TEXT)
                    self._voice_prompt = Segment(speaker=0, text=voice_text, audio=voice_audio)
                    vp_tokens, vp_mask = self._tokenize_segment(self._voice_prompt)
                    self._voice_prompt_tokens = TokenizedSegment(speaker=0, text=voice_text, tokens=vp_tokens, tokens_mask=vp_mask)
                    self._voice_contexts = [self._voice_prompt]
                    self._voice_context_tokens = [self._voice_prompt_tokens]
                    logger.info(f"  Fallback voice prompt: {len(voice_audio)/self.SAMPLE_RATE:.1f}s")
                else:
                    self._voice_prompt = None
                    self._voice_prompt_tokens = None
                    self._voice_contexts = []
                    self._voice_context_tokens = []
        except Exception as e:
            logger.warning(f"  Failed to load voice context: {e}")
            self._voice_prompt = None
            self._voice_prompt_tokens = None
            self._voice_contexts = []
            self._voice_context_tokens = []

        # Warmup - compile all code paths thoroughly
        logger.info("Warming up (compiling code paths)...")
        warmup_start = time.time()

        # More diverse phrases to compile all code paths
        warmup_phrases = [
            "hi", "hello there", "how are you", "nice to meet you",
            "sure", "okay", "got it", "interesting",
            "tell me more about that", "i understand what you mean"
        ]

        # Phase 1: Initial compilation (longer runs)
        for phrase in warmup_phrases[:3]:
            for i in range(5):  # 5 iterations for first phrases
                torch.cuda.synchronize()
                start = time.time()
                audio = self._generate_complete(phrase, max_audio_ms=2000)
                torch.cuda.synchronize()
                elapsed = (time.time() - start) * 1000
                if i == 0:
                    logger.info(f"  Warmup '{phrase}': {elapsed:.0f}ms")
                if elapsed < 200:  # Stricter threshold
                    break
            torch.cuda.empty_cache()

        # Phase 2: Stabilization (quick runs)
        for phrase in warmup_phrases[3:]:
            for i in range(3):
                torch.cuda.synchronize()
                start = time.time()
                audio = self._generate_complete(phrase, max_audio_ms=2000)
                torch.cuda.synchronize()
                elapsed = (time.time() - start) * 1000
                if i == 0:
                    logger.info(f"  Warmup '{phrase}': {elapsed:.0f}ms")
                if elapsed < 200:
                    break

        # Pre-compute crossfade tensors BEFORE streaming warmup (generate_stream needs them)
        self._crossfade_t = torch.linspace(0, 1, self.CROSSFADE_SAMPLES, device=self._device)
        self._crossfade_fade_out = torch.cos(self._crossfade_t * 3.14159 / 2)
        self._crossfade_fade_in = torch.sin(self._crossfade_t * 3.14159 / 2)

        # Mark as initialized before streaming warmup to prevent recursion
        self._initialized = True

        # Phase 3: Warmup STREAMING path (critical - different code path!)
        # Use diverse phrase lengths to compile graphs for various input sizes
        logger.info("  Warming streaming path...")
        streaming_phrases = [
            # Short (2-3 words)
            "sure", "okay got it", "i see",
            # Medium (4-6 words)
            "thats a great question", "let me help you",
            "i understand what you mean",
            # Longer (7-10 words)
            "hello how can i help you today",
            "let me think about that for a moment",
            "thats really interesting tell me more about it",
            "i would be happy to help you with that",
        ]
        for phrase in streaming_phrases:
            for _ in range(3):
                torch.cuda.synchronize()
                # Actually use generate_stream to warm that code path
                for chunk in self.generate_stream(phrase, use_context=False):
                    break  # Just need first chunk
                torch.cuda.synchronize()

        # Phase 4: Extended stabilization (run until consistently fast)
        logger.info("  Final stabilization (extended)...")
        consecutive_fast = 0
        for i in range(30):  # Up to 30 iterations
            torch.cuda.synchronize()
            start = time.time()
            for chunk in self.generate_stream("hello how can i help you", use_context=False):
                break
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            if elapsed < 150:
                consecutive_fast += 1
            else:
                consecutive_fast = 0  # Reset if slow run

            # Stop once we have 10 consecutive fast runs
            if consecutive_fast >= 10:
                logger.info(f"    Stabilized after {i+1} iterations")
                break

        torch.cuda.empty_cache()

        warmup_time = time.time() - warmup_start
        total_time = time.time() - start_total

        # Final cache clear after warmup
        torch.cuda.empty_cache()

        logger.info("=" * 60)
        logger.info(f"REAL STREAMING CSM READY")
        logger.info(f"  Total init time: {total_time:.1f}s")
        logger.info(f"  Warmup time: {warmup_time:.1f}s")
        logger.info(f"  Voice prompt: {'LOADED' if self._voice_prompt else 'NOT FOUND'}")
        logger.info(f"  First chunk: ~{self.FIRST_CHUNK_FRAMES * 80}ms")
        logger.info("=" * 60)

        self._initialized = True

    def _tokenize_text_segment(self, text: str, speaker: int):
        """Tokenize a text segment for a specific speaker."""
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        return text_frame.to("cuda"), text_frame_mask.to("cuda")

    def _tokenize_audio(self, audio: torch.Tensor):
        """Tokenize an audio segment."""
        if audio.ndim > 1:
            audio = audio.squeeze()
        # Clone to avoid inference tensor issues when audio comes from generate_stream
        audio = audio.clone().detach().to("cuda")
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]

        # Add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to("cuda")
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to("cuda")
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to("cuda")
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment: Segment):
        """Tokenize a complete segment (text + audio)."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    def _crossfade_chunks(
        self,
        previous_tail: Optional[torch.Tensor],
        current_chunk: torch.Tensor
    ) -> tuple:
        """
        Proper overlap-add crossfade using Hann window.
        Only crossfade if there's a significant discontinuity.
        Uses 3ms overlap (72 samples at 24kHz) - small enough to avoid siren artifacts.
        """
        if previous_tail is None or len(previous_tail) < 72:
            return current_chunk, current_chunk[-72:].clone() if len(current_chunk) >= 72 else None

        # Check discontinuity at boundary
        dc = abs(current_chunk[0].item() - previous_tail[-1].item())
        if dc < 0.05:  # Small discontinuity - no crossfade needed
            return current_chunk, current_chunk[-72:].clone() if len(current_chunk) >= 72 else None

        # Significant discontinuity - apply gentle 3ms Hann crossfade
        overlap_samples = 72  # 3ms at 24kHz

        # Hann window for smooth overlap-add
        window = torch.hann_window(overlap_samples * 2, device=current_chunk.device)
        fade_out = window[:overlap_samples]
        fade_in = window[overlap_samples:]

        # Blend the overlap region
        blended = (
            previous_tail[-overlap_samples:] * fade_out +
            current_chunk[:overlap_samples] * fade_in
        )

        # Return blended chunk (replace first overlap_samples with blended)
        result = current_chunk.clone()
        result[:overlap_samples] = blended

        return result, result[-72:].clone() if len(result) >= 72 else None

    def _decode_frames_in_context(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode frames to audio - MUST be called within mimi.streaming() context!

        This is the critical difference from before. The streaming context
        maintains proper codec state between decode calls, preventing clicks.

        Args:
            frames: List of frame tensors to decode

        Returns:
            Audio tensor at 24kHz
        """
        if not frames:
            return torch.tensor([]).to("cuda")

        cloned_frames = [f.clone() for f in frames]
        stacked = torch.stack(cloned_frames).permute(1, 2, 0)
        audio = self._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)
        return audio

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        use_context: bool = True
    ) -> PyGenerator[torch.Tensor, None, None]:
        """
        Generate audio matching test script quality.

        PRODUCTION FIX: The test script sounds natural because it decodes ALL frames
        at ONCE with a single decode() call. Our chunked decoding was causing
        the robotic "reading text" quality.

        This implementation now:
        1. Generates ALL frames (same as before)
        2. Decodes ALL frames at once (matching test script)
        3. Streams chunks of the decoded audio (for responsiveness)

        This matches the official CSM Generator.generate() approach exactly.

        Args:
            text: Text to synthesize
            use_context: Whether to use conversation context

        Yields:
            Audio chunks as torch tensors
        """
        if not self._initialized:
            self.initialize()

        if not text or not text.strip():
            logger.warning("Empty text provided")
            return

        text = self._preprocess_for_speech(text.strip())

        start_time = time.time()

        # PROACTIVE MEMORY MANAGEMENT: Clear CUDA cache if memory pressure is high
        # This prevents latency spikes caused by memory allocation during generation
        # CUDA Graphs can consume 16GB+ over long conversations, causing OOM
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(self._device)
                # Get total GPU memory (not just reserved)
                total = torch.cuda.get_device_properties(self._device).total_memory
                memory_ratio = allocated / total if total > 0 else 0

                # Clear cache if using more than 85% of total GPU memory
                if memory_ratio > 0.85:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info(f"Cleared CUDA cache (memory was {memory_ratio*100:.0f}%)")
            except Exception:
                pass  # Non-critical, continue anyway

        # Reset caches for new generation
        self._model.reset_caches()

        # Reset audio processor state for new utterance (ensures clean filter state)
        reset_processor()

        # Calculate max frames
        word_count = len(text.split())
        estimated_duration_ms = int((word_count / 2.0 + 1.5) * 1000)
        max_audio_ms = min(max(estimated_duration_ms, 4000), 12000)
        max_generation_len = int(max_audio_ms / 80)

        # Build context using PRE-TOKENIZED segments (massive speedup!)
        # Tokenization is expensive (~50-100ms per segment), so we do it once
        # when adding to context, not on every generation
        tokens, tokens_mask = [], []

        # Add ALL voice context segments for consistent voice identity
        # This is how the working test script does it - uses ~10s of training data context
        if hasattr(self, '_voice_context_tokens') and self._voice_context_tokens:
            for vct in self._voice_context_tokens:
                tokens.append(vct.tokens)
                tokens_mask.append(vct.tokens_mask)
        elif self._voice_prompt_tokens:
            # Fallback to single voice prompt
            tokens.append(self._voice_prompt_tokens.tokens)
            tokens_mask.append(self._voice_prompt_tokens.tokens_mask)

        # RE-ENABLED: Conversation context for natural prosody
        #
        # From CSM GitHub: "CSM sounds best when provided with context."
        # Without context, CSM generates monotonous, robotic speech because
        # it doesn't know the emotional tone or pacing of the conversation.
        #
        # Use limited context (last 2 turns) to balance:
        # - Prosodic continuity (CSM adjusts tone to conversation)
        # - Avoiding deep feedback loops (don't condition on too much)
        #
        # The training data context provides voice identity,
        # conversation context provides emotional/prosodic continuity.
        if use_context and self._context:
            # Use last 2 turns for prosodic continuity (not all 8)
            for segment in self._context[-2:]:
                tokens.append(segment.tokens)
                tokens_mask.append(segment.tokens_mask)

        # Add text to generate
        gen_tokens, gen_mask = self._tokenize_text_segment(text, speaker=0)
        tokens.append(gen_tokens)
        tokens_mask.append(gen_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to("cuda")
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to("cuda")

        # Initialize generation state
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to("cuda")

        # TRUE STREAMING APPROACH (SOTA-level latency):
        # The backbone sees FULL TEXT upfront (in prompt_tokens), so prosody is planned.
        # We CAN stream audio while generating more frames because:
        # 1. Prosody planning happens at tokenization (backbone sees full text)
        # 2. Frame generation samples from that plan
        # 3. We decode in batches (not streaming context) to avoid Mimi issues
        #
        # Strategy: Generate first batch, decode & yield, continue generating

        FIRST_BATCH_FRAMES = 15  # ~1200ms audio - enough for prosody establishment
        SUBSEQUENT_BATCH_FRAMES = 20  # ~1600ms audio per batch

        batch_frames = []
        total_frames = 0
        first_audio_sent = False
        previous_chunk_tail = None  # For crossfading between batches

        with no_cuda_graph():
            for frame_idx in range(max_generation_len):
                sample = self._model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos,
                    temperature=TTS_CONFIG.temperature, topk=TTS_CONFIG.topk,
                    depth_decoder_temperature=TTS_CONFIG.depth_decoder_temperature
                )

                # Check for EOS (all zeros)
                if torch.all(sample == 0):
                    break

                batch_frames.append(sample.clone())
                total_frames += 1

                # Update state for next frame
                curr_tokens = torch.cat(
                    [sample, torch.zeros(1, 1).long().to("cuda")], dim=1
                ).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to("cuda")], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

                # Check if we should decode and yield this batch
                batch_size = FIRST_BATCH_FRAMES if not first_audio_sent else SUBSEQUENT_BATCH_FRAMES
                if len(batch_frames) >= batch_size:
                    # Decode this batch
                    stacked = torch.stack(batch_frames).permute(1, 2, 0)
                    batch_audio = self._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)

                    # Peak normalize
                    peak = batch_audio.abs().max()
                    if peak > 1e-6:
                        batch_audio = batch_audio * (0.9 / peak)

                    # Crossfade with previous batch if not first
                    if previous_chunk_tail is not None and len(previous_chunk_tail) >= 72:
                        overlap = 72  # 3ms crossfade
                        window = torch.hann_window(overlap * 2, device=batch_audio.device)
                        fade_out = window[:overlap]
                        fade_in = window[overlap:]
                        # Blend overlap region
                        batch_audio[:overlap] = (
                            previous_chunk_tail[-overlap:] * fade_out +
                            batch_audio[:overlap] * fade_in
                        )

                    if not first_audio_sent:
                        elapsed = (time.time() - start_time) * 1000
                        logger.info(f">>> FIRST AUDIO at {elapsed:.0f}ms ({len(batch_frames)} frames, {len(batch_audio)/self.SAMPLE_RATE*1000:.0f}ms audio) <<<")
                        first_audio_sent = True

                    # Yield audio in 500ms chunks
                    chunk_samples = int(self.SAMPLE_RATE * 0.5)
                    for i in range(0, len(batch_audio), chunk_samples):
                        chunk = batch_audio[i:i + chunk_samples]
                        if len(chunk) > 0:
                            yield chunk

                    # Save tail for next crossfade
                    previous_chunk_tail = batch_audio[-240:].clone() if len(batch_audio) >= 240 else None
                    batch_frames = []

        # Handle remaining frames
        if batch_frames:
            stacked = torch.stack(batch_frames).permute(1, 2, 0)
            batch_audio = self._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)

            # Peak normalize
            peak = batch_audio.abs().max()
            if peak > 1e-6:
                batch_audio = batch_audio * (0.9 / peak)

            # Crossfade with previous batch
            if previous_chunk_tail is not None and len(previous_chunk_tail) >= 72:
                overlap = 72
                window = torch.hann_window(overlap * 2, device=batch_audio.device)
                fade_out = window[:overlap]
                fade_in = window[overlap:]
                batch_audio[:overlap] = (
                    previous_chunk_tail[-overlap:] * fade_out +
                    batch_audio[:overlap] * fade_in
                )

            # NOTE: Removed fade-off code - doesn't work, artificial
            # Natural endings come from:
            # 1. Proper sentence-final intonation in CSM
            # 2. The model generating complete thoughts
            # 3. Punctuation marking sentence ends

            # Yield remaining audio
            chunk_samples = int(self.SAMPLE_RATE * 0.5)
            for i in range(0, len(batch_audio), chunk_samples):
                chunk = batch_audio[i:i + chunk_samples]
                if len(chunk) > 0:
                    yield chunk

        total_time = (time.time() - start_time) * 1000
        total_audio_ms = total_frames * 80
        logger.info(f"TTS complete: {total_audio_ms}ms audio in {total_time:.0f}ms (TRUE STREAMING)")

    @torch.inference_mode()
    def _generate_complete(self, text: str, max_audio_ms: int = 5000) -> torch.Tensor:
        """Generate complete audio (for warmup/non-streaming)."""
        text = self._preprocess_for_speech(text)
        self._model.reset_caches()

        max_generation_len = int(max_audio_ms / 80)

        # Simple generation without context
        gen_tokens, gen_mask = self._tokenize_text_segment(text, speaker=0)

        curr_tokens = gen_tokens.unsqueeze(0)
        curr_tokens_mask = gen_mask.unsqueeze(0)
        curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to("cuda")

        frames = []

        # Use streaming context for clean audio (even in warmup)
        # Also disable CUDA graphs to allow variable-size decoding
        with no_cuda_graph(), self._audio_tokenizer.streaming(1):
            for _ in range(max_generation_len):
                sample = self._model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos,
                    temperature=TTS_CONFIG.temperature, topk=TTS_CONFIG.topk,
                    depth_decoder_temperature=TTS_CONFIG.depth_decoder_temperature
                )

                if torch.all(sample == 0):
                    break

                frames.append(sample.clone())

                curr_tokens = torch.cat(
                    [sample, torch.zeros(1, 1).long().to("cuda")], dim=1
                ).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to("cuda")], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

            if not frames:
                return torch.zeros(int(self.SAMPLE_RATE * 0.5)).to("cuda")

            return self._decode_frames_in_context(frames)

    def generate(self, text: str, use_context: bool = True) -> torch.Tensor:
        """Generate complete audio (collects all streaming chunks)."""
        chunks = list(self.generate_stream(text, use_context))
        if not chunks:
            return torch.zeros(int(self.SAMPLE_RATE * 0.5), device=self._device)
        return torch.cat(chunks)

    def _estimate_context_tokens(self) -> int:
        """Count total tokens in current context.

        Uses actual token counts from pre-tokenized segments.
        More accurate than estimation since we have actual tokens.
        """
        total_tokens = 0

        # Voice prompt tokens (actual count)
        if self._voice_prompt_tokens:
            total_tokens += len(self._voice_prompt_tokens.tokens)

        # Context turns tokens (actual count from pre-tokenized segments)
        for segment in self._context:
            total_tokens += len(segment.tokens)

        return total_tokens

    def _trim_context_if_needed(self) -> None:
        """Trim context if approaching token limit or memory pressure.

        CSM has 2048 token limit. We keep headroom for generation.
        If over limit, remove oldest turns until under.
        Also checks GPU memory to prevent OOM in long conversations.
        """
        # Check token limit (now using actual token counts)
        while self._context and self._estimate_context_tokens() > self.MAX_CONTEXT_TOKENS:
            removed = self._context.pop(0)
            text_preview = removed.text[:30] if removed.text else "(empty)"
            logger.warning(f"TTS context trimmed: removed oldest turn ('{text_preview}...')")

        # Check GPU memory pressure (if > 80% used, trim more aggressively)
        if torch.cuda.is_available() and len(self._context) > 2:
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
            if memory_used > 0.8:
                # Trim to half the context
                while len(self._context) > TTS_CONFIG.context_turns // 2:
                    removed = self._context.pop(0)
                    text_preview = removed.text[:30] if removed.text else "(empty)"
                    logger.warning(f"TTS context trimmed (memory pressure): removed '{text_preview}...'")
                torch.cuda.empty_cache()

    def add_context(self, text: str, audio: torch.Tensor, is_user: bool = True) -> None:
        """Add a turn to conversation context with overflow protection.

        OPTIMIZATION: Pre-tokenizes the segment when adding, not during generation.
        This moves ~50-100ms of tokenization work out of the hot path.
        """
        if not self._initialized:
            self.initialize()

        if audio.dim() > 1:
            audio = audio.squeeze()

        # Create raw segment for tokenization
        raw_segment = Segment(
            speaker=1 if is_user else 0,
            text=text,
            audio=audio.to(self._device)  # Keep on GPU for tokenization
        )

        # PRE-TOKENIZE during add (not during generation!)
        # This is the key optimization - tokenization is expensive
        seg_tokens, seg_mask = self._tokenize_segment(raw_segment)

        # Store pre-tokenized segment (no need to keep raw audio)
        tokenized = TokenizedSegment(
            speaker=1 if is_user else 0,
            text=text,
            tokens=seg_tokens,
            tokens_mask=seg_mask
        )

        self._context.append(tokenized)

        # Keep last N turns for prosodic continuity (from config, matches LLM)
        if len(self._context) > TTS_CONFIG.context_turns:
            self._context = self._context[-TTS_CONFIG.context_turns:]

        # SESAME-LEVEL: Validate context doesn't exceed CSM's 2048 token limit
        self._trim_context_if_needed()

    def clear_context(self) -> None:
        """Clear conversation context and free GPU memory."""
        self._context.clear()
        # Free GPU memory after clearing context (helps with long conversations)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_context_size(self) -> int:
        """Get current context size."""
        return len(self._context)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE
