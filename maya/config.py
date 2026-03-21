"""
Maya Configuration - All settings in one place.

Production-grade: typed, validated, documented.
Supports environment variable overrides for deployment flexibility.

Environment Variables:
    MAYA_PROJECT_ROOT: Override project root path
    MAYA_CSM_ROOT: Override CSM path
    MAYA_GPU_INDEX: Override GPU index for components
    MAYA_VLLM_URL: Override vLLM server URL
    MAYA_DEBUG: Enable debug logging
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import torch


# =============================================================================
# PATH CONFIGURATION (with environment variable support)
# =============================================================================

def _get_project_root() -> Path:
    """Get project root with environment variable fallback."""
    env_path = os.environ.get("MAYA_PROJECT_ROOT")
    if env_path:
        return Path(env_path)
    # Default paths to check
    candidates = [
        Path("/home/ec2-user/SageMaker/project_maya"),
        Path.cwd(),
        Path(__file__).parent.parent,
    ]
    for path in candidates:
        if (path / "maya").exists():
            return path
    return candidates[0]  # Fallback to default


def _get_csm_root() -> Path:
    """Get CSM root with environment variable fallback."""
    env_path = os.environ.get("MAYA_CSM_ROOT")
    if env_path:
        return Path(env_path)
    candidates = [
        Path(__file__).parent.parent / "csm",
        Path("/home/ec2-user/SageMaker/csm"),
        Path.cwd().parent / "csm",
    ]
    for path in candidates:
        if (path / "models.py").exists() or (path / "generator.py").exists():
            return path
    return candidates[0]


PROJECT_ROOT = _get_project_root()
CSM_ROOT = _get_csm_root()
ASSETS_DIR = PROJECT_ROOT / "assets"
FILLERS_DIR = ASSETS_DIR / "fillers"
VOICE_PROMPT_DIR = ASSETS_DIR / "voice_prompt"


# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AudioConfig:
    """Audio processing settings."""
    sample_rate: int = 24000          # CSM native sample rate
    chunk_duration_ms: int = 100      # WebSocket chunk size
    channels: int = 1                 # Mono audio
    dtype: str = "float32"            # Audio data type

    @property
    def chunk_samples(self) -> int:
        """Samples per chunk."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


AUDIO = AudioConfig()


# =============================================================================
# VAD CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class VADConfig:
    """Voice Activity Detection settings - Sesame-optimized."""
    threshold: float = 0.65           # Increased from 0.5 for fewer false positives (Sesame: 0.6-0.7)
    min_speech_ms: int = 200          # Minimum speech duration
    min_silence_ms: int = 350         # Silence before turn end (Sesame: 300-400ms)
    speech_pad_ms: int = 30           # Padding around speech
    sample_rate: int = 16000          # Silero VAD sample rate
    min_audio_energy: float = 0.02    # Higher threshold to ignore background noise/echoes
    echo_cooldown_ms: int = 150       # Echo cooldown after Maya speaks (was 600ms)


VAD = VADConfig()


# =============================================================================
# STT CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class STTConfig:
    """Speech-to-Text settings."""
    model_size: str = "large-v3-turbo"  # Best accuracy with good speed
    device: str = "cuda"
    compute_type: str = "float16"       # For speed
    language: str = "en"                # English only for speed
    beam_size: int = 1                  # Greedy for speed
    vad_filter: bool = True             # Use internal VAD


STT = STTConfig()


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class LLMConfig:
    """Language Model settings - SESAME AI LEVEL natural conversation."""
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    device: str = "cuda"
    device_index: int = 0               # Use GPU 0 (fallback safe - works with single GPU)
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 40            # Allow complete 10-15 word natural sentences
    temperature: float = 0.8            # Sesame uses 0.75 - balance variety & consistency
    top_p: float = 0.92
    do_sample: bool = True
    max_history_turns: int = 12         # 6 full exchanges for context (~2 min like Sesame)

    # Maya - Natural human conversation (Sesame AI level)
    # Key insight: Focus on WARMTH and GENUINE first, length second
    # Disfluencies are strategic (30-50%), not forced every response
    system_prompt: str = """you are maya, a warm and genuine person having a voice conversation

speak naturally in 8-15 words, like a real human would
respond to what they actually said with genuine interest
be warm and present, not robotic or scripted

when it naturally fits, you can start with:
- hmm when thinking or considering
- yeah when agreeing or acknowledging
- oh when surprised or interested
- aww when being empathetic
but only when it actually fits the moment, not every response

use contractions: im, youre, dont, cant, wont, thats, whats, ive

natural response examples:
- yeah im doing pretty good, hows everything with you
- hmm thats interesting, what made you think of that
- oh really, tell me more about that
- aww that sounds tough, im here for you
- that makes sense, what happened next

adapt to their emotional tone - gentle if sad, excited if happy
ask follow-up questions to show genuine interest
speak like a real person having a genuine conversation"""


LLM = LLMConfig()


# =============================================================================
# TTS CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TTSConfig:
    """Text-to-Speech settings."""
    device: str = "cuda"
    device_index: int = int(os.environ.get("MAYA_GPU_INDEX", "0"))
    speaker_id: int = 0               # Maya's voice
    max_audio_length_ms: int = 5000   # 5 seconds max - prevents endless generation
    context_turns: int = 8            # More context for prosodic continuity (Sesame uses 90s)

    # Generation parameters - SESAME AI LEVEL natural speech
    # CSM has TWO temperature parameters:
    # 1. temperature - controls backbone (codebook 0) = semantic/prosody planning
    # 2. depth_decoder_temperature - controls depth decoder (codebooks 1-31) = acoustic detail
    #
    # HuggingFace CSM documentation recommends:
    # - temperature=0.96 for prosodic variation (not too robotic, not too chaotic)
    # - depth_decoder_temperature=0.7 for stable, clean acoustic output
    #
    # This split is CRITICAL for Sesame-level quality:
    # - High backbone temp = expressive prosody, natural intonation
    # - Lower depth temp = clean, artifact-free audio
    #
    # RESEARCH INSIGHT: Temperature controls sampling diversity
    # Higher temp (1.0+) = more pitch variation, less monotonous
    # But too high = unstable, chaotic
    # Testing: 1.0 for backbone to increase prosodic variation
    temperature: float = 1.0          # Backbone: 1.0 for more prosody variation
    depth_decoder_temperature: float = 0.7  # Depth decoder: lower for acoustic stability
    topk: int = 60                    # Larger top-k for more diversity

    # Fine-tuned model - CORRECT training approach (val_loss 9.8858, 82% fewer clicks)
    # This model was trained with davidbrowne17/csm-streaming method
    custom_model_path: str = str(PROJECT_ROOT / "training/checkpoints/csm_maya_correct/best_model/model_merged.pt")

    # Voice prompt - MUST match training data!
    # Model fine-tuned on Expresso ex04 data, so voice prompt must be Expresso
    # Using Sesame prompt with Expresso model causes GIBBERISH output
    voice_prompt_path: str = str(VOICE_PROMPT_DIR / "maya_voice_prompt_expresso.pt")

    # Fast mode for short utterances (starters)
    fast_max_audio_ms: int = 3000     # 3 seconds max for starters
    fast_batch_size: int = 4          # Smaller batch = faster first chunk

    # Streaming parameters - PROSODY OPTIMIZED
    # davidbrowne17/csm-streaming: 20 frames initial chunk for prosody
    # Larger first chunk = better prosody establishment = more human-like
    # Trade-off: ~1.2s first audio instead of ~400ms, but MUCH better prosody
    first_chunk_frames: int = 15      # ~1.2s - gives decoder enough for prosody planning
    chunk_frames: int = 15            # ~1.2s - consistent chunk size
    crossfade_samples: int = 72       # 3ms at 24kHz - gentle Hann crossfade

    # Voice prompt limits
    max_voice_prompt_seconds: int = 30  # Max voice prompt duration

    # Audio normalization - Professional broadcast standards
    # -16 LUFS is the standard for voice agents and podcasts
    # -1 dBTP is the true peak ceiling to prevent clipping
    target_lufs: float = -16.0        # Professional loudness target for voice
    true_peak_limit: float = 0.89     # -1 dBTP (10^(-1/20) ≈ 0.89)
    peak_normalize_target: float = 0.5  # Legacy: -6dB peak normalization (fallback)


TTS = TTSConfig()


# =============================================================================
# FILLER CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class FillerConfig:
    """Filler and backchannel settings."""
    # Thinking fillers (play when user stops speaking)
    thinking_fillers: tuple = (
        "Hmm, let me think...",
        "Well...",
        "Let's see...",
        "So...",
        "Okay...",
    )

    # Backchannels (play while user speaks)
    backchannels: tuple = (
        "Mm-hmm.",
        "Yeah.",
        "Right.",
        "Uh-huh.",
        "I see.",
    )

    # Timing
    backchannel_interval_min: float = 4.0   # Minimum seconds between
    backchannel_interval_max: float = 7.0   # Maximum seconds between
    backchannel_volume: float = 0.25        # 25% volume

    # Filler duration target
    filler_duration_ms: int = 5000          # 5 second fillers


FILLER = FillerConfig()


# =============================================================================
# LATENCY TARGETS
# =============================================================================

@dataclass(frozen=True)
class LatencyConfig:
    """Latency targets and budgets."""
    # Component targets
    vad_target_ms: int = 50
    stt_target_ms: int = 400
    llm_ttft_target_ms: int = 200
    tts_first_chunk_ms: int = 200

    # Total budget
    total_target_ms: int = 850

    # Perceived latency (with filler trick)
    perceived_target_ms: int = 0  # Instant!

    # Timeout thresholds (log warning if exceeded)
    stt_timeout_ms: int = 2000     # 2 second STT timeout
    llm_timeout_ms: int = 3000     # 3 second LLM timeout
    tts_timeout_ms: int = 5000     # 5 second TTS timeout


LATENCY = LatencyConfig()


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def validate_config() -> bool:
    """Validate configuration at startup with detailed reporting."""
    import logging
    logger = logging.getLogger(__name__)

    errors = []
    warnings = []

    # Check paths
    if not PROJECT_ROOT.exists():
        errors.append(f"Project root not found: {PROJECT_ROOT}")
    else:
        logger.info(f"Project root: {PROJECT_ROOT}")

    if not CSM_ROOT.exists():
        errors.append(f"CSM not found: {CSM_ROOT}")
    else:
        logger.info(f"CSM root: {CSM_ROOT}")

    # Check voice prompt
    if VOICE_PROMPT_DIR.exists():
        voice_prompts = list(VOICE_PROMPT_DIR.glob("*.pt"))
        if not voice_prompts:
            warnings.append("No voice prompts found in assets/voice_prompt/")
        else:
            logger.info(f"Voice prompts: {[p.name for p in voice_prompts]}")
    else:
        warnings.append(f"Voice prompt directory not found: {VOICE_PROMPT_DIR}")

    # Check CUDA
    if not torch.cuda.is_available():
        errors.append("CUDA not available - GPU required")
    else:
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPUs available: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            logger.info(f"  GPU {i}: {props.name} ({vram_gb:.1f}GB)")
            if i == 0 and vram_gb < 16:
                warnings.append(f"GPU 0 has only {vram_gb:.1f}GB VRAM (16GB+ recommended)")

    # Check custom model path
    if not Path(TTS.custom_model_path).exists():
        warnings.append(f"Custom TTS model not found: {TTS.custom_model_path}")
        logger.info("Will use base CSM-1B model")
    else:
        logger.info(f"Custom TTS model: {TTS.custom_model_path}")

    # Report
    for warning in warnings:
        logger.warning(f"CONFIG WARNING: {warning}")

    if errors:
        for error in errors:
            logger.error(f"CONFIG ERROR: {error}")
        return False

    logger.info("Configuration validated successfully")
    return True


def get_gpu_index() -> int:
    """Get GPU index with environment variable support and safety checks."""
    env_gpu = os.environ.get("MAYA_GPU_INDEX")
    if env_gpu:
        try:
            idx = int(env_gpu)
            if idx < torch.cuda.device_count():
                return idx
        except ValueError:
            pass
    # Default to GPU 0 (safest)
    return 0


def get_vllm_url() -> str:
    """Get vLLM URL with environment variable support."""
    return os.environ.get("MAYA_VLLM_URL", "http://localhost:8001")
