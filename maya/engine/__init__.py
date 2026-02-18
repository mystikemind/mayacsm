"""
Maya Engines - Core AI components.

PRODUCTION (Sesame AI Level):
- TrueStreamingSTTEngine: Real-time partial hypotheses during speech (~25ms final)
- VLLMEngine: vLLM with Unix socket support (~65ms)
- RealStreamingTTSEngine: 2-frame first chunk (~105ms)

Core:
- VAD: Voice Activity Detection (Silero + Smart Turn Detection)
- STT: Speech-to-Text (Whisper via Docker)
- LLM: Language Model (Llama via vLLM)
- TTS: Text-to-Speech (CSM-1B Streaming)

Advanced:
- StreamingSTT: Prefetch-enabled STT
- AudioEnhancer: Noise reduction + echo detection
- TurnDetector: Prosody-based turn detection
"""

from .vad import VADEngine

# PRODUCTION - Sesame AI level components
from .stt_true_streaming import TrueStreamingSTTEngine, VADStreamingSTT
from .llm_vllm import VLLMEngine
from .tts_streaming_real import RealStreamingTTSEngine

# Core components
from .stt import STTEngine
from .stt_faster import FasterSTTEngine
from .stt_local import LocalSTTEngine
from .llm import LLMEngine
from .llm_optimized import OptimizedLLMEngine
from .tts import TTSEngine
from .tts_streaming import StreamingTTSEngine
from .tts_compiled import CompiledTTSEngine
from .starter_cache import StarterCache, get_starter_cache

# Advanced components
from .stt_streaming import StreamingSTTEngine
from .audio_enhancer import AudioEnhancer
from .turn_detector import ProsodyTurnDetector

__all__ = [
    # Production (Sesame level)
    "TrueStreamingSTTEngine", "VADStreamingSTT", "VLLMEngine", "RealStreamingTTSEngine",
    # Core
    "VADEngine", "STTEngine", "FasterSTTEngine", "LocalSTTEngine",
    "LLMEngine", "OptimizedLLMEngine", "TTSEngine", "StreamingTTSEngine",
    "CompiledTTSEngine", "StarterCache", "get_starter_cache",
    # Advanced
    "StreamingSTTEngine", "AudioEnhancer", "ProsodyTurnDetector",
]
