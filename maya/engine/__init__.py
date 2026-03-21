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

# TTS-only server — only import what's needed
from .tts_streaming_real import RealStreamingTTSEngine

__all__ = ["RealStreamingTTSEngine"]
