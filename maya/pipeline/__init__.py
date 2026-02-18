"""
Maya Pipeline - Main orchestration.

PRODUCTION (Sesame AI Level):
- ProductionPipeline: < 200ms first audio, true streaming STT, vLLM, 2-frame TTS

Legacy (deprecated):
- MayaPipeline: Original pipeline with filler support
- FastMayaPipeline: Lightning fast streaming pipeline
- SeamlessMayaPipeline: Direct response (no fillers)
- SmartMayaPipeline: Filler + overlap strategy
- StreamingMayaPipeline: TRUE STREAMING - First audio in ~600ms
"""

# PRODUCTION - Use this for Sesame AI level performance
from .production_pipeline import ProductionPipeline

# Legacy pipelines (kept for comparison/fallback)
from .orchestrator import MayaPipeline
from .fast_orchestrator import FastMayaPipeline
from .seamless_orchestrator import SeamlessMayaPipeline
from .smart_orchestrator import SmartMayaPipeline
from .streaming_orchestrator import StreamingMayaPipeline

__all__ = [
    # Production (recommended)
    "ProductionPipeline",
    # Legacy
    "MayaPipeline", "FastMayaPipeline", "SeamlessMayaPipeline",
    "SmartMayaPipeline", "StreamingMayaPipeline"
]
