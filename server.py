"""
Maya CSM-1B TTS Server

Drop-in replacement for EchoTTS in MayaClone.
Exposes POST /v1/audio/speech — same interface MayaClone already calls.
Returns streaming 16-bit PCM audio at 24kHz.
"""

import os
import sys
import logging
from pathlib import Path

# Load .env before any maya imports so env vars are set at module-import time
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Maya CSM-1B TTS Server")

# Global TTS engine (loaded once at startup)
_tts_engine = None


class TTSRequest(BaseModel):
    input: str
    voice: str = "maya"
    stream: bool = True
    response_format: str = "pcm"
    speed: float = 1.0
    # EchoTTS-specific params — accepted but ignored
    block_sizes: list = []
    num_steps: list = []
    cfg_scale_text: float = 2.5
    cfg_scale_speaker: float = 5.0
    truncation_factor: float = 0.9
    speaker_kv_scale: float = 1.0
    seed: int = 0


@app.on_event("startup")
async def startup():
    global _tts_engine
    logger.info("Loading CSM-1B TTS engine…")
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    _tts_engine = RealStreamingTTSEngine()
    _tts_engine.initialize()
    logger.info("CSM-1B TTS engine ready")


@app.get("/health")
async def health():
    ready = _tts_engine is not None and _tts_engine.is_initialized
    return {
        "status": "healthy" if ready else "loading",
        "initialized": ready,
        "startup_warmup_complete": ready,
    }


@app.post("/v1/audio/speech")
async def speech(req: TTSRequest):
    if _tts_engine is None or not _tts_engine.is_initialized:
        return JSONResponse({"error": "TTS engine not ready"}, status_code=503)

    text = req.input
    # MayaClone prepends [S1] — strip it before sending to CSM
    if text.startswith("[S1]"):
        text = text[4:].strip()
    if not text:
        return JSONResponse({"error": "empty input"}, status_code=400)

    SAMPLE_RATE = 24000

    def generate_pcm():
        for chunk in _tts_engine.generate_stream(text):
            audio_np = chunk.cpu().float().numpy()
            audio_np = np.clip(audio_np, -1.0, 1.0)
            pcm = (audio_np * 32767).astype(np.int16)
            yield pcm.tobytes()

    return StreamingResponse(
        generate_pcm(),
        media_type="application/octet-stream",
        headers={"X-Audio-Sample-Rate": str(SAMPLE_RATE)},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
