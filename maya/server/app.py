"""
Maya WebSocket Server - Real-time voice communication.

PRODUCTION SERVER - Uses the unified ProductionPipeline for Sesame AI level latency.

Handles:
- Bidirectional audio streaming via WebSocket
- Multiple concurrent connections
- Clean reconnection handling
- < 200ms first audio latency (Sesame AI level)
"""

import torch
import asyncio
import logging
import json
import struct
import numpy as np
import os
from enum import Enum, auto
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict
import uvicorn

from ..pipeline.production_pipeline import ProductionPipeline
from ..config import AUDIO, PROJECT_ROOT, validate_config

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Available pipeline modes."""
    PRODUCTION = auto()  # ProductionPipeline - Sesame AI level (<200ms)
    LEGACY_SEAMLESS = auto()  # Old SeamlessMayaPipeline (kept for comparison)


# Global pipeline instance (shared across connections)
pipeline = None

# PRODUCTION MODE (recommended)
# Set MAYA_PIPELINE_MODE environment variable to override
PIPELINE_MODE = PipelineMode.PRODUCTION


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Maya Voice AI",
        description="Sesame AI Level Conversational Voice AI - <200ms first audio",
        version="5.0.0"  # v5: Production pipeline
    )

    @app.on_event("startup")
    async def startup():
        """Initialize pipeline on startup."""
        global pipeline

        # Validate configuration
        logger.info("Validating configuration...")
        if not validate_config():
            logger.error("Configuration validation failed - check logs above")
            # Continue anyway for development, but log warning

        # Check for mode override from environment
        mode_override = os.environ.get("MAYA_PIPELINE_MODE", "").upper()
        current_mode = PIPELINE_MODE

        if mode_override == "LEGACY":
            current_mode = PipelineMode.LEGACY_SEAMLESS
            logger.info("Mode override: LEGACY (from environment)")

        # Create pipeline based on mode
        if current_mode == PipelineMode.PRODUCTION:
            logger.info("=" * 60)
            logger.info("PRODUCTION PIPELINE - Sesame AI Level")
            logger.info("Target: < 200ms first audio latency")
            logger.info("=" * 60)
            pipeline = ProductionPipeline()
        else:
            # Legacy mode for comparison/fallback
            logger.info("Using LEGACY seamless pipeline")
            from ..pipeline import SeamlessMayaPipeline
            pipeline = SeamlessMayaPipeline()

        await pipeline.initialize()
        logger.info("Maya server started - Ready for connections")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main UI."""
        return get_ui_html()

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "initialized": pipeline.is_initialized if pipeline else False
        }

    @app.get("/stats")
    async def stats():
        """Get pipeline statistics."""
        if pipeline:
            return pipeline.get_stats()
        return {"error": "Pipeline not initialized"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for voice communication."""
        await websocket.accept()
        logger.info("WebSocket connection established")

        # Connection state flag
        is_connected = True

        try:
            # Set up audio callback with connection check
            async def send_audio(audio: torch.Tensor):
                """Send audio to client in chunks to prevent WebSocket timeout."""
                nonlocal is_connected
                if not is_connected:
                    logger.debug("Skipping audio send - not connected")
                    return

                # Convert to numpy
                audio_np = audio.cpu().numpy()
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)

                # Send complete audio as single message for smooth playback
                # Chunking with delays causes choppy audio
                total_samples = len(audio_np)
                try:
                    await websocket.send_bytes(audio_np.tobytes())
                    logger.info(f"Sent audio ({total_samples} samples, {total_samples/24000:.1f}s)")
                except Exception as e:
                    logger.error(f"Error sending audio: {e}")
                    is_connected = False

            pipeline.set_audio_callback(send_audio)

            # Send initial greeting
            logger.info("Starting conversation (generating greeting)...")
            await pipeline.start_conversation()
            logger.info("Greeting sent, waiting for user audio...")

            # Process incoming audio
            chunks_received = 0
            while is_connected:
                try:
                    # Receive audio data with timeout
                    data = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=60.0  # 60 second timeout
                    )
                    chunks_received += 1

                    # Convert to tensor
                    audio_np = np.frombuffer(data, dtype=np.float32)
                    audio_tensor = torch.from_numpy(audio_np.copy())

                    # Log every 50th chunk to see audio is being received
                    if chunks_received % 50 == 0:
                        max_amp = audio_tensor.abs().max().item()
                        logger.info(f"Received {chunks_received} audio chunks, max amplitude: {max_amp:.4f}")

                    # Process
                    await pipeline.process_audio_chunk(audio_tensor)

                except asyncio.TimeoutError:
                    logger.info("Connection timeout - no data received")
                    break
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected by client")
                    break
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    continue

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            is_connected = False
            pipeline.set_audio_callback(None)  # Remove callback
            await pipeline.reset()
            logger.info("WebSocket connection closed")

    return app


def get_ui_html() -> str:
    """Generate the web UI HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maya - Voice AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: white;
        }

        .container {
            text-align: center;
            padding: 40px;
            max-width: 600px;
        }

        h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #e94560, #ff6b9d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            font-size: 1.2rem;
            color: #8892b0;
            margin-bottom: 40px;
        }

        .visualizer {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            height: 100px;
            margin-bottom: 40px;
        }

        .bar {
            width: 6px;
            background: linear-gradient(180deg, #e94560, #ff6b9d);
            border-radius: 3px;
            transition: height 0.1s ease;
        }

        .call-button {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            font-size: 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 30px;
            z-index: 100;
            position: relative;
            -webkit-tap-highlight-color: transparent;
            touch-action: manipulation;
        }

        .call-button.start {
            background: linear-gradient(135deg, #00d9a5, #00b894);
            color: white;
            box-shadow: 0 10px 30px rgba(0, 217, 165, 0.3);
        }

        .call-button.start:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 40px rgba(0, 217, 165, 0.4);
        }

        .call-button.stop {
            background: linear-gradient(135deg, #e94560, #ff6b9d);
            color: white;
            box-shadow: 0 10px 30px rgba(233, 69, 96, 0.3);
            animation: pulse 2s infinite;
        }

        .call-button.stop:hover {
            transform: scale(1.05);
        }

        @keyframes pulse {
            0%, 100% { box-shadow: 0 10px 30px rgba(233, 69, 96, 0.3); }
            50% { box-shadow: 0 10px 50px rgba(233, 69, 96, 0.5); }
        }

        .status {
            font-size: 1rem;
            color: #8892b0;
            margin-bottom: 20px;
            min-height: 24px;
        }

        .status.connected {
            color: #00d9a5;
        }

        .status.error {
            color: #e94560;
        }

        .transcript {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            text-align: left;
            max-height: 200px;
            overflow-y: auto;
        }

        .transcript-line {
            margin-bottom: 10px;
            font-size: 0.95rem;
        }

        .transcript-line.user {
            color: #8892b0;
        }

        .transcript-line.maya {
            color: #e94560;
        }

        .transcript-line span {
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Maya</h1>
        <p class="subtitle">The World's Best Conversational Voice AI</p>

        <div class="visualizer" id="visualizer">
            <!-- Bars will be added by JS -->
        </div>

        <button class="call-button start" id="callButton">
            Start
        </button>

        <p class="status" id="status">Click Start to begin</p>

        <div class="transcript" id="transcript">
            <div class="transcript-line maya"><span>Maya:</span> Ready to talk!</div>
        </div>
    </div>

    <script>
        // Configuration
        var SAMPLE_RATE = 24000;
        var CHUNK_SIZE = 2400;

        // State
        var isConnected = false;
        var websocket = null;
        var audioContext = null;
        var mediaStream = null;
        var callButton = null;
        var statusEl = null;
        var visualizer = null;
        var bars = null;

        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM Ready');

            // Get UI elements
            callButton = document.getElementById('callButton');
            statusEl = document.getElementById('status');
            visualizer = document.getElementById('visualizer');

            console.log('callButton:', callButton);
            console.log('statusEl:', statusEl);

            // Initialize visualizer bars
            if (visualizer) {
                for (var i = 0; i < 20; i++) {
                    var bar = document.createElement('div');
                    bar.className = 'bar';
                    bar.style.height = '10px';
                    visualizer.appendChild(bar);
                }
                bars = document.querySelectorAll('.bar');
            }

            // Set up button click
            if (callButton) {
                callButton.onclick = function(e) {
                    e.preventDefault();
                    console.log('Button clicked!');
                    handleButtonClick();
                };
                console.log('Button handler attached');
            } else {
                console.error('Button not found!');
            }
        });

        // Handle button click
        function handleButtonClick() {
            console.log('handleButtonClick, isConnected:', isConnected);
            setStatus('Starting...', '');
            if (isConnected) {
                disconnect();
            } else {
                connect();
            }
        }

        // Connect to server
        async function connect() {
            console.log('connect() called');
            setStatus('Requesting microphone...', '');

            // Get microphone FIRST (this triggers permission dialog)
            console.log('Calling getUserMedia...');
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: SAMPLE_RATE,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });
            console.log('Microphone access granted!');
            setStatus('Microphone ready, connecting...', '');

            // Initialize audio context AFTER getting mic (needs user gesture)
            audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
            console.log('AudioContext created, state:', audioContext.state);

            // IMPORTANT: Resume AudioContext (required by Chrome)
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
                console.log('AudioContext resumed');
            }

            // Create WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            console.log('Creating WebSocket to:', protocol + '//' + window.location.host + '/ws');
            websocket = new WebSocket(protocol + '//' + window.location.host + '/ws');
            websocket.binaryType = 'arraybuffer';

            websocket.onopen = function() {
                setStatus('Connected - Speak to Maya!', 'connected');
                console.log('WebSocket connected, starting audio capture...');
                isConnected = true;
                callButton.textContent = 'End';
                callButton.className = 'call-button stop';
                resetAudioBuffer();  // Reset audio buffer for new session
                startAudioCapture();
            };

            websocket.onmessage = function(event) {
                if (event.data instanceof ArrayBuffer) {
                    var audioData = new Float32Array(event.data);
                    playAudio(audioData);
                }
            };

            websocket.onclose = function() {
                console.log('WebSocket closed');
                disconnect();
            };

            websocket.onerror = function(error) {
                setStatus('Connection error', 'error');
                console.error('WebSocket error:', error);
            };
        }

        // Disconnect
        function disconnect() {
            if (websocket) {
                websocket.close();
                websocket = null;
            }

            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }

            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }

            isConnected = false;
            callButton.textContent = 'Start';
            callButton.className = 'call-button start';
            setStatus('Disconnected', '');
            resetVisualizer();
        }

        // Start audio capture
        let audioChunksSent = 0;
        let scriptProcessor = null;

        async function startAudioCapture() {
            try {
                console.log('Starting audio capture...');
                console.log('AudioContext state:', audioContext.state);
                console.log('MediaStream active:', mediaStream.active);
                console.log('MediaStream tracks:', mediaStream.getAudioTracks().length);

                // Check if tracks are enabled
                const tracks = mediaStream.getAudioTracks();
                tracks.forEach((track, i) => {
                    console.log(`Track ${i}: enabled=${track.enabled}, readyState=${track.readyState}`);
                });

                const source = audioContext.createMediaStreamSource(mediaStream);
                console.log('MediaStreamSource created');

                // Use ScriptProcessor with smallest stable buffer for low latency
                // 1024 samples @ 24kHz = ~43ms latency (vs 85ms with 2048)
                scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);
                console.log('ScriptProcessor created with bufferSize: 1024 (low latency mode)');

                scriptProcessor.onaudioprocess = (e) => {
                    if (!isConnected || !websocket || websocket.readyState !== WebSocket.OPEN) {
                        return;
                    }

                    const inputData = e.inputBuffer.getChannelData(0);

                    // Copy to new Float32Array to avoid issues
                    const audioData = new Float32Array(inputData.length);
                    audioData.set(inputData);

                    // Check if there's actual audio
                    let maxVal = 0;
                    for (let i = 0; i < audioData.length; i++) {
                        const abs = Math.abs(audioData[i]);
                        if (abs > maxVal) maxVal = abs;
                    }

                    // Send to server
                    websocket.send(audioData.buffer);
                    audioChunksSent++;

                    // Update status with audio level
                    if (audioChunksSent % 20 === 0) {
                        console.log(`Sent ${audioChunksSent} chunks, level: ${maxVal.toFixed(4)}`);
                        if (maxVal > 0.01) {
                            setStatus(`Listening... (level: ${(maxVal * 100).toFixed(0)}%)`, 'connected');
                        } else {
                            setStatus('Listening... (speak louder)', 'connected');
                        }
                    }

                    // Update visualizer
                    updateVisualizer(audioData);
                };

                // Connect the audio graph
                source.connect(scriptProcessor);
                scriptProcessor.connect(audioContext.destination);

                console.log('Audio capture pipeline connected!');
                setStatus('Listening... speak now!', 'connected');

            } catch (error) {
                console.error('Audio capture error:', error);
                setStatus('Audio error: ' + error.message, 'error');
            }
        }

        // Queue-based audio playback for streaming TTS chunks
        // Chunks are scheduled seamlessly using AudioContext timing
        var currentSource = null;
        var nextPlayTime = 0;
        var scheduledSources = [];

        function playAudio(audioData) {
            if (!audioContext) return;

            var buffer = audioContext.createBuffer(1, audioData.length, SAMPLE_RATE);
            buffer.getChannelData(0).set(audioData);

            var source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);

            // Schedule seamlessly after previous chunk
            var now = audioContext.currentTime;
            var startTime = Math.max(now + 0.005, nextPlayTime);
            source.start(startTime);
            nextPlayTime = startTime + buffer.duration;

            scheduledSources.push(source);
            currentSource = source;

            source.onended = function() {
                var idx = scheduledSources.indexOf(source);
                if (idx > -1) scheduledSources.splice(idx, 1);
                if (currentSource === source) currentSource = null;
            };

            console.log('Queued ' + (audioData.length / SAMPLE_RATE).toFixed(2) + 's at T+' + (startTime - now).toFixed(3));
            updateVisualizerFromBuffer(audioData);
        }

        // Stop all queued audio (for barge-in or new session)
        function stopPlayback() {
            scheduledSources.forEach(function(source) {
                try { source.stop(); } catch(e) {}
            });
            scheduledSources = [];
            currentSource = null;
            nextPlayTime = 0;
        }

        // Reset audio state for new session
        function resetAudioBuffer() {
            stopPlayback();
        }

        // Update visualizer from input
        function updateVisualizer(audioData) {
            const chunkSize = Math.floor(audioData.length / bars.length);
            for (let i = 0; i < bars.length; i++) {
                let sum = 0;
                for (let j = 0; j < chunkSize; j++) {
                    sum += Math.abs(audioData[i * chunkSize + j]);
                }
                const average = sum / chunkSize;
                const height = Math.max(10, Math.min(80, average * 500));
                bars[i].style.height = height + 'px';
            }
        }

        // Update visualizer from playback buffer
        function updateVisualizerFromBuffer(audioData) {
            const chunkSize = Math.floor(audioData.length / bars.length);
            for (let i = 0; i < bars.length; i++) {
                let sum = 0;
                for (let j = 0; j < chunkSize; j++) {
                    sum += Math.abs(audioData[i * chunkSize + j]);
                }
                const average = sum / chunkSize;
                const height = Math.max(10, Math.min(80, average * 300));
                bars[i].style.height = height + 'px';
                bars[i].style.background = 'linear-gradient(180deg, #e94560, #ff6b9d)';
            }

            // Reset after a moment
            setTimeout(resetVisualizer, 200);
        }

        // Reset visualizer
        function resetVisualizer() {
            bars.forEach(bar => {
                bar.style.height = '10px';
            });
        }

        // Set status message
        function setStatus(message, className) {
            if (statusEl) {
                statusEl.textContent = message;
                statusEl.className = 'status ' + className;
            }
            console.log('Status:', message);
        }

        // Add transcript line
        function addTranscript(speaker, text) {
            const line = document.createElement('div');
            line.className = 'transcript-line ' + speaker.toLowerCase();
            line.innerHTML = `<span>${speaker}:</span> ${text}`;
            transcript.appendChild(line);
            transcript.scrollTop = transcript.scrollHeight;
        }
    </script>
</body>
</html>'''


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the Maya server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
