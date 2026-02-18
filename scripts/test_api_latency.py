#!/usr/bin/env python3
"""
Test latency via the running Maya API.

The server has already warmed up, so this should show
the actual production latency.
"""

import asyncio
import websockets
import json
import time
import struct
import numpy as np


async def test_websocket_latency():
    """Test end-to-end latency via WebSocket."""
    print("=" * 60)
    print("MAYA API LATENCY TEST")
    print("=" * 60)

    uri = "ws://localhost:8000/ws/audio"

    try:
        async with websockets.connect(uri) as ws:
            print("\nConnected to Maya WebSocket")

            # Test 1: Send text message and measure time to first audio
            test_messages = [
                "hello",
                "how are you",
                "tell me about yourself",
            ]

            for msg in test_messages:
                print(f"\n--- Test: '{msg}' ---")

                # Send text (simulating STT result)
                start = time.time()

                # Send as text message to trigger response
                await ws.send(json.dumps({
                    "type": "text",
                    "text": msg
                }))

                # Wait for first audio response
                first_audio_time = None
                audio_chunks = 0
                total_samples = 0

                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=30.0)

                        if isinstance(response, bytes):
                            if first_audio_time is None:
                                first_audio_time = (time.time() - start) * 1000

                            # Count audio samples (16-bit PCM)
                            samples = len(response) // 2
                            total_samples += samples
                            audio_chunks += 1

                            # Check if this is likely the last chunk
                            # (small chunk or we've received enough audio)
                            if samples < 4096 or total_samples > 24000 * 5:
                                break

                        elif isinstance(response, str):
                            data = json.loads(response)
                            if data.get("type") == "turn_end":
                                break

                    except asyncio.TimeoutError:
                        print("  Timeout waiting for response")
                        break

                total_time = (time.time() - start) * 1000
                audio_duration = total_samples / 24000 if total_samples > 0 else 0

                print(f"  TTFA: {first_audio_time:.0f}ms" if first_audio_time else "  TTFA: N/A")
                print(f"  Total: {total_time:.0f}ms")
                print(f"  Audio: {audio_duration:.2f}s ({audio_chunks} chunks)")

                await asyncio.sleep(1.0)  # Brief pause between tests

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def test_http_endpoint():
    """Test HTTP text-to-speech endpoint if available."""
    import aiohttp

    print("\n" + "=" * 60)
    print("HTTP ENDPOINT TEST")
    print("=" * 60)

    url = "http://localhost:8000/api/tts"

    test_texts = [
        "hmm let me think",
        "oh thats interesting",
        "yeah i understand what youre saying",
    ]

    async with aiohttp.ClientSession() as session:
        for text in test_texts:
            print(f"\n--- Test: '{text}' ---")

            start = time.time()
            try:
                async with session.post(url, json={"text": text}) as resp:
                    if resp.status == 200:
                        # Read streaming response
                        first_byte_time = None
                        total_bytes = 0

                        async for chunk in resp.content.iter_any():
                            if first_byte_time is None:
                                first_byte_time = (time.time() - start) * 1000
                            total_bytes += len(chunk)

                        total_time = (time.time() - start) * 1000
                        audio_duration = (total_bytes // 2) / 24000

                        print(f"  TTFA: {first_byte_time:.0f}ms" if first_byte_time else "  TTFA: N/A")
                        print(f"  Total: {total_time:.0f}ms")
                        print(f"  Audio: {audio_duration:.2f}s")
                    else:
                        print(f"  Error: HTTP {resp.status}")
                        text = await resp.text()
                        print(f"  {text[:200]}")
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket_latency())
