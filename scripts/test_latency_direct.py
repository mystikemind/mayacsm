#!/usr/bin/env python3
"""
Direct latency test - No Gradio, no WebSocket, just raw pipeline timing.
This is how a real audio engineer would test.
"""
import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import time
import asyncio

async def test_direct_pipeline():
    print("=" * 60)
    print("DIRECT PIPELINE LATENCY TEST")
    print("No Gradio, no WebSocket - raw performance")
    print("=" * 60)
    
    from maya.pipeline import SmartMayaPipeline
    
    pipeline = SmartMayaPipeline()
    
    # Capture audio output
    audio_chunks = []
    first_audio_time = None
    test_start = None
    
    async def audio_callback(audio):
        nonlocal first_audio_time
        if first_audio_time is None:
            first_audio_time = time.time()
        audio_chunks.append(audio)
    
    pipeline.set_audio_callback(audio_callback)
    
    print("\nInitializing pipeline...")
    await pipeline.initialize()
    
    print(f"\nFillers cached: {len(pipeline._filler_cache)}")
    for text, audio in pipeline._filler_cache.items():
        duration = len(audio) / 24000
        print(f"  - '{text}': {duration:.2f}s")
    
    # Test with simulated user audio (speech-like)
    print("\n" + "=" * 60)
    print("TEST: Simulating user saying 'Hello Maya'")
    print("=" * 60)
    
    # Generate fake speech audio (2 seconds of voice-like signal)
    duration = 2.0
    sample_rate = 24000
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Mix of frequencies to simulate speech
    user_audio = (
        0.3 * torch.sin(2 * 3.14159 * 200 * t) +  # Fundamental
        0.2 * torch.sin(2 * 3.14159 * 400 * t) +  # Harmonic
        0.1 * torch.sin(2 * 3.14159 * 800 * t) +  # Higher harmonic
        0.1 * torch.randn_like(t)  # Noise
    )
    user_audio = user_audio * 0.5  # Normalize
    
    # Manually trigger response (bypass VAD)
    pipeline._user_audio_buffer = [user_audio]
    pipeline.conversation.user_started_speaking()
    
    test_start = time.time()
    audio_chunks.clear()
    first_audio_time = None
    
    # Manually call speech end handler
    await pipeline._handle_speech_end()
    
    total_time = time.time() - test_start
    
    if first_audio_time:
        time_to_first_audio = first_audio_time - test_start
        total_samples = sum(len(c) for c in audio_chunks)
        audio_duration = total_samples / sample_rate
        
        print(f"\n--- RESULTS ---")
        print(f"Time to FIRST audio: {time_to_first_audio*1000:.0f}ms")
        print(f"  (This includes filler)")
        print(f"Total response time: {total_time*1000:.0f}ms")
        print(f"Audio generated: {audio_duration:.1f}s")
        print(f"Audio chunks: {len(audio_chunks)}")
        
        # Calculate when actual response started (after filler)
        filler_duration = 0
        for text, audio in pipeline._filler_cache.items():
            filler_duration = max(filler_duration, len(audio) / 24000)
        
        print(f"\nFiller duration: ~{filler_duration:.2f}s")
        print(f"Time from filler end to response: ~{(time_to_first_audio - filler_duration)*1000:.0f}ms (estimated)")
    else:
        print("No audio generated!")
    
    # Show metrics
    stats = pipeline.get_stats()
    print(f"\nPipeline stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_direct_pipeline())
