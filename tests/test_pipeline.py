#!/usr/bin/env python3
"""
Test the full Maya pipeline.
"""

import torch
import asyncio
import sys
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')


async def test_pipeline():
    """Test the complete pipeline."""
    print("="*60)
    print("TESTING MAYA PIPELINE")
    print("="*60)

    from maya.pipeline import MayaPipeline
    import torchaudio

    # Track output audio
    output_audio = []

    async def capture_audio(audio):
        output_audio.append(audio)
        duration = len(audio) / 24000
        print(f"  → Received {duration:.2f}s audio")

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = MayaPipeline()
    pipeline.set_audio_callback(capture_audio)
    await pipeline.initialize()

    # Test greeting
    print("\n2. Testing greeting...")
    await pipeline.start_conversation()
    print(f"  → Greeting complete, {len(output_audio)} audio chunks")

    # Simulate user speech with a test audio file
    print("\n3. Simulating user speech...")

    # Create a simple test: just send silence to trigger VAD silence detection
    # In real use, this would be actual recorded speech

    # First, simulate speech start
    speech_audio = torch.randn(24000 * 2) * 0.3  # 2 seconds of noise (simulates speech)
    silence_audio = torch.zeros(24000 * 1)  # 1 second of silence

    # Process in chunks
    chunk_size = 2400  # 100ms chunks

    print("  → Sending 'speech' audio...")
    for i in range(0, len(speech_audio), chunk_size):
        chunk = speech_audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        await pipeline.process_audio_chunk(chunk)
        await asyncio.sleep(0.01)  # Small delay to simulate real-time

    print("  → Sending silence (end of speech)...")
    for i in range(0, len(silence_audio), chunk_size):
        chunk = silence_audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        await pipeline.process_audio_chunk(chunk)
        await asyncio.sleep(0.01)

    # Wait for response
    print("  → Waiting for response...")
    await asyncio.sleep(15)  # Give time for full response

    # Save output
    if output_audio:
        combined = torch.cat(output_audio)
        output_path = "/home/ec2-user/SageMaker/project_maya/tests/pipeline_output.wav"
        torchaudio.save(output_path, combined.unsqueeze(0).cpu(), 24000)
        print(f"\n4. Saved {len(combined)/24000:.2f}s audio to {output_path}")

    # Print stats
    print("\n5. Pipeline stats:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
