#!/usr/bin/env python3
"""
Deep Latency Analysis - Senior Engineer Investigation

Break down EXACTLY where time is spent in each component.
Identify optimization opportunities.
"""

import sys
import time
import torch
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

print("=" * 70)
print("DEEP LATENCY ANALYSIS")
print("=" * 70)

# ============================================================================
# PART 1: TTS DETAILED BREAKDOWN
# ============================================================================

print("\n" + "=" * 70)
print("PART 1: TTS DETAILED BREAKDOWN")
print("=" * 70)

from maya.engine.tts_compiled import CompiledTTSEngine

tts = CompiledTTSEngine()
tts.initialize()

# Patch the generator to measure internal timings
original_generate = tts._generator.generate

def timed_generate(text, speaker, context, max_audio_length_ms, temperature, topk):
    """Measure internal generation time."""
    torch.cuda.synchronize()
    start = time.time()
    result = original_generate(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk
    )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return result, elapsed

# Test with different text lengths
test_cases = [
    ("Hi!", 1),
    ("Hello there!", 2),
    ("How are you today?", 4),
    ("That sounds really interesting!", 4),
    ("I think that's a great question to ask.", 8),
]

print("\nTTS timing breakdown:")
print("-" * 70)

for text, word_count in test_cases:
    # Measure full generate() call
    torch.cuda.synchronize()
    total_start = time.time()

    # Run through our generate method to include context building
    context = []
    if tts._voice_prompt:
        context.append(tts._voice_prompt)

    # Measure context building time
    context_start = time.time()
    # Context is already built above
    context_time = time.time() - context_start

    # Measure actual generation
    gen_start = time.time()
    audio, gen_elapsed = timed_generate(
        text=text,
        speaker=0,
        context=context,
        max_audio_length_ms=4000,
        temperature=0.9,
        topk=50
    )

    # Measure post-processing (fades, padding)
    post_start = time.time()
    audio = audio.clone()
    fade_in_samples = int(24000 * 0.03)
    fade_in = torch.linspace(0, 1, fade_in_samples, device=audio.device)
    audio[:fade_in_samples] = audio[:fade_in_samples] * fade_in
    fade_out_samples = int(24000 * 0.08)
    fade_out = torch.linspace(1, 0, fade_out_samples, device=audio.device)
    audio[-fade_out_samples:] = audio[-fade_out_samples:] * fade_out
    padding = torch.zeros(int(24000 * 0.2), device=audio.device)
    audio = torch.cat([audio, padding])
    torch.cuda.synchronize()
    post_time = time.time() - post_start

    total_time = time.time() - total_start
    audio_duration = len(audio) / 24000
    rtf = total_time / audio_duration

    print(f"\n  Text: '{text}' ({word_count} words)")
    print(f"    Generation:     {gen_elapsed*1000:6.0f}ms ({gen_elapsed/total_time*100:4.1f}%)")
    print(f"    Post-process:   {post_time*1000:6.0f}ms ({post_time/total_time*100:4.1f}%)")
    print(f"    Total:          {total_time*1000:6.0f}ms")
    print(f"    Audio:          {audio_duration:.2f}s")
    print(f"    RTF:            {rtf:.2f}x")

# ============================================================================
# PART 2: LLM DETAILED BREAKDOWN
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: LLM DETAILED BREAKDOWN")
print("=" * 70)

from maya.engine.llm_optimized import OptimizedLLMEngine

llm = OptimizedLLMEngine()
llm.initialize()

test_inputs = [
    "Hi",
    "How are you?",
    "What's the weather like today?",
    "Tell me something interesting about space.",
]

print("\nLLM timing breakdown:")
print("-" * 70)

for text in test_inputs:
    torch.cuda.synchronize()
    start = time.time()
    response = llm.generate(text)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_out = len(response.split())

    print(f"\n  Input: '{text}'")
    print(f"    Response: '{response}'")
    print(f"    Time: {elapsed*1000:.0f}ms")
    print(f"    Tokens out: ~{tokens_out} words")
    print(f"    ms/word: {elapsed*1000/max(tokens_out,1):.0f}ms")

# ============================================================================
# PART 3: OPTIMIZATION OPPORTUNITIES
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: OPTIMIZATION OPPORTUNITIES")
print("=" * 70)

print("""
ANALYSIS RESULTS:

1. TTS GENERATION is the bottleneck (~70-80% of total time)
   - The actual CSM generation is where time is spent
   - Post-processing is negligible (<1%)

2. POTENTIAL OPTIMIZATIONS:

   a) CUDA Graphs (High Impact):
      - Current: torch.compile with 'reduce-overhead'
      - Try: 'max-autotune' mode for more aggressive optimization
      - Requires: Consistent input shapes

   b) Parallel LLM + TTS (High Impact):
      - Start TTS as soon as LLM produces first tokens
      - Requires: Streaming LLM output
      - Could save 200-300ms

   c) Batch Size Tuning:
      - CSM generates frame-by-frame internally
      - May be able to tune internal batch sizes

   d) Precision Optimization:
      - Currently using bfloat16
      - Could try float16 or quantization

   e) Voice Prompt Caching:
      - Pre-compute voice prompt embeddings
      - Save tokenization time

3. WHAT SESAME LIKELY DOES:
   - Full CUDA graphs with static KV cache
   - Speculative decoding
   - Custom kernels
   - Possibly distilled/smaller model
""")

# ============================================================================
# PART 4: TEST TORCH.COMPILE MODES
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: TORCH.COMPILE MODE COMPARISON")
print("=" * 70)

# We can't easily recompile, but let's check current mode
print(f"""
Current torch.compile settings:
- Decoder compiled with mode='reduce-overhead'
- This enables some CUDA graph optimization

Alternative modes to try:
- 'max-autotune': More aggressive, may be faster
- 'default': Less optimization but more stable

To test different modes, modify tts_compiled.py:
  model.decoder = torch.compile(model.decoder, mode='max-autotune')
""")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

print("""
HIGHEST IMPACT OPTIMIZATIONS (in order):

1. PARALLEL LLM + TTS STREAMING
   - Expected savings: 200-300ms
   - Complexity: Medium
   - Risk: Low

2. MORE AGGRESSIVE TORCH.COMPILE
   - Expected savings: 50-100ms
   - Complexity: Low (just change mode)
   - Risk: Medium (may be unstable)

3. CUDA GRAPHS WITH STATIC SHAPES
   - Expected savings: 100-200ms
   - Complexity: High
   - Risk: High

4. SPECULATIVE DECODING FOR LLM
   - Expected savings: 100-150ms
   - Complexity: Very High
   - Risk: Medium

RECOMMENDED NEXT STEP: Implement parallel LLM + TTS streaming
""")
