#!/usr/bin/env python3
"""
Test improved Maya personality with latency check.
"""

import sys
import time
import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

print("=" * 70)
print("TESTING IMPROVED MAYA PERSONALITY")
print("=" * 70)

# Initialize LLM
print("\n[1] Initializing LLM...")
from maya.engine.llm_optimized import OptimizedLLMEngine

llm = OptimizedLLMEngine()
llm.initialize()

# Test conversations
test_inputs = [
    "Hello!",
    "How are you doing today?",
    "I'm feeling a bit sad",
    "I got a promotion at work!",
    "What's your favorite thing to do?",
    "Tell me something interesting",
    "I had a really long day",
    "Do you like music?",
    "I'm learning to cook",
    "What do you think about that?",
]

print("\n[2] Testing personality responses...")
print("-" * 70)

results = []

for user_input in test_inputs:
    start = time.time()
    response = llm.generate(user_input)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000

    word_count = len(response.split())
    results.append({
        'input': user_input,
        'response': response,
        'words': word_count,
        'time_ms': elapsed
    })

    status = "✅" if word_count <= 10 else "⚠️"
    print(f"\n  User: \"{user_input}\"")
    print(f"  Maya: \"{response}\"")
    print(f"  {status} {word_count} words, {elapsed:.0f}ms")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

avg_words = sum(r['words'] for r in results) / len(results)
avg_time = sum(r['time_ms'] for r in results) / len(results)
max_words = max(r['words'] for r in results)

print(f"""
  Average words: {avg_words:.1f}
  Max words: {max_words}
  Average LLM time: {avg_time:.0f}ms

  Target: 5-8 words, <200ms LLM time
  Status: {'✅ GOOD' if avg_words <= 10 and avg_time < 300 else '⚠️ NEEDS TUNING'}
""")

# Check if any responses are too long
long_responses = [r for r in results if r['words'] > 10]
if long_responses:
    print("  Responses that are too long:")
    for r in long_responses:
        print(f"    - \"{r['response']}\" ({r['words']} words)")

print("=" * 70)
