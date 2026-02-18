#!/usr/bin/env python3
"""Test the vLLM-based LLM engine."""

import sys
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

from maya.engine.llm_vllm import VLLMEngine


def main():
    print("=" * 60)
    print("TESTING VLLMEngine")
    print("=" * 60)

    engine = VLLMEngine()
    engine.initialize()

    test_inputs = [
        "hey how are you",
        "tell me about yourself",
        "thats really interesting",
        "what do you think about AI",
        "thanks for chatting",
    ]

    print("\n" + "=" * 60)
    print("CONVERSATION TEST")
    print("=" * 60)

    times = []
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")

        start = time.time()
        response = engine.generate(user_input)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        print(f"Maya ({elapsed:.0f}ms): {response}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average latency: {sum(times)/len(times):.0f}ms")
    print(f"Min: {min(times):.0f}ms")
    print(f"Max: {max(times):.0f}ms")
    print(f"Target: <100ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
