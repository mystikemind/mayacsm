#!/usr/bin/env python3
"""Test the new FastLLMEngine."""

import sys
import time
import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

from maya.engine.llm_fast import FastLLMEngine


def main():
    print("=" * 60)
    print("TESTING FastLLMEngine")
    print("=" * 60)

    engine = FastLLMEngine()
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

        torch.cuda.synchronize()
        start = time.time()
        response = engine.generate(user_input)
        torch.cuda.synchronize()

        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        print(f"Maya ({elapsed:.0f}ms): {response}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average latency: {sum(times)/len(times):.0f}ms")
    print(f"Min: {min(times):.0f}ms")
    print(f"Max: {max(times):.0f}ms")
    print(f"Target: <200ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
