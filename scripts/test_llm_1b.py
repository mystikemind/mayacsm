#!/usr/bin/env python3
"""
Test Llama 3.2 1B latency vs 3B.
1B should be ~3x faster for same quality tier.
"""

import sys
import time
import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')


def test_model(model_id: str, name: str):
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load
    print("Loading...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Loaded in {time.time()-start:.1f}s")

    # Test prompts
    test_cases = [
        ("Short", [
            {"role": "system", "content": "Respond naturally in 6-8 words."},
            {"role": "user", "content": "Hey how are you?"}
        ]),
        ("With history", [
            {"role": "system", "content": "You are maya, a friendly assistant. Respond naturally in 6-8 words."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hey! Nice to meet you, how's it going?"},
            {"role": "user", "content": "I'm doing well, thanks. What about you?"},
        ]),
    ]

    for test_name, messages in test_cases:
        print(f"\n{test_name} prompt test:")

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        print(f"  Input tokens: {inputs['input_ids'].shape[1]}")

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7, use_cache=True)

        # Time it
        times = []
        for i in range(5):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"  Run {i+1}: {elapsed:.0f}ms - '{response}'")

        avg = sum(times) / len(times)
        print(f"  Average: {avg:.0f}ms")

    return model, tokenizer


def main():
    print("=" * 60)
    print("LLM MODEL COMPARISON: 1B vs 3B")
    print("=" * 60)

    # Test 1B
    model_1b, _ = test_model("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B")

    # Clean up
    del model_1b
    torch.cuda.empty_cache()

    # Test 3B
    model_3b, _ = test_model("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B")

    del model_3b
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
