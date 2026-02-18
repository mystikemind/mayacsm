#!/usr/bin/env python3
"""
Diagnose LLM latency - find where time is spent.
"""

import sys
import time
import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

from maya.config import LLM


def diagnose():
    print("=" * 60)
    print("LLM LATENCY DIAGNOSIS")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(LLM.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Tokenizer loaded: {(time.time()-start)*1000:.0f}ms")

    # Load model - BF16
    print("\n2. Loading model (BF16)...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        LLM.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",  # Direct CUDA, no auto
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use scaled_dot_product_attention
    )
    model.eval()
    print(f"   Model loaded: {(time.time()-start)*1000:.0f}ms")

    # Prepare test input
    messages = [
        {"role": "system", "content": "Respond in 6-8 words."},
        {"role": "user", "content": "Hello, how are you?"}
    ]

    # Time tokenization
    print("\n3. Tokenization timing...")
    start = time.time()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenize_template_ms = (time.time() - start) * 1000

    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    tokenize_encode_ms = (time.time() - start) * 1000
    print(f"   Chat template: {tokenize_template_ms:.1f}ms")
    print(f"   Encoding: {tokenize_encode_ms:.1f}ms")
    print(f"   Input tokens: {inputs['input_ids'].shape[1]}")

    # Time generation WITHOUT compile
    print("\n4. Generation WITHOUT torch.compile...")
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
        print(f"   Run {i+1}: {elapsed:.0f}ms - '{response}'")

    print(f"   Average (no compile): {sum(times[2:])/len(times[2:]):.0f}ms (excluding warmup)")

    # Now test WITH torch.compile
    print("\n5. Applying torch.compile (reduce-overhead)...")
    start = time.time()
    compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    print(f"   Compile call: {(time.time()-start)*1000:.0f}ms")

    print("\n6. Generation WITH torch.compile...")
    times_compiled = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = compiled_model.generate(
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
        times_compiled.append(elapsed)

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"   Run {i+1}: {elapsed:.0f}ms - '{response}'")

    print(f"   Average (compiled, last 5): {sum(times_compiled[5:])/5:.0f}ms")

    # Now try max-autotune mode
    print("\n7. Testing max-autotune mode...")

    # Clear cache and reload
    del compiled_model
    del model
    torch.cuda.empty_cache()

    model2 = AutoModelForCausalLM.from_pretrained(
        LLM.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model2.eval()

    compiled_model2 = torch.compile(model2, mode='max-autotune', fullgraph=False)

    print("   Warming up max-autotune...")
    times_autotune = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = compiled_model2.generate(
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
        times_autotune.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.0f}ms")

    print(f"   Average (max-autotune, last 5): {sum(times_autotune[5:])/5:.0f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"No compile avg: {sum(times[2:])/len(times[2:]):.0f}ms")
    print(f"reduce-overhead avg: {sum(times_compiled[5:])/5:.0f}ms")
    print(f"max-autotune avg: {sum(times_autotune[5:])/5:.0f}ms")
    print("=" * 60)


if __name__ == "__main__":
    diagnose()
