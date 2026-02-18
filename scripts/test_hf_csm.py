"""
Test HuggingFace CSM vs Original CSM Generator

GOAL: Validate HF implementation produces SAME quality audio as original
before integrating it for speed improvements.

This script:
1. Generates audio with ORIGINAL generator (known good quality)
2. Generates audio with HF Transformers CSM (potentially faster)
3. Compares RTF and saves audio for listening comparison
"""

import torch
import torchaudio
import time
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Test phrase - short like our LLM outputs
TEST_TEXT = "Hi there! I'm Maya. How can I help you?"

print("=" * 60)
print("HuggingFace CSM vs Original CSM - Quality Validation")
print("=" * 60)

# -----------------------------------------------------------------------------
# Test 1: Original CSM Generator (known good quality)
# -----------------------------------------------------------------------------
print("\n[1/2] Testing ORIGINAL CSM Generator...")

from generator import load_csm_1b

original_gen = load_csm_1b(device="cuda")

# Warmup
print("  Warming up...")
_ = original_gen.generate(text="hello", speaker=0, context=[], max_audio_length_ms=2000)

# Generate test audio
print(f"  Generating: '{TEST_TEXT}'")
start = time.time()
original_audio = original_gen.generate(
    text=TEST_TEXT,
    speaker=0,
    context=[],
    max_audio_length_ms=10000,  # Same as current settings
    temperature=0.9,
    topk=50
)
original_time = time.time() - start
original_duration = len(original_audio) / 24000
original_rtf = original_time / original_duration

print(f"  Duration: {original_duration:.2f}s")
print(f"  Generation time: {original_time:.2f}s")
print(f"  RTF: {original_rtf:.2f}x")

# Save
torchaudio.save("/tmp/test_original_csm.wav", original_audio.unsqueeze(0).cpu(), 24000)
print(f"  Saved: /tmp/test_original_csm.wav")

# Free memory
del original_gen
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Test 2: HuggingFace Transformers CSM with Static Cache
# -----------------------------------------------------------------------------
print("\n[2/2] Testing HuggingFace CSM with static cache...")

from transformers import CsmForConditionalGeneration, AutoProcessor

# Enable TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print("  Loading model...")
processor = AutoProcessor.from_pretrained("sesame/csm-1b")
model = CsmForConditionalGeneration.from_pretrained(
    "sesame/csm-1b",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)

# Configure static cache (key optimization from PDF)
print("  Configuring static cache...")
model.generation_config.max_length = 250
model.generation_config.max_new_tokens = None
model.generation_config.cache_implementation = "static"
model.depth_decoder.generation_config.cache_implementation = "static"

# Prepare inputs
inputs = processor(
    text=TEST_TEXT,
    audio=None,  # No context audio
    return_tensors="pt"
).to("cuda")

# Generation config (greedy for speed as PDF suggests)
gen_kwargs = {
    "do_sample": True,  # Keep sampling for quality comparison
    "temperature": 0.9,
    "depth_decoder_do_sample": True,
    "depth_decoder_temperature": 0.9,
}

# Warmup (critical - first run compiles)
print("  Warming up (this takes a while for compilation)...")
warmup_inputs = processor(text="hello", audio=None, return_tensors="pt").to("cuda")
for i in range(3):
    print(f"    Warmup {i+1}/3...")
    _ = model.generate(**warmup_inputs, **gen_kwargs)
print("  Warmup complete")

# Generate test audio
print(f"  Generating: '{TEST_TEXT}'")
start = time.time()
hf_output = model.generate(**inputs, **gen_kwargs)
hf_time = time.time() - start

# Decode audio
hf_audio = hf_output.audio_values[0].cpu()
hf_duration = len(hf_audio) / model.config.audio_encoder.sampling_rate
hf_rtf = hf_time / hf_duration

print(f"  Duration: {hf_duration:.2f}s")
print(f"  Generation time: {hf_time:.2f}s")
print(f"  RTF: {hf_rtf:.2f}x")

# Save
if hf_audio.dim() == 1:
    hf_audio = hf_audio.unsqueeze(0)
torchaudio.save("/tmp/test_hf_csm.wav", hf_audio, model.config.audio_encoder.sampling_rate)
print(f"  Saved: /tmp/test_hf_csm.wav")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Original CSM: RTF={original_rtf:.2f}x ({original_time:.1f}s for {original_duration:.1f}s audio)")
print(f"HF CSM:       RTF={hf_rtf:.2f}x ({hf_time:.1f}s for {hf_duration:.1f}s audio)")
print(f"Speedup:      {original_rtf/hf_rtf:.2f}x faster")
print()
print("Audio files saved:")
print("  1. /tmp/test_original_csm.wav (REFERENCE - known good)")
print("  2. /tmp/test_hf_csm.wav (TEST - check if quality matches)")
print()
print("IMPORTANT: Listen to both files and compare quality!")
print("Only use HF CSM if audio quality is acceptable.")
print("=" * 60)
