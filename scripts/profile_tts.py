"""
Profile the Original CSM Generator to find optimization opportunities.

Measures:
1. Frame generation loop time
2. Audio decoding (Mimi) time
3. Watermarking time
"""

import torch
import torchaudio
import time
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from generator import load_csm_1b, Segment
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark

# Short test phrase (like our LLM output)
TEST_TEXT = "I'm great, how about you?"

print("=" * 60)
print("CSM TTS Profiling")
print("=" * 60)

print("\nLoading generator...")
generator = load_csm_1b(device="cuda")

# Warmup
print("Warming up...")
for _ in range(3):
    _ = generator.generate(text="hello", speaker=0, context=[], max_audio_length_ms=2000)
print("Warmup complete\n")

# Now profile a generation manually
print(f"Profiling: '{TEST_TEXT}'")
print("-" * 40)

# Reset caches
generator._model.reset_caches()

# Tokenize text
t0 = time.time()
gen_tokens, gen_mask = generator._tokenize_text_segment(TEST_TEXT, 0)
tokenize_time = (time.time() - t0) * 1000

prompt_tokens = gen_tokens.long().to(generator.device).unsqueeze(0)
prompt_tokens_mask = gen_mask.bool().to(generator.device).unsqueeze(0)
curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(generator.device)

# Frame generation loop
max_frames = int(5000 / 80)  # 5 seconds max
samples = []

t0 = time.time()
for i in range(max_frames):
    sample = generator._model.generate_frame(prompt_tokens, prompt_tokens_mask, curr_pos, 0.9, 50)
    if torch.all(sample == 0):
        break
    samples.append(sample)
    prompt_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(generator.device)], dim=1).unsqueeze(1)
    prompt_tokens_mask = torch.cat(
        [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(generator.device)], dim=1
    ).unsqueeze(1)
    curr_pos = curr_pos[:, -1:] + 1

frame_time = (time.time() - t0) * 1000
num_frames = len(samples)

# Audio decoding
t0 = time.time()
audio_tokens = torch.stack(samples).permute(1, 2, 0)  # (1, 32, frames)
audio = generator._audio_tokenizer.decode(audio_tokens).squeeze(0).squeeze(0)
decode_time = (time.time() - t0) * 1000

# Watermarking
t0 = time.time()
audio_wm, wm_sample_rate = watermark(generator._watermarker, audio, generator.sample_rate, CSM_1B_GH_WATERMARK)
audio_wm = torchaudio.functional.resample(audio_wm, orig_freq=wm_sample_rate, new_freq=generator.sample_rate)
watermark_time = (time.time() - t0) * 1000

# Results
audio_duration = len(audio) / generator.sample_rate
total_time = tokenize_time + frame_time + decode_time + watermark_time

print(f"Frames generated: {num_frames}")
print(f"Audio duration:   {audio_duration:.2f}s")
print()
print("TIME BREAKDOWN:")
print(f"  Tokenize:       {tokenize_time:6.0f}ms ({100*tokenize_time/total_time:4.1f}%)")
print(f"  Frame gen:      {frame_time:6.0f}ms ({100*frame_time/total_time:4.1f}%)")
print(f"  Audio decode:   {decode_time:6.0f}ms ({100*decode_time/total_time:4.1f}%)")
print(f"  Watermark:      {watermark_time:6.0f}ms ({100*watermark_time/total_time:4.1f}%)")
print(f"  ---------------")
print(f"  TOTAL:          {total_time:6.0f}ms")
print()
print(f"RTF (with watermark):    {(total_time/1000)/audio_duration:.2f}x")
print(f"RTF (without watermark): {(total_time-watermark_time)/1000/audio_duration:.2f}x")
print()
print(f"Time per frame:  {frame_time/num_frames:.1f}ms")

# Save both versions for comparison
torchaudio.save("/tmp/profile_with_wm.wav", audio_wm.unsqueeze(0).cpu(), generator.sample_rate)
torchaudio.save("/tmp/profile_without_wm.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
print()
print("Saved: /tmp/profile_with_wm.wav (with watermark)")
print("Saved: /tmp/profile_without_wm.wav (without watermark - compare quality)")
print("=" * 60)
