"""
Test torch.compile on CSM Depth Decoder specifically.

The depth decoder runs 31 iterations per frame - it's the bottleneck.
"""

import torch
import torchaudio
import time
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from models import Model
from generator import load_llama3_tokenizer
from moshi.models import loaders
from huggingface_hub import hf_hub_download

TEST_TEXT = "I'm great, how about you?"

print("=" * 60)
print("Testing torch.compile on CSM Depth Decoder")
print("=" * 60)

# Load model
print("\nLoading model...")
model = Model.from_pretrained("sesame/csm-1b")
model.to(device="cuda", dtype=torch.bfloat16)
model.eval()

print(f"\nModel decoder: {type(model.decoder)}")

# Try different compile modes
print("\nApplying torch.compile to depth decoder...")
try:
    model.decoder = torch.compile(
        model.decoder,
        mode='reduce-overhead',  # Fast for repeated patterns
        fullgraph=False,  # Allow graph breaks if needed
    )
    print("  Decoder compiled with mode='reduce-overhead'")
except Exception as e:
    print(f"  Failed: {e}")

# Also try compiling backbone (might help)
print("Applying torch.compile to backbone...")
try:
    model.backbone = torch.compile(
        model.backbone,
        mode='max-autotune',  # More aggressive
        fullgraph=False,
    )
    print("  Backbone compiled with mode='max-autotune'")
except Exception as e:
    print(f"  Failed: {e}")

# Setup generator components
model.setup_caches(1)
text_tokenizer = load_llama3_tokenizer()

device = next(model.parameters()).device
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)
mimi.set_num_codebooks(32)

class CompiledGenerator:
    def __init__(self, model, tokenizer, audio_tokenizer):
        self._model = model
        self._text_tokenizer = tokenizer
        self._audio_tokenizer = audio_tokenizer
        self.sample_rate = audio_tokenizer.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text, speaker):
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        return text_frame.to(self.device), text_frame_mask.to(self.device)

    @torch.inference_mode()
    def generate(self, text, speaker=0, max_audio_length_ms=5000, temperature=0.9, topk=50):
        self._model.reset_caches()

        max_frames = int(max_audio_length_ms / 80)
        gen_tokens, gen_mask = self._tokenize_text_segment(text, speaker)

        prompt_tokens = gen_tokens.long().unsqueeze(0)
        prompt_tokens_mask = gen_mask.bool().unsqueeze(0)
        curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(self.device)

        samples = []
        for _ in range(max_frames):
            sample = self._model.generate_frame(prompt_tokens, prompt_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break
            samples.append(sample)
            prompt_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            prompt_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        return audio

generator = CompiledGenerator(model, text_tokenizer, mimi)

# Extended warmup (compilation can take multiple runs)
print("\nWarming up (compilation happens incrementally)...")
for i in range(10):
    start = time.time()
    _ = generator.generate("hello", max_audio_length_ms=2000)
    elapsed = (time.time() - start) * 1000
    print(f"  Warmup {i+1}/10: {elapsed:.0f}ms")
    if elapsed < 1500 and i >= 4:  # Stop early if converged
        print("  Converged!")
        break

# Benchmark
print(f"\nBenchmarking: '{TEST_TEXT}'")
print("-" * 40)

times = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.time()
    audio = generator.generate(TEST_TEXT, max_audio_length_ms=5000)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)

    duration = len(audio) / generator.sample_rate
    rtf = elapsed / duration
    print(f"  Run {i+1}: {elapsed*1000:.0f}ms for {duration:.2f}s audio (RTF: {rtf:.2f}x)")

# Remove slowest run (potential outlier)
times.sort()
times = times[:-1]  # Remove slowest

avg_time = sum(times) / len(times)
duration = len(audio) / generator.sample_rate
avg_rtf = avg_time / duration

print()
print(f"BEST RTF: {min(times)/duration:.2f}x")
print(f"AVERAGE RTF (excluding worst): {avg_rtf:.2f}x")

# Save audio
torchaudio.save("/tmp/test_compiled_decoder.wav", audio.unsqueeze(0).cpu().float(), generator.sample_rate)
print(f"\nSaved: /tmp/test_compiled_decoder.wav")
print("=" * 60)
