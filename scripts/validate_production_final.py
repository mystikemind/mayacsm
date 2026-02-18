#!/usr/bin/env python3
"""
Comprehensive Production Validation for Maya Pipeline.

Tests:
1. TTS quality (UTMOS) across diverse prompts
2. First-chunk latency
3. Chunk boundary quality (click detection)
4. Post-processing effect
5. Trailing silence handling
6. LLM response quality
7. End-to-end pipeline timing
"""

import sys, os, time, re, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from maya.config import TTS, LLM, AUDIO

# Test prompts covering different scenarios
TEST_PROMPTS = [
    # Short responses (2-4 words)
    "oh really",
    "yeah for sure",
    "mhm",
    "thats awesome",
    # Medium responses (5-10 words)
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "hmm yeah that makes sense",
    "okay cool, thanks for letting me know",
    # Longer responses (10-15 words)
    "honestly i think thats a great idea, you should definitely go for it",
    "yeah i know what you mean, sometimes things just dont work out",
    # Emotional tags
    "<laugh> thats hilarious, what happened next",
    "<sigh> that sounds tough, im sorry youre dealing with that",
    # Edge cases
    "hi im maya, how can i help you",  # greeting
    "hey, are you still there",  # idle prompt
]

AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MIN = AUDIO_TOKEN_BASE
AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * 4096) - 1
CUSTOM_TOKEN_OFFSET = 128256


def generate_tts(text, voice="jess", stream=False):
    """Generate audio via TTS llama-server."""
    prompt = f"<|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"
    payload = {
        "prompt": prompt,
        "max_tokens": TTS.max_tokens,
        "temperature": TTS.temperature,
        "top_p": TTS.top_p,
        "top_k": TTS.top_k,
        "min_p": TTS.min_p,
        "repeat_penalty": TTS.repeat_penalty,
        "repeat_last_n": TTS.repeat_last_n,
        "stop": ["<custom_token_2>"],
        "stream": stream,
    }
    t0 = time.time()
    resp = requests.post("http://127.0.0.1:5006/v1/completions", json=payload, timeout=60)
    resp.raise_for_status()
    gen_time = time.time() - t0

    text_output = resp.json()["choices"][0]["text"]
    token_ids = []
    for match in re.finditer(r'<custom_token_(\d+)>', text_output):
        custom_num = int(match.group(1))
        tid = CUSTOM_TOKEN_OFFSET + custom_num
        if AUDIO_TOKEN_MIN <= tid <= AUDIO_TOKEN_MAX:
            token_ids.append(tid)

    return token_ids, gen_time


def generate_tts_streaming(text, voice="jess"):
    """Generate audio via TTS streaming, measure first-token latency."""
    prompt = f"<|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"
    payload = {
        "prompt": prompt,
        "max_tokens": TTS.max_tokens,
        "temperature": TTS.temperature,
        "top_p": TTS.top_p,
        "top_k": TTS.top_k,
        "min_p": TTS.min_p,
        "repeat_penalty": TTS.repeat_penalty,
        "repeat_last_n": TTS.repeat_last_n,
        "stop": ["<custom_token_2>"],
        "stream": True,
    }
    t0 = time.time()
    resp = requests.post("http://127.0.0.1:5006/v1/completions", json=payload, timeout=60, stream=True)
    resp.raise_for_status()

    first_token_time = None
    token_ids = []
    buffer = ""

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(data_str)
            token_text = data["choices"][0].get("text", "")
            buffer += token_text

            while True:
                match = re.search(r'<custom_token_(\d+)>', buffer)
                if not match:
                    break
                custom_num = int(match.group(1))
                tid = CUSTOM_TOKEN_OFFSET + custom_num
                if AUDIO_TOKEN_MIN <= tid <= AUDIO_TOKEN_MAX:
                    if first_token_time is None:
                        first_token_time = time.time() - t0
                    token_ids.append(tid)
                buffer = buffer[match.end():]
        except (json.JSONDecodeError, KeyError):
            continue

    total_time = time.time() - t0
    return token_ids, first_token_time or 0, total_time


def decode_snac(token_ids, snac, device):
    """Decode SNAC tokens to audio."""
    n = (len(token_ids) // 7) * 7
    if n < 7:
        return None
    token_ids = token_ids[:n]
    codes = [t - AUDIO_TOKEN_BASE for t in token_ids]
    l0, l1, l2 = [], [], []
    for i in range(n // 7):
        b = 7 * i
        l0.append(max(0, min(4095, codes[b])))
        l1.append(max(0, min(4095, codes[b + 1] - 4096)))
        l2.append(max(0, min(4095, codes[b + 2] - 2 * 4096)))
        l2.append(max(0, min(4095, codes[b + 3] - 3 * 4096)))
        l1.append(max(0, min(4095, codes[b + 4] - 4 * 4096)))
        l2.append(max(0, min(4095, codes[b + 5] - 5 * 4096)))
        l2.append(max(0, min(4095, codes[b + 6] - 6 * 4096)))

    snac_codes = [
        torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(device),
    ]
    with torch.inference_mode():
        audio = snac.decode(snac_codes)
    return audio.squeeze().cpu().numpy()


def detect_clicks(audio_np, threshold=0.3, window=10):
    """Detect clicks/pops as sudden amplitude jumps."""
    if len(audio_np) < window * 2:
        return 0
    diff = np.abs(np.diff(audio_np))
    clicks = np.sum(diff > threshold)
    return clicks


def measure_trailing_silence(audio_np, sr=24000, threshold=0.01, window_ms=50):
    """Measure trailing silence duration in ms."""
    window = int(sr * window_ms / 1000)
    last_voiced = 0
    for i in range(0, len(audio_np) - window, window // 2):
        rms = np.sqrt(np.mean(audio_np[i:i+window] ** 2))
        if rms > threshold:
            last_voiced = i + window
    trailing_ms = (len(audio_np) - last_voiced) / sr * 1000
    return trailing_ms


def test_llm(transcript):
    """Test LLM response."""
    payload = {
        "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{LLM.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "max_tokens": LLM.max_new_tokens,
        "temperature": LLM.temperature,
        "top_p": LLM.top_p,
        "stop": ["<|eot_id|>", "<|end_of_text|>"],
        "stream": False,
    }
    t0 = time.time()
    resp = requests.post("http://127.0.0.1:5007/v1/completions", json=payload, timeout=30)
    resp.raise_for_status()
    llm_time = time.time() - t0
    response = resp.json()["choices"][0]["text"].strip()
    return response, llm_time


def main():
    device = "cuda:2"
    print("Loading SNAC + UTMOS...")
    from snac import SNAC
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    from maya.engine.audio_post_processor import post_process

    print("\n" + "=" * 80)
    print("  MAYA PRODUCTION VALIDATION")
    print("  Voice: jess | Params: temp=0.6 top_p=0.9 top_k=50 min_p=0.05 rep=1.1")
    print("=" * 80)

    # ===========================================================================
    # TEST 1: TTS Quality + Post-Processing
    # ===========================================================================
    print("\n--- TEST 1: TTS Quality (UTMOS) + Post-Processing ---")
    raw_scores, proc_scores, durations, click_counts, trailing_silences = [], [], [], [], []
    gen_times = []

    for i, text in enumerate(TEST_PROMPTS):
        tokens, gen_time = generate_tts(text)
        gen_times.append(gen_time)

        if len(tokens) < 7:
            print(f"  [{i+1:2d}] SKIP (no tokens): {text[:50]}")
            continue

        audio = decode_snac(tokens, snac, device)
        if audio is None:
            print(f"  [{i+1:2d}] SKIP (decode fail): {text[:50]}")
            continue

        dur = len(audio) / 24000
        durations.append(dur)

        # Raw UTMOS
        wav = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.inference_mode():
            raw_score = utmos(wav, torch.tensor([24000]).to(device)).item()
        raw_scores.append(raw_score)

        # Post-processed UTMOS
        proc = post_process(audio, sample_rate=24000)
        wav_p = torch.from_numpy(proc).unsqueeze(0).to(device)
        with torch.inference_mode():
            proc_score = utmos(wav_p, torch.tensor([24000]).to(device)).item()
        proc_scores.append(proc_score)

        # Click detection
        clicks = detect_clicks(audio)
        click_counts.append(clicks)

        # Trailing silence
        trail_ms = measure_trailing_silence(audio)
        trailing_silences.append(trail_ms)

        status = "OK" if raw_score >= 4.0 else "LOW"
        print(f"  [{i+1:2d}] {status} Raw={raw_score:.3f} Proc={proc_score:.3f} "
              f"Dur={dur:.1f}s Clicks={clicks} Trail={trail_ms:.0f}ms | {text[:45]}")

    # ===========================================================================
    # TEST 2: Streaming Latency
    # ===========================================================================
    print("\n--- TEST 2: Streaming First-Token Latency ---")
    stream_prompts = ["hey there", "yeah im doing great", "honestly i think thats a great idea"]
    first_token_times = []

    for text in stream_prompts:
        tokens, ft_time, total = generate_tts_streaming(text)
        first_token_times.append(ft_time * 1000)  # ms
        n_tokens = len(tokens)
        tok_s = n_tokens / total if total > 0 else 0
        print(f"  First token: {ft_time*1000:.0f}ms | Total: {total*1000:.0f}ms | "
              f"Tokens: {n_tokens} ({tok_s:.0f} tok/s) | {text[:40]}")

    # ===========================================================================
    # TEST 3: LLM Response Quality
    # ===========================================================================
    print("\n--- TEST 3: LLM Response Quality ---")
    llm_prompts = [
        "hey how are you doing today",
        "what do you think about artificial intelligence",
        "im feeling a bit stressed out",
        "do you like music",
        "tell me something interesting",
    ]
    llm_times = []

    for text in llm_prompts:
        response, lt = test_llm(text)
        llm_times.append(lt * 1000)
        word_count = len(response.split())
        status = "OK" if 3 <= word_count <= 20 else "WARN"
        print(f"  [{status}] {lt*1000:.0f}ms ({word_count}w): '{text[:30]}' -> '{response}'")

    # ===========================================================================
    # TEST 4: Chunk Boundary Simulation
    # ===========================================================================
    print("\n--- TEST 4: Chunk Boundary Quality ---")
    # Generate a longer response and decode in chunks like streaming does
    test_text = "yeah im doing pretty good, hows everything with you today"
    tokens, _ = generate_tts(test_text)
    if len(tokens) >= 21:
        # Simulate 3 chunks: 2+6+remainder frames
        chunks = []
        frame_sizes = [2, 6, 6, 6]  # Frame counts per chunk
        idx = 0
        for fs in frame_sizes:
            chunk_tokens = tokens[idx:idx + fs * 7]
            if len(chunk_tokens) >= 7:
                audio = decode_snac(chunk_tokens, snac, device)
                if audio is not None:
                    chunks.append(audio)
            idx += fs * 7
            if idx >= len(tokens):
                break

        if len(chunks) >= 2:
            # Check boundary quality
            for i in range(len(chunks) - 1):
                end_val = chunks[i][-1]
                start_val = chunks[i+1][0]
                jump = abs(end_val - start_val)
                print(f"  Chunk {i}->{i+1}: boundary jump = {jump:.4f} "
                      f"({'CLICK RISK' if jump > 0.1 else 'OK'})")

            # Full audio (concatenated)
            full = np.concatenate(chunks)
            full_clicks = detect_clicks(full)
            print(f"  Concatenated {len(chunks)} chunks: {full_clicks} clicks, "
                  f"{len(full)/24000:.2f}s")

    # ===========================================================================
    # SUMMARY
    # ===========================================================================
    print("\n" + "=" * 80)
    print("  PRODUCTION VALIDATION SUMMARY")
    print("=" * 80)

    if raw_scores:
        print(f"\n  TTS Quality:")
        print(f"    UTMOS raw:       {np.mean(raw_scores):.3f} ± {np.std(raw_scores):.3f} "
              f"(min={np.min(raw_scores):.3f}, max={np.max(raw_scores):.3f})")
        print(f"    UTMOS processed: {np.mean(proc_scores):.3f} ± {np.std(proc_scores):.3f}")
        print(f"    Post-proc delta: {np.mean(proc_scores) - np.mean(raw_scores):+.3f}")
        below_4 = sum(1 for s in raw_scores if s < 4.0)
        print(f"    Below 4.0:       {below_4}/{len(raw_scores)} samples")

    if durations:
        print(f"\n  Audio Duration:")
        print(f"    Mean: {np.mean(durations):.2f}s (min={np.min(durations):.2f}, max={np.max(durations):.2f})")

    if click_counts:
        print(f"\n  Audio Artifacts:")
        print(f"    Clicks per sample: {np.mean(click_counts):.1f} (max={np.max(click_counts)})")

    if trailing_silences:
        print(f"    Trailing silence:  {np.mean(trailing_silences):.0f}ms "
              f"(max={np.max(trailing_silences):.0f}ms)")
        long_silence = sum(1 for s in trailing_silences if s > 1000)
        print(f"    >1s silence:       {long_silence}/{len(trailing_silences)}")

    if gen_times:
        print(f"\n  TTS Generation:")
        tokens_per_sample = [len(generate_tts(TEST_PROMPTS[0])[0])]  # rough estimate
        print(f"    Mean gen time:   {np.mean(gen_times)*1000:.0f}ms")

    if first_token_times:
        print(f"\n  Streaming Latency:")
        print(f"    First token:     {np.mean(first_token_times):.0f}ms "
              f"(min={np.min(first_token_times):.0f}, max={np.max(first_token_times):.0f})")
        # First audio = 2 frames = 14 tokens, ~110ms at 129 tok/s
        est_first_audio = np.mean(first_token_times) + (14 / 129 * 1000)
        print(f"    Est first audio: {est_first_audio:.0f}ms (first_tok + 14tok SNAC)")

    if llm_times:
        print(f"\n  LLM Performance:")
        print(f"    Mean latency:    {np.mean(llm_times):.0f}ms")

    # Overall verdict
    print(f"\n  {'=' * 40}")
    issues = []
    if raw_scores and np.mean(raw_scores) < 4.0:
        issues.append(f"UTMOS below 4.0 ({np.mean(raw_scores):.3f})")
    if first_token_times and np.mean(first_token_times) > 200:
        issues.append(f"First token too slow ({np.mean(first_token_times):.0f}ms)")
    if llm_times and np.mean(llm_times) > 500:
        issues.append(f"LLM too slow ({np.mean(llm_times):.0f}ms)")
    if trailing_silences and np.max(trailing_silences) > 2000:
        issues.append(f"Excessive trailing silence ({np.max(trailing_silences):.0f}ms)")

    if issues:
        print(f"  VERDICT: NEEDS WORK")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  VERDICT: PRODUCTION READY")

    print(f"  {'=' * 40}\n")


if __name__ == "__main__":
    main()
