#!/usr/bin/env python3
"""
Q4_K_M vs Q8_0 Quality Comparison for Orpheus TTS
===================================================

Tests whether higher quantization (Q8_0) produces better quality audio.
Q8_0 uses ~1GB more VRAM but may preserve more model information.

Starts a temporary Q8_0 server on port 5008 for comparison.
"""

import sys, os, time, re, json, signal, subprocess
import torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
import requests
from pathlib import Path
from snac import SNAC
from maya.config import TTS, DEVICES

# Test prompts - conversational speech (no emotion tags to isolate quantization effect)
TEST_PROMPTS = [
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "honestly i think thats a great idea, you should definitely go for it",
    "what made you think of that",
    "oh wow i didnt know that, tell me more about it",
    "yeah i know what you mean, sometimes things just dont work out",
    "hey there, im maya, its nice to meet you",
    "hmm yeah that makes sense, ive been thinking about that too",
    "do you want to grab coffee sometime",
    "thats really interesting, i never thought about it that way",
]

AUDIO_TOKEN_BASE = 128266
SNAC_CODEBOOK_SIZE = 4096
CUSTOM_TOKEN_OFFSET = 128256
AUDIO_TOKEN_MIN = AUDIO_TOKEN_BASE
AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * SNAC_CODEBOOK_SIZE) - 1

GGUF_DIR = "/home/ec2-user/SageMaker/.cache/huggingface/hub/models--QuantFactory--orpheus-3b-0.1-ft-GGUF"
LLAMA_SERVER = "/home/ec2-user/SageMaker/llama.cpp/build/bin/llama-server"
LLAMA_LIB_PATH = "/home/ec2-user/SageMaker/llama.cpp/build/ggml/src:/home/ec2-user/SageMaker/llama.cpp/build/ggml/src/ggml-cuda"


def find_gguf(quant):
    for sd in Path(GGUF_DIR).glob("snapshots/*"):
        for g in sd.glob(f"*{quant}*"):
            if g.resolve().exists():
                return str(g)
    return None


def start_server(gguf_path, port, gpu_index):
    """Start a llama-server and wait for readiness."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["LD_LIBRARY_PATH"] = LLAMA_LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        LLAMA_SERVER, "-m", gguf_path,
        "-c", "4096", "-ngl", "99",
        "--host", "127.0.0.1", "--port", str(port),
        "-fa", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "-np", "1", "--mlock",
    ]

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    for _ in range(60):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return proc
        except:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"Server died: {proc.stdout.read().decode()[-300:]}")

    raise TimeoutError("Server failed to start")


def stop_server(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except:
                pass


def extract_audio_tokens(text_output):
    tokens = []
    for m in re.finditer(r'<custom_token_(\d+)>', text_output):
        tid = CUSTOM_TOKEN_OFFSET + int(m.group(1))
        if AUDIO_TOKEN_MIN <= tid <= AUDIO_TOKEN_MAX:
            tokens.append(tid)
    return tokens


def decode_snac(token_ids, snac_model):
    n = (len(token_ids) // 7) * 7
    if n < 7:
        return None
    token_ids = token_ids[:n]
    codes = [t - AUDIO_TOKEN_BASE for t in token_ids]
    l0, l1, l2 = [], [], []
    for i in range(n // 7):
        b = 7 * i
        l0.append(max(0, min(4095, codes[b])))
        l1.append(max(0, min(4095, codes[b+1] - 4096)))
        l2.append(max(0, min(4095, codes[b+2] - 2*4096)))
        l2.append(max(0, min(4095, codes[b+3] - 3*4096)))
        l1.append(max(0, min(4095, codes[b+4] - 4*4096)))
        l2.append(max(0, min(4095, codes[b+5] - 5*4096)))
        l2.append(max(0, min(4095, codes[b+6] - 6*4096)))

    device = next(snac_model.parameters()).device
    snac_codes = [
        torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(device),
    ]
    with torch.inference_mode():
        audio = snac_model.decode(snac_codes)
    return audio.squeeze().cpu()


def generate_and_score(url, text, voice, snac_model, utmos_model, device, session):
    processed = text.lower().strip()
    processed = re.sub(r'\.$', '', processed).strip()
    prompt = f"<|begin_of_text|><custom_token_3>{voice}: {processed}<custom_token_4><custom_token_5>"

    words = len(processed.split())
    est_s = max(words / 3.0, 0.5)
    margin = 2.5 if words <= 3 else (2.0 if words <= 8 else 1.8)
    max_tokens = min(max(int(est_s * margin * 82) + 20, 80), 420)

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": TTS.temperature,
        "top_p": TTS.top_p,
        "top_k": TTS.top_k,
        "min_p": TTS.min_p,
        "repeat_penalty": TTS.repeat_penalty,
        "repeat_last_n": TTS.repeat_last_n,
        "stop": ["<custom_token_2>"],
        "stream": False,
    }

    t0 = time.time()
    resp = session.post(f"{url}/v1/completions", json=payload, timeout=120)
    resp.raise_for_status()
    gen_time = time.time() - t0

    data = resp.json()
    text_out = data["choices"][0]["text"]
    tokens = extract_audio_tokens(text_out)
    audio = decode_snac(tokens, snac_model)

    if audio is None or audio.numel() < 100:
        return None

    audio_np = audio.numpy().astype('float32')
    duration = len(audio_np) / 24000

    # Score
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device)).item()

    return {"score": score, "duration": duration, "gen_time": gen_time, "n_tokens": len(tokens)}


def main():
    device = "cuda:2"
    gpu_index = 2  # TTS GPU

    print("Loading SNAC + UTMOS...")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    # Check if Q4_K_M server already running on port 5006
    q4_url = "http://127.0.0.1:5006"
    q4_running = False
    try:
        r = requests.get(f"{q4_url}/health", timeout=2)
        q4_running = r.status_code == 200
    except:
        pass

    q4_proc = None
    if not q4_running:
        print("Starting Q4_K_M server on port 5006...")
        q4_path = find_gguf("Q4_K_M")
        q4_proc = start_server(q4_path, 5006, gpu_index)
    else:
        print("Q4_K_M server already running on port 5006")

    # Start Q8_0 on port 5008
    print("Starting Q8_0 server on port 5008...")
    q8_path = find_gguf("Q8_0")
    q8_proc = start_server(q8_path, 5008, gpu_index)
    q8_url = "http://127.0.0.1:5008"

    session = requests.Session()

    # Warmup both
    print("Warming up both servers...")
    for url in [q4_url, q8_url]:
        for _ in range(2):
            generate_and_score(url, "hello", "jess", snac, utmos, device, session)

    print(f"\n{'=' * 90}")
    print(f"  Q4_K_M vs Q8_0 QUALITY COMPARISON")
    print(f"  Voice: jess | {len(TEST_PROMPTS)} prompts | 3 runs each for stability")
    print(f"{'=' * 90}\n")

    configs = [
        ("Q4_K_M", q4_url),
        ("Q8_0", q8_url),
    ]

    all_results = {}

    for quant_name, url in configs:
        print(f"\n{'─' * 90}")
        print(f"  {quant_name}")
        print(f"{'─' * 90}")

        scores = []
        durations = []
        gen_times = []

        for i, text in enumerate(TEST_PROMPTS):
            # Run 3 times for stability (TTS is stochastic)
            prompt_scores = []
            for run in range(3):
                result = generate_and_score(url, text, "jess", snac, utmos, device, session)
                if result:
                    prompt_scores.append(result["score"])
                    if run == 0:
                        durations.append(result["duration"])
                        gen_times.append(result["gen_time"])

            if prompt_scores:
                avg_score = np.mean(prompt_scores)
                scores.append(avg_score)
                print(
                    f"  [{i+1:2d}] UTMOS={avg_score:.3f} (runs: {', '.join(f'{s:.3f}' for s in prompt_scores)}) "
                    f"| '{text[:50]}'"
                )

        if scores:
            all_results[quant_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "dur_mean": np.mean(durations),
                "gen_mean": np.mean(gen_times),
                "scores": scores,
            }

    # Cleanup Q8_0 server
    print("\nStopping Q8_0 server...")
    stop_server(q8_proc)
    if q4_proc:
        stop_server(q4_proc)

    # Summary
    print(f"\n{'=' * 90}")
    print(f"  QUANTIZATION QUALITY COMPARISON SUMMARY")
    print(f"{'=' * 90}\n")

    for name, stats in all_results.items():
        print(f"  {name}:")
        print(f"    UTMOS:    {stats['mean']:.3f} ± {stats['std']:.3f} (min={stats['min']:.3f}, max={stats['max']:.3f})")
        print(f"    Duration: {stats['dur_mean']:.1f}s avg")
        print(f"    Gen time: {stats['gen_mean']:.2f}s avg")

    if "Q4_K_M" in all_results and "Q8_0" in all_results:
        diff = all_results["Q8_0"]["mean"] - all_results["Q4_K_M"]["mean"]
        print(f"\n  Quality difference (Q8_0 - Q4_K_M): {diff:+.3f}")
        if diff > 0.05:
            print(f"  RECOMMENDATION: Switch to Q8_0 (+{diff:.3f} UTMOS, worth the ~1GB extra VRAM)")
        elif diff > 0:
            print(f"  MARGINAL: Q8_0 slightly better but difference may not be perceptible")
        else:
            print(f"  KEEP Q4_K_M: No quality benefit from Q8_0")

        # Per-prompt comparison
        print(f"\n  Per-prompt comparison:")
        q4s = all_results["Q4_K_M"]["scores"]
        q8s = all_results["Q8_0"]["scores"]
        n = min(len(q4s), len(q8s))
        q4_wins = sum(1 for i in range(n) if q4s[i] > q8s[i])
        q8_wins = n - q4_wins
        print(f"    Q4_K_M wins: {q4_wins}/{n}")
        print(f"    Q8_0 wins:   {q8_wins}/{n}")


if __name__ == "__main__":
    main()
