#!/usr/bin/env python3
"""
Generate an OPTIMAL voice prompt for Maya based on research findings.

Research-backed settings:
1. Duration: 5-8 seconds (optimal: 2-10s per CSM docs)
2. Temperature: 0.9 (must match TTS generation for consistency)
3. topk: 50 (match TTS generation)
4. Single natural conversational phrase (not concatenated fragments)
5. Include emotional variety and natural speech patterns

This replaces the 26.5 second fragmented prompt with an optimal one.
"""

import sys
import os
import torch
import torchaudio
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

OUTPUT_DIR = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimal voice prompt text - natural, warm, conversational
# Should be ~5-8 seconds when spoken naturally
# Includes prosodic variety: question, warmth, different sentence types
MAYA_PROMPTS = [
    # Primary - warm greeting with natural flow
    "oh hey, hi! its really nice to meet you. im maya. so tell me, whats been on your mind?",
    # Alternative - slightly different emotional tone
    "hi there! im maya, and i just love talking to people. so, how are you doing today?",
    # Alternative - more casual
    "hey! im maya. you know, i really enjoy these conversations. whats going on with you?",
]

# Settings that MATCH tts_streaming_real.py (critical for voice consistency)
TEMPERATURE = 0.9  # Must match TTS generation
TOPK = 50          # Must match TTS generation
SPEAKER = 0        # Maya is speaker 0

def load_csm():
    """Load CSM model."""
    from generator import load_csm_1b
    print("Loading CSM-1B...")
    start = time.time()
    model = load_csm_1b(device="cuda")
    print(f"Model loaded in {time.time()-start:.1f}s")
    return model

def generate_take(generator, text, temperature=TEMPERATURE, topk=TOPK):
    """Generate a single take of the voice prompt."""
    # Calculate max audio length based on text
    # ~400ms per word for natural speech
    word_count = len(text.split())
    max_ms = min(max(word_count * 400, 3000), 10000)  # 3-10 seconds

    audio = generator.generate(
        text=text,
        speaker=SPEAKER,
        context=[],  # No context for reference
        max_audio_length_ms=max_ms,
        temperature=temperature,
        topk=topk,
    )
    return audio

def score_audio(audio, target_duration_range=(5.0, 8.0)):
    """
    Score audio quality based on:
    - Duration in optimal range
    - Good energy levels (not too quiet/loud)
    - No clipping
    - Low silence ratio
    """
    duration = len(audio) / 24000
    score = 0

    # Duration scoring (most important)
    min_dur, max_dur = target_duration_range
    if min_dur <= duration <= max_dur:
        score += 30  # Perfect range
    elif (min_dur - 1) <= duration <= (max_dur + 1):
        score += 20  # Acceptable range
    elif duration < min_dur - 2:
        score += 5   # Too short
    else:
        score += 10  # Too long

    # Energy check
    energy = audio.abs().mean().item()
    if 0.02 < energy < 0.06:
        score += 20  # Good energy
    elif 0.01 < energy < 0.08:
        score += 10  # Acceptable energy

    # Clipping check
    peak = audio.abs().max().item()
    if peak < 0.95:
        score += 15  # No clipping
    elif peak < 0.98:
        score += 5   # Minor clipping

    # Silence ratio (less than 30% silence is good)
    silence_ratio = (audio.abs() < 0.01).sum().item() / len(audio)
    if silence_ratio < 0.2:
        score += 15  # Very little silence
    elif silence_ratio < 0.3:
        score += 10  # Acceptable silence

    return score, {
        'duration': duration,
        'energy': energy,
        'peak': peak,
        'silence_ratio': silence_ratio
    }

def main():
    print("=" * 60)
    print("GENERATING OPTIMAL VOICE PROMPT FOR MAYA")
    print("=" * 60)
    print(f"\nTarget duration: 5-8 seconds")
    print(f"Temperature: {TEMPERATURE} (matches TTS generation)")
    print(f"TopK: {TOPK} (matches TTS generation)")
    print()

    generator = load_csm()

    best_audio = None
    best_score = -1
    best_text = None
    best_metrics = None

    # Generate multiple takes of each prompt
    for prompt_idx, text in enumerate(MAYA_PROMPTS):
        print(f"\n--- Prompt {prompt_idx + 1}: '{text[:50]}...' ---")

        for take in range(5):  # 5 takes per prompt
            print(f"  Take {take + 1}/5...", end=" ")
            start = time.time()

            audio = generate_take(generator, text)
            elapsed = time.time() - start

            score, metrics = score_audio(audio)
            print(f"Score: {score}, Duration: {metrics['duration']:.2f}s, "
                  f"Energy: {metrics['energy']:.4f}, Time: {elapsed:.1f}s")

            if score > best_score:
                best_score = score
                best_audio = audio.clone()
                best_text = text
                best_metrics = metrics
                print(f"  ^ NEW BEST!")

    if best_audio is None:
        print("\nERROR: No audio generated!")
        return

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    print(f"Text: {best_text}")
    print(f"Score: {best_score}")
    print(f"Duration: {best_metrics['duration']:.2f}s")
    print(f"Energy: {best_metrics['energy']:.4f}")
    print(f"Peak: {best_metrics['peak']:.4f}")
    print(f"Silence: {best_metrics['silence_ratio']*100:.1f}%")

    # Backup old voice prompt
    old_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt.pt')
    if os.path.exists(old_path):
        backup_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt_backup_long.pt')
        os.rename(old_path, backup_path)
        print(f"\nBacked up old prompt to: {backup_path}")

    # Save new optimal voice prompt
    output_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt.pt')
    torch.save({
        'audio': best_audio.cpu(),
        'text': best_text,
        'sample_rate': 24000,
        'duration_seconds': best_metrics['duration'],
        'settings': {
            'temperature': TEMPERATURE,
            'topk': TOPK,
            'speaker': SPEAKER,
        },
        'research_note': 'Optimal prompt: 5-8s duration, matching TTS temp/topk'
    }, output_path)
    print(f"Saved to: {output_path}")

    # Save WAV for listening
    wav_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt.wav')
    torchaudio.save(wav_path, best_audio.cpu().unsqueeze(0), 24000)
    print(f"WAV saved to: {wav_path}")

    print("\n" + "=" * 60)
    print("VOICE PROMPT OPTIMIZATION COMPLETE")
    print(f"Previous: 26.5s @ temp=0.75")
    print(f"New: {best_metrics['duration']:.1f}s @ temp={TEMPERATURE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
