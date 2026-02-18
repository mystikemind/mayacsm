#!/usr/bin/env python3
"""
Generate a long (2-3 minute) voice prompt for Maya.

Research finding: "Optimal voice prompt length is 2-3 minutes...
Content should be conversational (not read speech), include various
intonations and emotional variety."

Strategy:
1. Generate multiple conversational segments
2. Use previous segments as context for voice consistency
3. Select best segments using audio quality metrics
4. Concatenate into final 2-3 minute prompt
"""

import sys
import os
import torch
import torchaudio
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

# Output directory
OUTPUT_DIR = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Conversational segments with emotional variety
# These should be natural, varied in tone and length
SEGMENTS = [
    # Warm greetings
    "hi there its really nice to meet you im maya",
    "oh hey how are you doing today",
    "welcome back its so good to hear from you again",

    # Interested responses
    "oh thats really interesting tell me more about that",
    "wow i didnt know that thats fascinating",
    "hmm let me think about that for a moment",

    # Empathetic responses
    "i totally understand what youre going through",
    "that sounds really challenging im sorry to hear that",
    "i can tell this means a lot to you",

    # Thoughtful responses
    "you know thats a great point i hadnt considered that",
    "well i think there are a few ways to look at this",
    "let me share my thoughts on that",

    # Encouraging responses
    "youre doing great keep it up",
    "i believe in you you can definitely do this",
    "thats amazing im so happy for you",

    # Casual conversation
    "so whats been on your mind lately",
    "tell me something interesting about your day",
    "i love having these conversations with you",

    # Questions
    "what do you think about that",
    "how does that make you feel",
    "would you like to hear more",

    # Varied intonation
    "really thats so cool",
    "no way are you serious",
    "aww thats so sweet",
    "yeah i totally agree with you",
    "hmm interesting point",

    # More conversational variety (to reach 2-3 min)
    "oh that reminds me of something",
    "you know what i was thinking",
    "well from my perspective",
    "i think that makes a lot of sense",
    "lets explore that a bit more",
    "thats a really good question",
    "i appreciate you sharing that with me",
    "you bring up an excellent point",
    "thats something worth considering",
    "i hadnt thought about it that way before",
    "that gives me something to think about",
    "i can see where youre coming from",
    "thats really thoughtful of you",
    "im glad we can talk about this",
    "you always have such interesting ideas",
    "i love learning new things from you",
    "this is such a great conversation",
    "tell me more about what you mean",
    "i find that absolutely fascinating",
    "you have a wonderful way of explaining things",
]


def load_csm():
    """Load CSM model."""
    from generator import load_csm_1b
    print("Loading CSM-1B...")
    return load_csm_1b(device="cuda")


def generate_segment(generator, text, context=[], temperature=0.8, topk=60):
    """Generate a single audio segment."""
    # Shorter max to prevent padding/silence
    word_count = len(text.split())
    max_ms = min(max(word_count * 400, 2000), 5000)  # ~400ms per word

    audio = generator.generate(
        text=text,
        speaker=0,
        context=context,
        max_audio_length_ms=max_ms,
        temperature=temperature,
        topk=topk,
    )
    return audio


def calculate_quality_score(audio):
    """
    Simple quality score based on:
    - Energy consistency (not too quiet or loud)
    - No extreme peaks (clipping)
    - Reasonable duration
    """
    if len(audio) < 12000:  # < 0.5s
        return 0.0

    # Check for clipping
    if torch.max(torch.abs(audio)) > 0.98:
        return 0.3

    # Check energy
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms < 0.005 or rms > 0.6:
        return 0.4

    # Check for silence (be more lenient)
    silence_ratio = torch.sum(torch.abs(audio) < 0.01).item() / len(audio)
    if silence_ratio > 0.7:
        return 0.5

    return 1.0


def main():
    print("=" * 60)
    print("GENERATING LONG VOICE PROMPT (2-3 minutes)")
    print("=" * 60)
    print()

    generator = load_csm()

    # Import Segment for context
    from generator import Segment

    # Generate all segments, building context as we go
    generated_segments = []
    context = []
    target_duration = 150  # 2.5 minutes in seconds
    total_duration = 0

    print(f"Target duration: {target_duration}s")
    print(f"Generating {len(SEGMENTS)} segments...")
    print()

    for i, text in enumerate(SEGMENTS):
        if total_duration >= target_duration:
            print(f"Reached target duration ({total_duration:.1f}s)")
            break

        print(f"[{i+1}/{len(SEGMENTS)}] Generating: '{text[:40]}...'")
        start = time.time()

        # Generate with context for consistency
        audio = generate_segment(
            generator,
            text,
            context=context[-3:] if context else [],  # Use last 3 segments as context
            temperature=0.75,  # Slightly lower for consistency
            topk=60
        )

        elapsed = time.time() - start
        duration = len(audio) / 24000
        quality = calculate_quality_score(audio)

        print(f"  Duration: {duration:.1f}s, Quality: {quality:.1f}, Time: {elapsed:.1f}s")

        if quality >= 0.5:
            # Save segment for context
            segment = Segment(speaker=0, text=text, audio=audio)
            context.append(segment)

            generated_segments.append({
                'text': text,
                'audio': audio.cpu(),
                'duration': duration,
                'quality': quality
            })
            total_duration += duration
        else:
            print(f"  Skipping low quality segment")

        print()

    if not generated_segments:
        print("ERROR: No segments generated!")
        return

    print("=" * 60)
    print(f"Generated {len(generated_segments)} segments, total: {total_duration:.1f}s")
    print("=" * 60)

    # Concatenate all segments with small gaps
    gap = torch.zeros(int(24000 * 0.3))  # 300ms gap between segments

    audio_parts = []
    all_text = []

    for seg in generated_segments:
        audio_parts.append(seg['audio'])
        audio_parts.append(gap)
        all_text.append(seg['text'])

    # Remove last gap
    audio_parts = audio_parts[:-1]

    final_audio = torch.cat(audio_parts)
    final_text = " ... ".join(all_text)
    final_duration = len(final_audio) / 24000

    print(f"Final audio duration: {final_duration:.1f}s ({final_duration/60:.1f} minutes)")
    print()

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt_long.pt')
    torch.save({
        'audio': final_audio,
        'text': final_text,
        'sample_rate': 24000,
        'duration_seconds': final_duration,
        'num_segments': len(generated_segments),
        'settings': {
            'temperature': 0.75,
            'topk': 60,
            'speaker': 0,
        }
    }, output_path)
    print(f"Saved to: {output_path}")

    # Also save as WAV for listening
    wav_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt_long.wav')
    torchaudio.save(wav_path, final_audio.unsqueeze(0), 24000)
    print(f"WAV saved to: {wav_path}")

    # Update the main voice prompt path to use the long one
    main_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt.pt')
    torch.save({
        'audio': final_audio,
        'text': final_text,
        'sample_rate': 24000,
        'duration_seconds': final_duration,
        'num_segments': len(generated_segments),
        'settings': {
            'temperature': 0.75,
            'topk': 60,
            'speaker': 0,
        }
    }, main_path)
    print(f"Also saved to main path: {main_path}")

    print()
    print("=" * 60)
    print("VOICE PROMPT GENERATION COMPLETE")
    print(f"Duration: {final_duration:.1f}s ({final_duration/60:.1f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
