#!/usr/bin/env python3
"""
Generate a SESAME-LEVEL voice prompt for Maya (2-3 minutes).

Based on Sesame AI research:
- Duration: 2-3 minutes of varied conversational speech
- Content: NOT read speech - actual conversational patterns
- Variety: Different emotions, intonations, sentence types
- Natural disfluencies: um, uh, hmm, like, you know

This creates the voice identity that makes Maya sound human.
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

# Sesame-style conversational segments with emotional variety
# Each segment captures a different aspect of natural human speech
SEGMENTS = [
    # ===== WARM GREETINGS (establish friendly baseline) =====
    "oh hey! yeah im doing pretty good, you know, just hanging out",
    "hi there! its so nice to finally talk to you",
    "oh hi! hmm let me think, yeah i think im free to chat",

    # ===== THINKING/PROCESSING (natural disfluencies) =====
    "hmm thats actually a really good question, let me think about that",
    "um well i guess the thing is, like, it depends on how you look at it",
    "oh thats interesting, hmm, yeah i think i know what you mean",
    "well um, you know, i hadnt really thought about it that way before",

    # ===== EMOTIONAL REACTIONS (varied energy) =====
    "wait really? no way! thats so cool, tell me more",
    "oh wow thats amazing! im so happy for you",
    "aw man that sounds really rough, im sorry youre going through that",
    "ooh yeah i totally get that feeling, ive been there too",
    "huh, thats kind of weird actually, dont you think?",

    # ===== EMPATHETIC RESPONSES (warm, caring) =====
    "yeah i can totally understand why youd feel that way",
    "oh that must have been so hard, how are you doing now?",
    "im here for you, you know, whatever you need",
    "aww thats really sweet of you to say, thank you",

    # ===== CURIOUS/INTERESTED (engaging) =====
    "ooh thats fascinating, how did you figure that out?",
    "wait so what happened next? im so curious now",
    "oh really? i didnt know that, thats so interesting",
    "hmm tell me more about that, it sounds really cool",

    # ===== AGREEMENT/VALIDATION (supportive) =====
    "yeah exactly! thats what i was thinking too",
    "oh for sure, i completely agree with you on that",
    "mhm mhm, yeah that makes total sense to me",
    "right right, i see what youre saying now",

    # ===== CASUAL CONVERSATION (relaxed) =====
    "so yeah anyway, what have you been up to lately?",
    "oh nice nice, that sounds like a good time",
    "haha yeah i know what you mean, its like that sometimes",
    "well you know how it is, some days are just like that",

    # ===== THOUGHTFUL RESPONSES (deeper conversation) =====
    "you know, i think the thing that matters most is how it makes you feel",
    "hmm well if i had to choose, id probably say the second one",
    "i mean at the end of the day, you gotta do whats right for you",
    "thats a tough one, let me think about it for a sec",

    # ===== EXCLAMATIONS (varied energy/emotion) =====
    "oh my gosh yes! i love that so much",
    "whoa thats crazy! i cant believe that happened",
    "ugh i know right? its so frustrating sometimes",
    "yay! thats such great news, im so excited for you",

    # ===== QUESTIONS (natural inquiry) =====
    "so what do you think about all that?",
    "how does that make you feel though?",
    "wait have you tried doing it this way instead?",
    "do you wanna talk more about it?",

    # ===== TRANSITIONS (conversational flow) =====
    "oh that reminds me of something actually",
    "speaking of which, did you hear about that thing?",
    "anyway um where were we? oh right yeah",
    "but yeah so what i was saying before is",

    # ===== CLOSING WARMTH (friendly endings) =====
    "this has been really nice, i love talking with you",
    "yeah we should definitely do this again sometime",
    "take care of yourself okay? talk to you soon",
    "alright well it was great chatting, bye for now!",
]

# CSM settings that MATCH tts_streaming_real.py
TEMPERATURE = 0.9
TOPK = 50
SPEAKER = 0


def load_csm():
    """Load CSM model."""
    from generator import load_csm_1b
    print("Loading CSM-1B...")
    start = time.time()
    model = load_csm_1b(device="cuda")
    print(f"Model loaded in {time.time()-start:.1f}s")
    return model


def generate_segment(generator, text, context=[], temperature=TEMPERATURE, topk=TOPK):
    """Generate a single audio segment with context."""
    word_count = len(text.split())
    max_ms = min(max(word_count * 450, 2500), 8000)  # ~450ms per word, natural pacing

    audio = generator.generate(
        text=text,
        speaker=SPEAKER,
        context=context,
        max_audio_length_ms=max_ms,
        temperature=temperature,
        topk=topk,
    )
    return audio


def score_audio(audio, min_duration=1.5, max_duration=6.0):
    """Score audio quality."""
    duration = len(audio) / 24000
    score = 0

    # Duration scoring
    if min_duration <= duration <= max_duration:
        score += 30
    elif duration < min_duration:
        score += 10  # Too short
    else:
        score += 15  # Too long but usable

    # Energy check
    energy = audio.abs().mean().item()
    if 0.02 < energy < 0.08:
        score += 25  # Good energy
    elif 0.01 < energy < 0.12:
        score += 15  # Acceptable

    # Clipping check
    peak = audio.abs().max().item()
    if peak < 0.9:
        score += 20  # No clipping
    elif peak < 0.98:
        score += 10  # Minor clipping

    # Silence ratio
    silence_ratio = (audio.abs() < 0.01).sum().item() / len(audio)
    if silence_ratio < 0.25:
        score += 25  # Good
    elif silence_ratio < 0.4:
        score += 15  # Acceptable

    return score, duration, energy, peak


def main():
    print("=" * 70)
    print("GENERATING SESAME-LEVEL VOICE PROMPT (2-3 minutes)")
    print("=" * 70)
    print(f"\nTarget: 2-3 minutes of varied conversational speech")
    print(f"Segments: {len(SEGMENTS)}")
    print(f"Temperature: {TEMPERATURE} (matches TTS)")
    print(f"TopK: {TOPK} (matches TTS)")
    print()

    generator = load_csm()

    # Import Segment for context
    from generator import Segment

    generated = []
    context = []
    total_duration = 0
    target_duration = 150  # 2.5 minutes target

    print("Generating segments with voice continuity...")
    print()

    for i, text in enumerate(SEGMENTS):
        if total_duration >= target_duration:
            print(f"\n>>> Reached target duration ({total_duration:.1f}s) <<<")
            break

        print(f"[{i+1}/{len(SEGMENTS)}] '{text[:45]}...'")
        start = time.time()

        # Generate with context for voice continuity
        best_audio = None
        best_score = -1

        # Try 2 takes per segment, pick best
        for take in range(2):
            audio = generate_segment(
                generator, text,
                context=context[-3:] if context else [],  # Last 3 for continuity
                temperature=TEMPERATURE,
                topk=TOPK
            )
            score, duration, energy, peak = score_audio(audio)

            if score > best_score:
                best_score = score
                best_audio = audio.clone()
                best_duration = duration

        elapsed = time.time() - start
        print(f"    Duration: {best_duration:.1f}s, Score: {best_score}, Time: {elapsed:.1f}s")

        if best_score >= 40:  # Quality threshold
            # Add to context for voice continuity
            segment = Segment(speaker=SPEAKER, text=text, audio=best_audio)
            context.append(segment)

            generated.append({
                'text': text,
                'audio': best_audio.cpu(),
                'duration': best_duration,
                'score': best_score
            })
            total_duration += best_duration
        else:
            print(f"    [SKIP] Low quality")

    if not generated:
        print("\nERROR: No segments generated!")
        return

    print()
    print("=" * 70)
    print(f"Generated {len(generated)} segments, total: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print("=" * 70)

    # Concatenate with small natural gaps
    gap = torch.zeros(int(24000 * 0.25))  # 250ms gap (natural pause)

    audio_parts = []
    all_text = []
    for seg in generated:
        audio_parts.append(seg['audio'])
        audio_parts.append(gap)
        all_text.append(seg['text'])

    # Remove last gap
    audio_parts = audio_parts[:-1]

    final_audio = torch.cat(audio_parts)
    final_text = " ... ".join(all_text)
    final_duration = len(final_audio) / 24000

    print(f"\nFinal voice prompt: {final_duration:.1f}s ({final_duration/60:.1f} minutes)")

    # Backup old prompt
    old_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt.pt')
    if os.path.exists(old_path):
        backup_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt_backup_phase2.pt')
        import shutil
        shutil.copy(old_path, backup_path)
        print(f"Backed up previous prompt to: {backup_path}")

    # Save new comprehensive voice prompt
    torch.save({
        'audio': final_audio,
        'text': final_text,
        'sample_rate': 24000,
        'duration_seconds': final_duration,
        'num_segments': len(generated),
        'settings': {
            'temperature': TEMPERATURE,
            'topk': TOPK,
            'speaker': SPEAKER,
        },
        'version': 'sesame_level_v1',
        'description': 'Comprehensive 2-3min voice prompt with emotional variety'
    }, old_path)
    print(f"Saved to: {old_path}")

    # Save WAV for listening
    wav_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt.wav')
    torchaudio.save(wav_path, final_audio.unsqueeze(0), 24000)
    print(f"WAV saved to: {wav_path}")

    # Also save the long version separately
    long_path = os.path.join(OUTPUT_DIR, 'maya_voice_prompt_sesame.pt')
    torch.save({
        'audio': final_audio,
        'text': final_text,
        'sample_rate': 24000,
        'duration_seconds': final_duration,
        'num_segments': len(generated),
        'settings': {
            'temperature': TEMPERATURE,
            'topk': TOPK,
            'speaker': SPEAKER,
        },
        'version': 'sesame_level_v1',
    }, long_path)
    print(f"Also saved to: {long_path}")

    print()
    print("=" * 70)
    print("SESAME-LEVEL VOICE PROMPT COMPLETE")
    print(f"Duration: {final_duration:.1f}s ({final_duration/60:.1f} minutes)")
    print(f"Segments: {len(generated)} high-quality segments")
    print("=" * 70)


if __name__ == "__main__":
    main()
