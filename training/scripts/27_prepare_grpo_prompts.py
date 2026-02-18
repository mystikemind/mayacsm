#!/usr/bin/env python3
"""
Prepare GRPO Training Prompts for Orpheus TTS
==============================================

Generate diverse conversational prompts for GRPO training.
GRPO doesn't need ground-truth audio - it generates samples from the model
and scores them with UTMOS to learn what sounds good.

Prompt categories:
1. Short responses (3-8 words) - backchannels, acknowledgments
2. Medium responses (8-20 words) - conversational sentences
3. Long responses (20-40 words) - extended thoughts
4. Emotional expressions - happy, sad, confused, emphatic
5. Questions and statements
6. Common conversational patterns

Target: 2000-4000 diverse prompts covering natural conversation patterns.
"""

import json
import random
from pathlib import Path

# Short conversational responses (1-8 words)
SHORT_RESPONSES = [
    "Mhm.", "Yeah.", "Oh really?", "For sure.", "Right.", "Okay.",
    "That's cool.", "Oh wow.", "Nice.", "Got it.", "Hmm.",
    "Yeah for sure.", "Oh that's great.", "Ah I see.", "Makes sense.",
    "Tell me more.", "No way.", "Oh gosh.", "Totally.", "Exactly.",
    "Yeah definitely.", "Oh interesting.", "Good point.", "Fair enough.",
    "That's true.", "Same here.", "I agree.", "Me too.", "Yep.",
    "Nope.", "Oh cool.", "Oh no.", "Ugh.", "Ha!", "Aww.",
    "Wow that's amazing!", "Oh how sweet!", "That's hilarious!",
    "I love that.", "That's so nice.", "Oh that sucks.", "Bless.",
    "That's rough.", "Oh honey.", "Ha ha, no.", "Wait really?",
    "Seriously?", "Oh stop it.", "Get out.", "Are you kidding?",
    "Oh my god.", "That's wild.", "Shut up.", "Come on.",
    "Yeah I think so.", "Not really.", "Maybe.", "I guess so.",
    "I don't know.", "I mean yeah.", "Kind of.", "Sort of.",
    "I hope so.", "We'll see.", "Probably.", "I doubt it.",
]

# Medium conversational sentences (8-20 words)
MEDIUM_RESPONSES = [
    "Yeah I'm doing pretty good, how's everything with you?",
    "Oh that's so cool, tell me more about that.",
    "Aww that sounds really tough, I'm sorry you're dealing with that.",
    "Oh my gosh that's hilarious, I can't believe that happened!",
    "Hmm yeah that makes sense, I've been thinking about that too.",
    "Honestly I think that's a great idea, you should definitely go for it.",
    "Yeah I know what you mean, sometimes things just don't work out.",
    "Okay cool, thanks for letting me know.",
    "That reminds me of something that happened last week.",
    "Well you know what I think, let's just go for it!",
    "I was just thinking about that the other day actually.",
    "Oh that's really interesting, I never thought about it that way.",
    "Yeah I've been there before, it's not easy but it gets better.",
    "Oh wow, that must have been really exciting for you!",
    "Hmm I'm not sure about that, what do you think?",
    "Yeah I totally get it, no worries at all.",
    "That's so funny because the same thing happened to me.",
    "Oh really, I didn't know that, thanks for telling me.",
    "Yeah we should definitely hang out sometime soon.",
    "I think you're right about that, it makes a lot of sense.",
    "Oh that's awesome, congratulations on that!",
    "Yeah it's been a really busy week for me too.",
    "Hmm that's a tough one, let me think about it.",
    "Oh no, are you okay? That sounds terrible.",
    "Yeah I remember that, it was such a good time.",
    "I mean I get what you're saying but I kinda disagree.",
    "Oh that's sweet of you to say, thank you.",
    "Yeah honestly I've been feeling the same way lately.",
    "That's a great question, I'm glad you brought it up.",
    "Oh for sure, I'd love to help you with that.",
    "Yeah but here's the thing though, it's not that simple.",
    "I know right? It's crazy how fast time goes by.",
    "Oh that makes me so happy to hear that.",
    "Yeah I was gonna say the same thing actually.",
    "Hmm well when you put it that way it does sound better.",
    "Oh come on, don't be so hard on yourself.",
    "Yeah let me look into that and I'll get back to you.",
    "That's really brave of you, I'm proud of you for doing that.",
    "Oh man, that's gotta be frustrating, I'm sorry.",
    "Yeah we can totally do that, what time works for you?",
    "I don't know, something about it just doesn't feel right to me.",
    "Oh that's perfect, that works out really well actually.",
    "Yeah I was thinking the exact same thing, great minds think alike!",
    "Hmm well that's one way to look at it I guess.",
    "Oh really? I had no idea, that's so surprising.",
    "Yeah it's definitely something to think about for sure.",
    "That's such a good point, I never considered that before.",
    "Oh wow, you must be so excited about that!",
    "Yeah I mean at the end of the day, it's your decision.",
    "Hmm I'll have to check my schedule but I think I'm free.",
]

# Longer conversational responses (20-40 words)
LONG_RESPONSES = [
    "You know what, I've been thinking about this a lot lately and I honestly think the best approach is to just take it one step at a time and see what happens.",
    "Oh my gosh, that reminds me of this one time when I was in college and something really similar happened to me and I had absolutely no idea what to do about it.",
    "Yeah I totally understand where you're coming from on that, and I think you make a really valid point, but have you thought about looking at it from a different perspective?",
    "Honestly I think the most important thing right now is that you take care of yourself first and don't worry too much about what everyone else thinks about the situation.",
    "Well here's what I think we should do, let's make a plan for this weekend and figure out all the details so we don't have to stress about it later.",
    "I know it might seem overwhelming right now but trust me, things will work out in the end, they always do, you just have to be patient and keep going.",
    "That's actually a really fascinating topic, I was just reading an article about it the other day and it completely changed the way I think about the whole thing.",
    "Oh absolutely, I couldn't agree with you more on that, it's like everyone seems to forget about the most important part and just focuses on the wrong things.",
    "You know what really bothers me about the whole situation is that nobody seems to want to talk about it openly and honestly, like it's some kind of secret or something.",
    "I mean sure, it's not perfect and there are definitely some things that could be improved, but overall I think it's a pretty solid plan and we should go with it.",
]

# Emotional/expressive sentences
EMOTIONAL_HAPPY = [
    "Oh that is the best news I've heard all day, I'm so happy for you!",
    "Yay, I'm so excited! This is going to be amazing!",
    "That makes me so happy, you have no idea how much that means to me.",
    "I can't stop smiling right now, this is just the best thing ever.",
    "Oh my goodness, that's wonderful! I knew things would work out for you!",
    "This is literally the best day ever, I'm on cloud nine right now!",
    "I'm so thrilled to hear that, you totally deserve it!",
    "Oh wow, that just made my entire week, thank you so much!",
]

EMOTIONAL_SAD = [
    "Oh I'm so sorry to hear that, that must be really hard for you.",
    "That breaks my heart, I wish I could do something to help.",
    "I know this is a really difficult time, and I just want you to know I'm here for you.",
    "That's really sad, I can't imagine how you must be feeling right now.",
    "I feel so bad about what happened, are you doing okay?",
    "It hurts to hear that, I really wish things could be different.",
    "I'm sorry you're going through this, nobody deserves to feel this way.",
    "That's tough, sending you a big hug right now, you deserve it.",
]

EMOTIONAL_CONFUSED = [
    "Wait, what? I'm so confused right now, can you explain that again?",
    "Hmm, I'm not sure I follow, what exactly do you mean by that?",
    "That doesn't make any sense to me, how is that even possible?",
    "I'm lost, can you start from the beginning and walk me through it?",
    "Wait hold on, so you're saying that they actually did that? That's weird.",
    "I thought it was supposed to be the other way around, am I wrong?",
    "Okay now I'm really confused, which one is it supposed to be?",
    "Hmm that's strange, I could have sworn it was different last time.",
]

EMOTIONAL_EMPHATIC = [
    "No but seriously, this is actually really important and we need to talk about it.",
    "Listen, I really think you need to reconsider this, it's a big deal.",
    "I cannot stress this enough, you absolutely have to try this place, it's incredible.",
    "Trust me on this one, I've been doing this for years and I know what I'm talking about.",
    "This is hands down the best thing I've ever experienced, no exaggeration.",
    "I'm telling you, this is going to change everything, mark my words.",
    "Please just hear me out on this, I really think it's worth considering.",
    "Okay but for real though, this is not something we should take lightly.",
]

# Questions
QUESTIONS = [
    "So what do you think about all of that?",
    "Have you ever been in a situation like that before?",
    "What would you do if you were in my shoes?",
    "Do you think that's the right thing to do?",
    "How long have you been thinking about this?",
    "What made you decide to go with that option?",
    "Is there anything else you want to talk about?",
    "Do you want to grab something to eat later?",
    "What's the plan for this weekend?",
    "Have you talked to anyone else about this?",
    "What's the worst that could happen?",
    "Don't you think there might be a better way?",
    "Wait, when did that happen exactly?",
    "How did they react when you told them?",
    "Are you sure about that? It doesn't seem right.",
    "Do you need any help with anything?",
    "What time are you thinking of leaving?",
    "Can you believe they actually said that?",
    "So what's the verdict, are we doing this or not?",
    "How are you feeling about everything lately?",
]

# Storytelling / narrative
NARRATIVE = [
    "So yesterday I was walking down the street and I saw the most random thing, you're not gonna believe this.",
    "Okay so get this, I was at the grocery store and this person comes up to me and says the weirdest thing.",
    "So remember that thing I told you about last week? Well there's been an update and it's kind of wild.",
    "Let me tell you what happened to me this morning, it was honestly the funniest thing ever.",
    "So I finally tried that restaurant you recommended and oh my god, you were so right about it.",
    "I had the craziest dream last night and I just have to tell someone about it.",
    "Okay so the thing about that movie is, you have to watch it twice to really get it, trust me.",
    "So I was talking to my friend about this and she brought up a really interesting point.",
]


def generate_prompts(target_count: int = 3000, seed: int = 42) -> list:
    """Generate diverse conversational prompts."""
    random.seed(seed)

    all_prompts = []

    # Add all curated prompts
    categories = {
        "short": SHORT_RESPONSES,
        "medium": MEDIUM_RESPONSES,
        "long": LONG_RESPONSES,
        "happy": EMOTIONAL_HAPPY,
        "sad": EMOTIONAL_SAD,
        "confused": EMOTIONAL_CONFUSED,
        "emphatic": EMOTIONAL_EMPHATIC,
        "question": QUESTIONS,
        "narrative": NARRATIVE,
    }

    for cat, prompts in categories.items():
        for p in prompts:
            all_prompts.append({"text": p, "category": cat})

    # Augment with variations to reach target count
    base_count = len(all_prompts)
    print(f"Base prompts: {base_count}")

    # Augmentation: slight variations of existing prompts
    augmented = []

    # Add "well" / "oh" / "hmm" prefixes
    prefixes = ["Well ", "Oh ", "Hmm ", "Yeah ", "So ", "I mean ", "Like ", "Honestly ", "Actually "]
    for prompt in list(all_prompts):
        if len(augmented) + base_count >= target_count:
            break
        prefix = random.choice(prefixes)
        if not prompt["text"].startswith(prefix.strip()):
            new_text = prefix + prompt["text"][0].lower() + prompt["text"][1:]
            augmented.append({"text": new_text, "category": prompt["category"]})

    all_prompts.extend(augmented)

    # Shuffle and truncate
    random.shuffle(all_prompts)
    if len(all_prompts) > target_count:
        all_prompts = all_prompts[:target_count]

    print(f"Total prompts: {len(all_prompts)}")

    # Print category distribution
    from collections import Counter
    cats = Counter(p["category"] for p in all_prompts)
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return all_prompts


def format_for_orpheus(prompts: list, voice_name: str = "maya") -> list:
    """Format prompts for Orpheus model input.

    Format: <|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>
    Token IDs: BOS=128000, START_HUMAN=128259, END_HUMAN=128260, START_AI=128261
    """
    formatted = []
    for p in prompts:
        formatted.append({
            "text": p["text"],
            "category": p["category"],
            "voice": voice_name,
            # The actual token formatting happens in the GRPO trainer
            "prompt_text": f"{voice_name}: {p['text']}",
        })
    return formatted


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=3000)
    parser.add_argument("--voice", default="maya")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output or str(
        Path("/home/ec2-user/SageMaker/project_maya/training/data/grpo_prompts.json")
    )

    print("=" * 60)
    print("  GRPO Prompt Generation")
    print("=" * 60)

    prompts = generate_prompts(target_count=args.target)
    formatted = format_for_orpheus(prompts, voice_name=args.voice)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2)

    print(f"\nSaved {len(formatted)} prompts to {output_path}")

    # Also save a simple text file for easy viewing
    txt_path = output_path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        for p in formatted:
            f.write(p["prompt_text"] + "\n")
    print(f"Saved text version to {txt_path}")


if __name__ == "__main__":
    main()
