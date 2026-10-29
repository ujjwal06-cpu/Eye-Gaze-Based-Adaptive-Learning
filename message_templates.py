"""
Week 6 -- Message Generation
==============================
Roadmap:
    Intent = WATER  -> "I need some water, please."
    Intent = HELP   -> "I need assistance. Please come here."

Why variation matters: an AAC user might select the same intent many
times a day (WATER could easily happen 10+ times). If the system says
the exact identical sentence every time, it sounds robotic and gives
listeners no sense of urgency/tone variation. So each intent gets a
small set of natural phrasings instead of one fixed template -- this
also sets up Week 7 nicely, since personalization can eventually
prefer whichever phrasing a specific user responds best to.
"""

import random

from intents import IntentName

TEMPLATES: dict[IntentName, list[str]] = {
    IntentName.WATER: [
        "I need some water, please.",
        "Could I have a glass of water?",
        "I'm thirsty, please bring me water.",
    ],
    IntentName.FOOD: [
        "I need some food, please.",
        "Could I have something to eat?",
        "I'm hungry, please bring me food.",
    ],
    IntentName.HELP: [
        "I need assistance. Please come here.",
        "Please help me right now.",
        "I could use some help, please.",
    ],
    IntentName.CALL: [
        "Please call my caregiver.",
        "Could you contact someone for me?",
        "I need you to make a call, please.",
    ],
    IntentName.EMERGENCY: [
        "This is an emergency. I need immediate help.",
        "Please help me right away, this is urgent.",
        "Emergency -- I need assistance immediately.",
    ],
    IntentName.MEDICINE: [
        "I need my medicine now.",
        "Could you bring me my medication, please?",
        "It's time for my medicine.",
    ],
    IntentName.TOILET: [
        "I need to use the restroom.",
        "Could you take me to the bathroom, please?",
        "I need help getting to the toilet.",
    ],
    IntentName.PAIN: [
        "I am in pain. Please check on me.",
        "Something hurts, please help.",
        "I need help, I'm in pain.",
    ],
    IntentName.ENTERTAINMENT: [
        "Could you turn on some entertainment for me?",
        "I'd like to watch or listen to something.",
        "Can you play some music, please?",
    ],
    IntentName.GENERAL_MESSAGE: [
        "I have something to say.",
        "There's something I'd like to tell you.",
        "I need to communicate something.",
    ],
}


def generate_message(intent: IntentName, variation: int | None = None) -> str:
    """Turn an intent into a natural sentence.

    Args:
        intent: The classified intent.
        variation: Optionally pick a specific template by index
            (useful for testing / reproducibility). If omitted, a
            random variation is chosen -- so repeated real-world use
            of the same intent doesn't sound identical every time.
    """
    options = TEMPLATES[intent]
    if variation is not None:
        return options[variation % len(options)]
    return random.choice(options)


if __name__ == "__main__":
    for intent in [IntentName.WATER, IntentName.HELP, IntentName.EMERGENCY]:
        print(f"{intent.value}:")
        for v in range(len(TEMPLATES[intent])):
            print(f"  [{v}] {generate_message(intent, variation=v)}")
