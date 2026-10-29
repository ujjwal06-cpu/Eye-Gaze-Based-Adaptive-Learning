"""
Week 2 -- Build the Communication Dataset
===========================================
Roadmap: "Create multiple variations for each intent."

This module holds hand-written example phrasings for each of the 10
intents defined in Week 1 (intents.py), and writes them out to a CSV
file with columns: text, intent.

Weeks 3-5 (NLP cleaning, ML classifiers, DistilBERT) will all train
and evaluate on this file, so the main thing that matters here is
covering realistic *variety* per intent -- different sentence
lengths, phrasings, and levels of directness -- not just paraphrasing
the same sentence ten times.
"""

import csv
from pathlib import Path

from intents import IntentName

# ---------------------------------------------------------------------------
# Example phrasings per intent. ~12 each, deliberately varied in
# length/directness so the classifier doesn't just learn one template.
# ---------------------------------------------------------------------------
EXAMPLES: dict[IntentName, list[str]] = {
    IntentName.WATER: [
        "I need water",
        "Please give me water",
        "I am thirsty",
        "Can I have some water",
        "I want to drink water",
        "Water please",
        "I need a glass of water",
        "I'm really thirsty right now",
        "Could you bring me water",
        "Give me water please",
        "I would like some water",
        "I need something to drink",
    ],
    IntentName.FOOD: [
        "I need food",
        "I am hungry",
        "Please give me food",
        "I want something to eat",
        "Can I have food",
        "Food please",
        "I'm starving",
        "I would like to eat something",
        "Could you bring me food",
        "I need a meal",
        "I want a snack",
        "Give me something to eat",
    ],
    IntentName.HELP: [
        "I need help",
        "Please help me",
        "Can you help me",
        "I need assistance",
        "Help please",
        "Someone help me",
        "I require assistance",
        "Can someone come help",
        "I need a hand with something",
        "Please assist me",
        "I need support right now",
        "Come help me please",
    ],
    IntentName.CALL: [
        "Please call my caregiver",
        "Call someone for me",
        "I want you to call my family",
        "Can you call the nurse",
        "Please contact my caregiver",
        "Call my mom",
        "I need you to make a call",
        "Please phone my caregiver",
        "Contact someone for me",
        "Call for assistance",
        "Can you ring my sister",
        "I need you to reach someone",
    ],
    IntentName.EMERGENCY: [
        "This is an emergency",
        "I need help immediately",
        "Emergency, please help",
        "Call for emergency help now",
        "This is urgent",
        "I need urgent help",
        "Please help right away",
        "Emergency situation",
        "I need help right now, it's urgent",
        "Something is seriously wrong",
        "This can't wait, emergency",
        "Get help now, please",
    ],
    IntentName.MEDICINE: [
        "I need my medicine",
        "Please give me my medication",
        "I need my pills",
        "Medicine please",
        "Can I have my medication",
        "I need my prescription",
        "It's time for my medicine",
        "I require my medication now",
        "Please bring my pills",
        "I need my dose",
        "Can you get my medicine",
        "I missed my medication, I need it",
    ],
    IntentName.TOILET: [
        "I need to use the restroom",
        "I need the bathroom",
        "Please take me to the toilet",
        "I need to use the toilet",
        "Bathroom please",
        "I need to go to the washroom",
        "Can you help me to the restroom",
        "I need to pee",
        "Take me to the bathroom please",
        "I need to use the loo",
        "I need to go, bathroom now",
        "Can you help me get to the toilet",
    ],
    IntentName.PAIN: [
        "I am in pain",
        "Something hurts",
        "I'm in a lot of pain",
        "Please check on me, it hurts",
        "I have pain",
        "My body hurts",
        "I'm hurting",
        "I feel pain",
        "This really hurts",
        "I need something for the pain",
        "It hurts a lot right now",
        "I'm not feeling well, it hurts",
    ],
    IntentName.ENTERTAINMENT: [
        "Can you turn on the TV",
        "I want to watch something",
        "Play some music please",
        "I want entertainment",
        "Turn on the radio",
        "I'm bored, play something",
        "Can we watch a movie",
        "Put on some music",
        "I want to listen to music",
        "Turn on the television please",
        "Can you play my favorite show",
        "I'd like some music on",
    ],
    IntentName.GENERAL_MESSAGE: [
        "I have something to say",
        "I want to tell you something",
        "Let me say something",
        "I need to communicate something",
        "I have a message",
        "There's something I want to say",
        "I want to talk",
        "Let me express something",
        "I have a general message",
        "I want to say something else",
        "There's something on my mind",
        "I need to tell you something important",
    ],
}


def build_rows() -> list[tuple[str, str]]:
    """Flatten EXAMPLES into (text, intent) rows."""
    rows = []
    for intent, phrasings in EXAMPLES.items():
        for text in phrasings:
            rows.append((text, intent.value))
    return rows


def write_csv(path: str = "dataset.csv") -> int:
    """Write the dataset to a CSV file with columns: text, intent.

    Returns the number of rows written.
    """
    rows = build_rows()
    out_path = Path(path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    n = write_csv()
    print(f"Wrote {n} labeled examples across {len(EXAMPLES)} intents -> dataset.csv")
