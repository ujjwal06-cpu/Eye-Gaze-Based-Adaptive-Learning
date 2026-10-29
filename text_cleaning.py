"""
Week 3 -- Text Cleaning
=========================
First stage of the NLP pipeline: Input -> [Text Cleaning] -> ...

Goal: strip out noise that doesn't help distinguish one intent from
another (casing, punctuation, extra whitespace) while keeping the
actual words intact. Nothing fancy on purpose -- this stage should be
boring and predictable, since every later stage depends on it being
consistent.
"""

import re


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace.

    Example:
        "I need Water, PLEASE!!"  ->  "i need water please"
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # drop punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()  # collapse repeated whitespace
    return text


if __name__ == "__main__":
    samples = [
        "I need Water, PLEASE!!",
        "  I'm   really thirsty right now...  ",
        "Can I have some water?",
    ]
    for s in samples:
        print(f"{s!r:45} -> {clean_text(s)!r}")
