"""
Week 3 -- Tokenization
========================
Second stage: Text Cleaning -> [Tokenization] -> Feature Extraction

Splits cleaned text into individual word tokens. Assumes input has
already been through clean_text() (lowercased, punctuation removed),
so this stage can stay a simple whitespace split.
"""

from text_cleaning import clean_text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens.

    Example:
        "i need water please" -> ["i", "need", "water", "please"]
    """
    return text.split()


def clean_and_tokenize(raw_text: str) -> list[str]:
    """Convenience: run cleaning then tokenization in one call."""
    return tokenize(clean_text(raw_text))


if __name__ == "__main__":
    samples = [
        "I need Water, PLEASE!!",
        "Please give me my medication",
        "This is an emergency!",
    ]
    for s in samples:
        print(f"{s!r:35} -> {clean_and_tokenize(s)}")
