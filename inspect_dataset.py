"""
Week 2 -- Inspect the Dataset
================================
Quick sanity checks on dataset.csv before it gets used for NLP/ML in
Weeks 3-5: how many examples per intent, any duplicates, a sample of
rows. Cheap to run, catches labeling mistakes early.
"""

import csv
from collections import Counter
from pathlib import Path


def load_rows(path: str = "dataset.csv") -> list[tuple[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(row["text"], row["intent"]) for row in reader]


def inspect(path: str = "dataset.csv"):
    rows = load_rows(path)
    texts = [t for t, _ in rows]
    intents = [i for _, i in rows]

    print("=" * 60)
    print("WEEK 2 DATASET INSPECTION")
    print("=" * 60)
    print(f"\nTotal examples: {len(rows)}")

    print("\nExamples per intent:")
    counts = Counter(intents)
    for intent, count in sorted(counts.items()):
        print(f"  - {intent:<16} {count}")

    dupes = [t for t, c in Counter(texts).items() if c > 1]
    print(f"\nDuplicate phrasings: {len(dupes)}")
    if dupes:
        for d in dupes:
            print(f"  ! {d!r}")

    print("\nSample rows:")
    for text, intent in rows[:5]:
        print(f'  ("{text}", {intent})')


if __name__ == "__main__":
    inspect()
