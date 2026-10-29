"""
Week 6 -- Full Pipeline Demo: Text -> Intent -> Message
===========================================================
    Simulated Gaze Selection (or typed text) -> ... -> Intent Classification -> Message Generation

Trains the Week 4 winner (SVM, linear kernel) on the FULL dataset
(no cross-validation split needed here -- CV was for evaluating
accuracy in Week 4; for actual inference we want the model trained on
every example we have), then classifies new input and generates a
message for it. This is the closest thing to your final system
working end-to-end so far.
"""

import csv
from pathlib import Path

from sklearn.svm import SVC

from feature_extraction import build_vectorizer
from intents import IntentName
from message_templates import generate_message


def load_dataset(path: str = "dataset.csv"):
    texts, labels = [], []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["intent"])
    return texts, labels


def train_classifier():
    """Train the Week 4 winner (SVM) on the full dataset for inference use."""
    texts, labels = load_dataset()
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    clf = SVC(kernel="linear")
    clf.fit(X, labels)
    return vectorizer, clf


def classify_and_generate(text: str, vectorizer, clf) -> tuple[str, str]:
    """Run one input all the way through: text -> intent -> message."""
    vec = vectorizer.transform([text])
    predicted_intent = clf.predict(vec)[0]
    message = generate_message(IntentName(predicted_intent))
    return predicted_intent, message


def run_demo():
    print("=" * 70)
    print("WEEK 6 DEMO -- Text -> Intent Classification -> Message Generation")
    print("=" * 70)

    vectorizer, clf = train_classifier()

    test_inputs = [
        "I'm so thirsty right now",
        "please help me I need someone",
        "it hurts so much",
        "can you call my mom",
        "I want to watch a movie",
        "this is really urgent, please help now",
    ]

    for text in test_inputs:
        intent, message = classify_and_generate(text, vectorizer, clf)
        print(f"\nInput:            \"{text}\"")
        print(f"Predicted intent: {intent}")
        print(f"Generated message: \"{message}\"")


if __name__ == "__main__":
    run_demo()
