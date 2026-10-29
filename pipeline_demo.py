"""
Week 3 -- Full NLP Pipeline Demo
===================================
    Input -> Text Cleaning -> Tokenization -> Feature Extraction -> Intent Classification

This wires all three earlier stages together on the real Week 2
dataset, then adds ONE simple baseline classifier (Multinomial Naive
Bayes) just to prove the full loop actually works end-to-end.

IMPORTANT: this is a placeholder classifier, not the real evaluation.
Week 4 is where Naive Bayes / Logistic Regression / SVM / Random
Forest get properly compared with accuracy/precision/recall/F1 and a
confusion matrix. Treat the accuracy number printed here as "does the
pipeline work at all", not as a real result to report.
"""

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from text_cleaning import clean_text
from tokenization import tokenize
from feature_extraction import build_vectorizer


def load_dataset(path: str = "dataset.csv") -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["intent"])
    return texts, labels


def run_pipeline_demo():
    print("=" * 70)
    print("WEEK 3 DEMO -- Input -> Cleaning -> Tokenization -> Features -> Classify")
    print("=" * 70)

    # ---- Stage-by-stage walkthrough on one example ------------------
    example = "I'm REALLY thirsty, can I have some Water??"
    cleaned = clean_text(example)
    tokens = tokenize(cleaned)
    print(f"\nRaw input:      {example!r}")
    print(f"Cleaned text:   {cleaned!r}")
    print(f"Tokens:         {tokens}")

    # ---- Load Week 2's dataset ---------------------------------------
    texts, labels = load_dataset()
    print(f"\nLoaded {len(texts)} labeled examples from dataset.csv")

    # ---- Feature extraction (TF-IDF) --------------------------------
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    print(f"Feature matrix shape: {X.shape}  "
          f"({X.shape[0]} examples x {X.shape[1]} vocabulary terms)")

    # ---- Baseline classifier (placeholder -- Week 4 does the real comparison) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n[Baseline Naive Bayes] Test accuracy: {acc:.2f} "
          f"(placeholder only -- Week 4 compares this properly)")

    # ---- Classify the earlier example end-to-end ---------------------
    example_vec = vectorizer.transform([example])
    predicted_intent = clf.predict(example_vec)[0]
    print(f"\nEnd-to-end classification of: {example!r}")
    print(f"  -> Predicted intent: {predicted_intent}")


if __name__ == "__main__":
    run_pipeline_demo()
