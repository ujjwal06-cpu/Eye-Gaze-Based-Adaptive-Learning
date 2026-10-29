"""
Week 4 -- ML Intent Classification: Compare 4 Models
=======================================================
Roadmap: compare Naive Bayes, Logistic Regression, SVM, Random Forest.
Measure accuracy, precision, recall, F1-score, confusion matrix.

METHODOLOGY NOTE: with only 12 examples per intent (120 total), a
single train/test split leaves ~3 test examples per class -- too
small to trust. Instead we use 5-fold STRATIFIED CROSS-VALIDATION via
cross_val_predict: every example gets predicted exactly once, by a
model that never saw it during training for that fold. This gives
metrics computed across the full dataset instead of a noisy 25%
slice, which is the more defensible choice to show a professor when
justifying your numbers.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import build_vectorizer


def load_dataset(path: str = "dataset.csv") -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["intent"])
    return texts, labels


MODELS = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (linear)": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}


def evaluate_model(name, model, X, y, cv):
    """Run 5-fold CV, return metrics dict + confusion matrix."""
    y_pred = cross_val_predict(model, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=sorted(set(y)))
    report = classification_report(y, y_pred, zero_division=0)
    return {
        "name": name, "accuracy": acc, "precision": precision,
        "recall": recall, "f1": f1, "confusion_matrix": cm, "report": report,
    }


def print_confusion_matrix(cm, labels):
    header = "".join(f"{l[:4]:>6}" for l in labels)
    print(f"{'':16}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>6}" for v in row)
        print(f"{labels[i]:16}{row_str}")


def plot_comparison(results, out_path="model_comparison.png"):
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, f1s, width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Week 4 -- Classifier Comparison (5-fold CV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved comparison chart -> {out_path}")


def save_summary_csv(results, out_path="model_comparison.csv"):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
        for r in results:
            writer.writerow([r["name"], f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                              f"{r['recall']:.4f}", f"{r['f1']:.4f}"])
    print(f"Saved metrics table -> {out_path}")


def run():
    print("=" * 70)
    print("WEEK 4 -- Comparing Naive Bayes / Logistic Regression / SVM / Random Forest")
    print("=" * 70)

    texts, labels = load_dataset()
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    label_list = sorted(set(labels))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name, model in MODELS.items():
        print(f"\n{'-'*70}\n{name}\n{'-'*70}")
        res = evaluate_model(name, model, X, labels, cv)
        results.append(res)
        print(f"Accuracy:  {res['accuracy']:.3f}")
        print(f"Precision (macro): {res['precision']:.3f}")
        print(f"Recall (macro):    {res['recall']:.3f}")
        print(f"F1 (macro):        {res['f1']:.3f}")
        print("\nConfusion matrix (rows=true, cols=predicted):")
        print_confusion_matrix(res["confusion_matrix"], label_list)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    for r in sorted(results, key=lambda r: -r["f1"]):
        print(f"{r['name']:<22}{r['accuracy']:>10.3f}{r['precision']:>12.3f}"
              f"{r['recall']:>10.3f}{r['f1']:>10.3f}")

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest model by macro F1: {best['name']} ({best['f1']:.3f})")

    save_summary_csv(results)
    plot_comparison(results)


if __name__ == "__main__":
    run()
