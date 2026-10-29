"""
Week 5 -- DistilBERT Fine-Tuning (RUN THIS ON GOOGLE COLAB, not locally)
===========================================================================
Why Colab: fine-tuning DistilBERT needs (a) internet access to download
~260MB of pretrained weights from Hugging Face, and (b) ideally a GPU.
Colab's free tier gives you both. This won't run in a sandboxed/offline
environment -- that's not a bug in this script, it's a hard requirement
of using a pretrained transformer.

SETUP (run this cell first in Colab):
    !pip install -q transformers datasets torch scikit-learn accelerate

Then:
    1. Upload dataset.csv (from Week 2/3/4) to the Colab file browser,
       OR paste its contents into a cell and write it out.
    2. Runtime -> Change runtime type -> GPU (T4 is fine, free tier).
    3. Run this whole script as one cell (or split as you like).

METHODOLOGY: same 5-fold stratified cross-validation as Week 4, so the
result is directly comparable to your SVM/Logistic Regression/Naive
Bayes/Random Forest numbers -- not a different, incomparable metric.
With only 120 examples, expect this to take a few minutes total across
all 5 folds on a T4 GPU.
"""

import csv
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    Trainer, TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
N_FOLDS = 5
N_EPOCHS = 8  # small dataset -> more epochs is fine, watch for overfitting
BATCH_SIZE = 8


def load_dataset(path="dataset.csv"):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["intent"])
    return texts, labels


def run_fold(train_texts, train_labels, test_texts, test_labels, num_labels, tokenizer):
    train_enc = tokenizer(train_texts, truncation=True, padding=True)
    test_enc = tokenizer(test_texts, truncation=True, padding=True)

    train_ds = Dataset.from_dict({**train_enc, "labels": train_labels})
    test_ds = Dataset.from_dict({**test_enc, "labels": test_labels})

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir="./tmp_trainer",
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    preds = trainer.predict(test_ds)
    pred_labels = np.argmax(preds.predictions, axis=1)
    return pred_labels


def run():
    texts, labels = load_dataset()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    label_names = list(le.classes_)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_true, all_pred = [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(texts, y), 1):
        print(f"\n=== Fold {fold}/{N_FOLDS} ===")
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = [int(y[i]) for i in train_idx]
        test_labels = [int(y[i]) for i in test_idx]

        pred = run_fold(train_texts, train_labels, test_texts, test_labels,
                         num_labels=len(label_names), tokenizer=tokenizer)
        all_true.extend(test_labels)
        all_pred.extend(pred)

    acc = accuracy_score(all_true, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_true, all_pred)

    print("\n" + "=" * 60)
    print("DISTILBERT -- 5-FOLD CV RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision (macro): {precision:.3f}")
    print(f"Recall (macro):    {recall:.3f}")
    print(f"F1 (macro):        {f1:.3f}")
    print("\nConfusion matrix:")
    print(label_names)
    print(cm)

    # Save in the SAME format as Week 4's model_comparison.csv so you can
    # concatenate them directly for your final Traditional-ML-vs-DistilBERT chart.
    with open("distilbert_result.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
        writer.writerow(["DistilBERT", f"{acc:.4f}", f"{precision:.4f}",
                          f"{recall:.4f}", f"{f1:.4f}"])
    print("\nSaved -> distilbert_result.csv (download this from Colab)")


if __name__ == "__main__":
    run()
