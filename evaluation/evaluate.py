import argparse
import random
from collections import defaultdict

import pandas as pd
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import build_label_maps, build_splits, load_all_genres, tokenize_splits
from utils import plot_confusion_matrix


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────
def evaluate(
    model_dir:   str = "distilbert-reviews-genres",
    model_name:  str = "distilbert-base-cased",
    max_length:  int = 512,
    per_genre:   int = 1_000,
    cache_path:  str = "genre_reviews_dict.pickle",
    n_examples:  int = 20,
) -> dict:
    """
    Load saved model, run inference on the test split, print report & confusion matrix.

    Returns a dict with keys: classification_report, predicted_labels, test_labels, test_texts.
    """
    # ── Data ───────────────────────────────────────────────────────────────
    genre_reviews = load_all_genres(cache_path=cache_path)
    train_texts, train_labels, test_texts, test_labels = build_splits(
        genre_reviews, per_genre=per_genre
    )
    label2id, id2label = build_label_maps(train_labels)

    _, test_dataset = tokenize_splits(
        train_texts, test_texts, train_labels, test_labels,
        label2id, model_name=model_name, max_length=max_length,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    # Minimal TrainingArguments just for the Trainer (no training happens)
    eval_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(model=model, args=eval_args)

    # ── Predict ────────────────────────────────────────────────────────────
    results = trainer.predict(test_dataset)
    predicted_ids = results.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in predicted_ids]

    # ── Report ─────────────────────────────────────────────────────────────
    report = classification_report(test_labels, predicted_labels)
    print("\n── Classification Report ──\n")
    print(report)

    # ── Sample correct predictions ─────────────────────────────────────────
    correct = [
        (t, p, txt)
        for t, p, txt in zip(test_labels, predicted_labels, test_texts)
        if t == p
    ]
    print(f"\n── {n_examples} Correct Predictions ──")
    for true, pred, text in random.sample(correct, min(n_examples, len(correct))):
        print(f"  LABEL: {true}")
        print(f"  TEXT : {text[:120]} …\n")

    # ── Sample incorrect predictions ───────────────────────────────────────
    wrong = [
        (t, p, txt)
        for t, p, txt in zip(test_labels, predicted_labels, test_texts)
        if t != p
    ]
    print(f"\n── {n_examples} Misclassifications ──")
    for true, pred, text in random.sample(wrong, min(n_examples, len(wrong))):
        print(f"  TRUE: {true}  |  PREDICTED: {pred}")
        print(f"  TEXT: {text[:120]} …\n")

    # ── Confusion matrix ───────────────────────────────────────────────────
    plot_confusion_matrix(test_labels, predicted_labels, remove_diagonal=False,
                          title="Confusion Matrix")
    plot_confusion_matrix(test_labels, predicted_labels, remove_diagonal=True,
                          title="Misclassification Matrix")

    return {
        "classification_report": report,
        "predicted_labels": predicted_labels,
        "test_labels": test_labels,
        "test_texts": test_texts,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained genre classifier")
    parser.add_argument("--model_dir",   default="distilbert-reviews-genres")
    parser.add_argument("--model_name",  default="distilbert-base-cased")
    parser.add_argument("--max_length",  type=int, default=512)
    parser.add_argument("--per_genre",   type=int, default=1000)
    parser.add_argument("--cache_path",  default="genre_reviews_dict.pickle")
    parser.add_argument("--n_examples",  type=int, default=20)
    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        per_genre=args.per_genre,
        cache_path=args.cache_path,
        n_examples=args.n_examples,
    )