import argparse
import os
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data.data import (
    build_label_maps,
    build_splits,
    load_all_genres,
    tokenize_splits,
)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}


# ─────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    print(f"Training device: {dev}")
    return dev


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────
def train(
    model,
    train_dataset,
    test_dataset,
    output_dir:       str   = "distilbert-reviews-genres",
    results_dir:      str   = "./results",
    num_epochs:       int   = 10,
    train_batch_size: int   = 16,
    eval_batch_size:  int   = 32,
    learning_rate:    float = 5e-5,
    warmup_steps:     int   = 50,
    weight_decay:     float = 0.01,
    logging_steps:    int   = 50,
) -> Trainer:
    """Fine-tune a pre-loaded model on train_dataset, evaluate on test_dataset."""

    os.environ["WANDB_DISABLED"] = "true"

    device = get_device()
    model = model.to(device)

    # ── Step count preview ─────────────────────────────────────────────────
    steps_per_epoch = max(1, (len(train_dataset) + train_batch_size - 1) // train_batch_size)
    total_steps     = steps_per_epoch * num_epochs
    print(f"  Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")

    use_fp16 = (device == "cuda")
    use_bf16 = (device == "mps")

    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=4 if device != "cpu" else 0,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n── Training ──")
    trainer.train()

    trainer.save_model(output_dir)
    print(f"\nModel saved → {output_dir}")

    return trainer


# ─────────────────────────────────────────────
# CLI entry-point (standalone use)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for genre classification")
    parser.add_argument("--model_name",       default="distilbert-base-cased")
    parser.add_argument("--output_dir",       default="distilbert-reviews-genres")
    parser.add_argument("--epochs",           type=int,   default=10)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--eval_batch_size",  type=int,   default=32)
    parser.add_argument("--learning_rate",    type=float, default=5e-5)
    parser.add_argument("--max_length",       type=int,   default=512)
    parser.add_argument("--per_genre",        type=int,   default=100)
    parser.add_argument("--cache_path",       default="genre_reviews_dict.pickle")
    args = parser.parse_args()

    from models.hf_model import load_model

    genre_reviews = load_all_genres(cache_path=args.cache_path)
    train_texts, train_labels, test_texts, test_labels = build_splits(
        genre_reviews, per_genre=args.per_genre
    )
    label2id, id2label = build_label_maps(train_labels)
    train_dataset, test_dataset = tokenize_splits(
        train_texts, test_texts, train_labels, test_labels,
        label2id, model_name=args.model_name, max_length=args.max_length,
    )
    _, model = load_model(
        model_name=args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    train(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
    )