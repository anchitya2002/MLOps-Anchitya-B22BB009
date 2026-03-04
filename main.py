import os
import random
from collections import defaultdict
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data.data import load_all_genres

# config
model_name      = "distilbert-base-cased"
device          = "cuda" if torch.cuda.is_available() else "cpu"
max_length      = 128
saved_model_dir = "distilbert-reviews-genres-lora"
per_genre       = 200

print(f"Using device: {device}", flush=True)

# load data
print("Loading reviews...", flush=True)
genre_reviews_dict = load_all_genres(cache_path="genre_reviews_dict.pickle", head=10000, sample_size=2000)

for genre, reviews in genre_reviews_dict.items():
    print(f"{genre}: {random.sample(reviews, 1)[0][:80]}")

# train/test split
train_texts, train_labels = [], []
test_texts,  test_labels  = [], []

for genre, reviews in genre_reviews_dict.items():
    sample = random.sample(reviews, min(per_genre, len(reviews)))
    split  = int(len(sample) * 0.8)
    for r in sample[:split]:
        train_texts.append(r)
        train_labels.append(genre)
    for r in sample[split:]:
        test_texts.append(r)
        test_labels.append(genre)

print(f"\nTrain: {len(train_texts)} | Test: {len(test_texts)}", flush=True)

# tfidf baseline
print("\nTF-IDF baseline...", flush=True)
vectorizer = TfidfVectorizer()
X_train    = vectorizer.fit_transform(train_texts)
X_test     = vectorizer.transform(test_texts)
lr_model   = LogisticRegression(max_iter=1000).fit(X_train, train_labels)
print(classification_report(test_labels, lr_model.predict(X_test)))

# tokenize
print("Tokenizing...", flush=True)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

unique_labels = sorted(set(train_labels))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=max_length)

train_labels_enc = [label2id[y] for y in train_labels]
test_labels_enc  = [label2id[y] for y in test_labels]


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ReviewDataset(train_encodings, train_labels_enc)
test_dataset  = ReviewDataset(test_encodings,  test_labels_enc)

# load base model
print("Loading model...", flush=True)
base_model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

# LoRA config
# target_modules: the attention projection layers in DistilBERT
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,       # sequence classification
    r=16,                              # rank — higher = more capacity, more params
    lora_alpha=32,                     # scaling factor (usually 2x rank)
    lora_dropout=0.1,                  # dropout on LoRA layers
    target_modules=["q_lin", "v_lin"], # query and value projections in DistilBERT
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # shows how few params are trained
model = model.to(device)


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}


training_args = TrainingArguments(
    output_dir="./results_lora",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-4,              # LoRA uses higher lr than full fine-tuning
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=(device == "cuda"),
    dataloader_num_workers=0,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Training with LoRA...", flush=True)
trainer.train()
trainer.save_model(saved_model_dir)
print(f"LoRA model saved to {saved_model_dir}", flush=True)

# evaluate
print("\nEvaluating...", flush=True)
trainer.evaluate()

predicted_results = trainer.predict(test_dataset)
predicted_ids     = predicted_results.predictions.argmax(-1).flatten().tolist()
predicted_labels  = [id2label[i] for i in predicted_ids]

print(classification_report(test_labels, predicted_labels))

# sample predictions
correct = [(t, p, txt) for t, p, txt in zip(test_labels, predicted_labels, test_texts) if t == p]
wrong   = [(t, p, txt) for t, p, txt in zip(test_labels, predicted_labels, test_texts) if t != p]

print("Correct predictions:")
for true, pred, text in random.sample(correct, min(5, len(correct))):
    print(f"  LABEL: {true}")
    print(f"  TEXT : {text[:100]}...\n")

print("Misclassifications:")
for true, pred, text in random.sample(wrong, min(5, len(wrong))):
    print(f"  TRUE: {true} | PREDICTED: {pred}")
    print(f"  TEXT: {text[:100]}...\n")

# confusion matrix
os.makedirs("plots", exist_ok=True)

def plot_confusion(true_labels, pred_labels, remove_diagonal=False, title="Confusion Matrix", save_path=None):
    counts = defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        if remove_diagonal and t == p:
            continue
        counts[(t, p)] += 1

    rows = [{"True Genre": t, "Predicted Genre": p, "Count": c}
            for (t, p), c in counts.items()]
    df_wide = pd.DataFrame(rows).pivot_table(
        index="True Genre", columns="Predicted Genre", values="Count"
    )

    plt.figure(figsize=(9, 7))
    sns.set(style="ticks", font_scale=1.2)
    sns.heatmap(df_wide, linewidths=1, cmap="Purples", annot=True, fmt=".0f")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    plt.show()

plot_confusion(test_labels, predicted_labels, remove_diagonal=False,
               title="Confusion Matrix (LoRA)", save_path="plots/confusion_matrix.png")
plot_confusion(test_labels, predicted_labels, remove_diagonal=True,
               title="Misclassification Matrix (LoRA)", save_path="plots/misclassification_matrix.png")