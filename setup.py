import os
from pathlib import Path

# Project files and contents
files = {
    "data/dataset_loader.py": r'''
import random
import gzip
import json
import requests

genre_url_dict = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}

def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)

    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))

def prepare_datasets():
    genre_reviews_dict = {}

    for genre, url in genre_url_dict.items():
        print(f"Loading {genre}")
        genre_reviews_dict[genre] = load_reviews(url)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews_dict.items():
        reviews = random.sample(reviews, 1000)

        for r in reviews[:800]:
            train_texts.append(r)
            train_labels.append(genre)

        for r in reviews[800:]:
            test_texts.append(r)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels
''',

    "models/hf_model.py": r'''
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_NAME = "distilbert-base-cased"

def load_tokenizer():
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def load_model(num_labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    ).to(device)
    return model, device
''',

    "utils/metrics.py": r'''
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
''',

    "training/train.py": r'''
import torch
from transformers import Trainer, TrainingArguments
from data.dataset_loader import prepare_datasets
from models.hf_model import load_tokenizer, load_model
from utils.metrics import compute_metrics

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train():
    train_texts, train_labels, test_texts, test_labels = prepare_datasets()

    tokenizer = load_tokenizer()

    unique_labels = sorted(list(set(train_labels)))
    label2id = {l: i for i, l in enumerate(unique_labels)}

    train_labels_enc = [label2id[x] for x in train_labels]
    test_labels_enc = [label2id[x] for x in test_labels]

    train_enc = tokenizer(train_texts, truncation=True, padding=True)
    test_enc = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = ReviewDataset(train_enc, train_labels_enc)
    test_dataset = ReviewDataset(test_enc, test_labels_enc)

    model, device = load_model(len(unique_labels))

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("final_model")

    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    train()
''',

    "evaluation/evaluate.py": r'''
from transformers import Trainer
from utils.metrics import compute_metrics

def evaluate_model(model, dataset):
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )
    return trainer.evaluate(dataset)
'''
}

# Create files
for path, content in files.items():
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content.strip() + "\n", encoding="utf-8")
