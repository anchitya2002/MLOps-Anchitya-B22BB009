import gzip
import json
import pickle
import random
import requests
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


# ─────────────────────────────────────────────
# Genre → URL mapping
# ─────────────────────────────────────────────
GENRE_URL_DICT = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


# ─────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ─────────────────────────────────────────────
# Streaming loader
# ─────────────────────────────────────────────
def load_reviews(url, head=10000, sample_size=2000):
    reviews = []

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if head and i >= head:
                break

            data = json.loads(line)
            text = data.get("review_text", "").strip()

            if text:
                reviews.append(text)

    return random.sample(reviews, min(sample_size, len(reviews)))


# ─────────────────────────────────────────────
# Load all genres
# ─────────────────────────────────────────────
def load_all_genres(cache_path="genre_reviews_dict.pickle"):

    try:
        genre_reviews = pickle.load(open(cache_path, "rb"))
        print("Loaded cached dataset")
        return genre_reviews
    except:
        pass

    genre_reviews = {}

    for genre, url in GENRE_URL_DICT.items():
        print("Loading:", genre)
        genre_reviews[genre] = load_reviews(url)

    pickle.dump(genre_reviews, open(cache_path, "wb"))

    return genre_reviews


# ─────────────────────────────────────────────
# Train/Test split
# ─────────────────────────────────────────────
def build_splits(genre_reviews, per_genre=1000, train_ratio=0.8):

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    n_train = int(per_genre * train_ratio)

    for genre, reviews in genre_reviews.items():

        sample = random.sample(reviews, min(per_genre, len(reviews)))

        for r in sample[:n_train]:
            train_texts.append(r)
            train_labels.append(genre)

        for r in sample[n_train:]:
            test_texts.append(r)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


# ─────────────────────────────────────────────
# Label mapping
# ─────────────────────────────────────────────
def build_label_maps(train_labels):

    unique = sorted(set(train_labels))

    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}

    return label2id, id2label


# ─────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────
def tokenize_splits(train_texts, test_texts, train_labels, test_labels, label2id):

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_dataset = ReviewDataset(train_enc, [label2id[l] for l in train_labels])
    test_dataset = ReviewDataset(test_enc, [label2id[l] for l in test_labels])

    return train_dataset, test_dataset



def save_splits(train_texts, train_labels, test_texts, test_labels, path="data"):
    import os
    os.makedirs(path, exist_ok=True)

    train_data = [
        {"text": t, "label": l}
        for t, l in zip(train_texts, train_labels)
    ]

    test_data = [
        {"text": t, "label": l}
        for t, l in zip(test_texts, test_labels)
    ]

    with open(f"{path}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    with open(f"{path}/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"Saved train/test datasets to {path}/")

# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading dataset...")

    # Load or download reviews (pickle cache)
    genre_reviews = load_all_genres()

    # Create train/test splits
    train_texts, train_labels, test_texts, test_labels = build_splits(genre_reviews)

    # Save JSON splits
    save_splits(train_texts, train_labels, test_texts, test_labels)

    # Create label mappings
    label2id, id2label = build_label_maps(train_labels)

    # Save label mappings
    with open("data/label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)

    with open("data/id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    # Tokenize datasets
    train_dataset, test_dataset = tokenize_splits(
        train_texts,
        test_texts,
        train_labels,
        test_labels,
        label2id
    )

    print("\nDataset Ready")
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Labels:", label2id)