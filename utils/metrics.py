"""
utils.py - Shared utilities: plotting, baseline TF-IDF model, misc helpers.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

sns.set(style="ticks", font_scale=1.2)


# ─────────────────────────────────────────────
# Confusion-matrix plots
# ─────────────────────────────────────────────
def plot_confusion_matrix(
    true_labels:      list[str],
    predicted_labels: list[str],
    remove_diagonal:  bool = False,
    title:            str  = "Confusion Matrix",
    figsize:          tuple = (9, 7),
    save_path:        str | None = None,
) -> None:
    """
    Plot a genre × genre heat-map.

    Parameters
    ----------
    remove_diagonal : if True, zero out the diagonal so misclassifications
                      are easier to see (i.e. the "Misclassification Matrix").
    """
    counts: defaultdict = defaultdict(int)
    for true, pred in zip(true_labels, predicted_labels):
        if remove_diagonal and true == pred:
            continue
        counts[(true, pred)] += 1

    rows = [
        {"True Genre": t, "Predicted Genre": p, "Count": c}
        for (t, p), c in counts.items()
    ]
    df = pd.DataFrame(rows)
    df_wide = df.pivot_table(
        index="True Genre", columns="Predicted Genre", values="Count"
    )

    plt.figure(figsize=figsize)
    sns.heatmap(df_wide, linewidths=1, cmap="Purples", annot=True, fmt=".0f")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# TF-IDF baseline
# ─────────────────────────────────────────────
def run_tfidf_baseline(
    train_texts:  list[str],
    train_labels: list[str],
    test_texts:   list[str],
    test_labels:  list[str],
) -> None:
    """Train a TF-IDF + Logistic Regression baseline and print its report."""
    print("── TF-IDF Baseline ──")
    vec   = TfidfVectorizer()
    X_tr  = vec.fit_transform(train_texts)
    X_te  = vec.transform(test_texts)
    model = LogisticRegression(max_iter=1_000).fit(X_tr, train_labels)
    preds = model.predict(X_te)
    print(classification_report(test_labels, preds))


# ─────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def gpu_info() -> None:
    """Print GPU availability and VRAM."""
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / 1024 ** 3
            print(f"  GPU {i}: {props.name}  ({vram:.1f} GB VRAM)")
    else:
        print("  No CUDA GPU detected — training on CPU.")

sns.set(style="ticks", font_scale=1.2)

def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    remove_diagonal=False,
    title="Confusion Matrix",
    figsize=(9, 7),
    save_path=None,
):
    counts = defaultdict(int)
    for true, pred in zip(true_labels, predicted_labels):
        if remove_diagonal and true == pred:
            continue
        counts[(true, pred)] += 1

    rows = [{"True Genre": t, "Predicted Genre": p, "Count": c}
            for (t, p), c in counts.items()]
    df = pd.DataFrame(rows)
    df_wide = df.pivot_table(index="True Genre", columns="Predicted Genre", values="Count")

    plt.figure(figsize=figsize)
    sns.heatmap(df_wide, linewidths=1, cmap="Purples", annot=True, fmt=".0f")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()