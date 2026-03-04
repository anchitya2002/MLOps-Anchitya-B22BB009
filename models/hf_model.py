"""
models/hf_model.py - Load DistilBERT tokenizer and model.
"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def load_model(
    model_name: str  = "distilbert-base-cased",
    num_labels: int  = 8,
    label2id:   dict = None,
    id2label:   dict = None,
):
    """
    Load DistilBERT tokenizer and sequence classification model.

    Returns
    -------
    tokenizer, model  (model moved to best available device)
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label or {},
        label2id=label2id or {},
    ).to(device)

    print(f"  Model loaded on: {device}")
    return tokenizer, model