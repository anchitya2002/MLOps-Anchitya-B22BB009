from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

repo_name = "anchitya/book-genre-classifier"

print("Loading model from Hugging Face...")
model = AutoModelForSequenceClassification.from_pretrained(repo_name)
tokenizer = AutoTokenizer.from_pretrained(repo_name)


print("Loading evaluation dataset...")
dataset = load_dataset("json", data_files={"test": "data/test.json"})


def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")


dataset = dataset.map(tokenize)

dataset = dataset.rename_column("label", "labels")

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy
    }


training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
    logging_strategy="no"
)


trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)


print("Running evaluation...")

results = trainer.evaluate()

print("\nEvaluation Results")
print("-------------------")
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"Runtime: {results['eval_runtime']:.2f} seconds")