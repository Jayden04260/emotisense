"""
Fine-tunes DistilBERT for text emotion classification, as a comparison
point against the production Linear SVM (see README "Future Improvements"
- TF-IDF has no concept of negation, e.g. "I am not happy at all" scores
confidently as Joy; a contextual transformer embedding should handle this
directly).

Does NOT replace results/emotion_model.pkl / vectorizer.pkl - saves to
results/distilbert_emotion_model/ instead, exactly like
compare_text_models.py's models_comparison/ pattern (a comparison point,
promoted manually if it wins).

Trained on a stratified subset (not the full 16,000-row training set) -
a smoke test (see git history / dev notes) found this CPU-only, 3.4GB-RAM
machine takes ~3 seconds per batch-of-4 training step, meaning a full
epoch over all 16,000 rows would take ~3.5 hours. SUBSET_SIZE below
trades data volume for a training run that finishes in a practical
timeframe; increase it if you have more time/a faster machine.

Run from the project root with:

    python src/train_text_distilbert.py
"""

import json
import os
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TRAIN_PATH = "data/raw/train.txt"
TEST_PATH = "data/raw/test.txt"
RESULTS_DIR = "results"
MODEL_DIR = os.path.join(RESULTS_DIR, "distilbert_emotion_model")
REPORT_PATH = os.path.join(RESULTS_DIR, "distilbert_classification_report.txt")
METADATA_PATH = os.path.join(RESULTS_DIR, "distilbert_model_metadata.json")

SUBSET_SIZE = 3000  # stratified sample of the 16,000-row training set - see module docstring
BATCH_SIZE = 8
EPOCHS = 2
MAX_LENGTH = 64  # this dataset is short sentences; covers virtually all of them
LEARNING_RATE = 2e-5


def load_data(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.rsplit(";", 1)
            texts.append(text)
            labels.append(label)
    return texts, labels


class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    train_texts_full, train_labels_full = load_data(TRAIN_PATH)
    test_texts, test_labels_str = load_data(TEST_PATH)

    label_names = sorted(set(train_labels_full))
    label_to_id = {l: i for i, l in enumerate(label_names)}
    print(f"Labels: {label_names}")

    # Stratified subset of the training set - see module docstring for why.
    subset_frac = SUBSET_SIZE / len(train_texts_full)
    train_texts, _, train_labels_str, _ = train_test_split(
        train_texts_full, train_labels_full,
        train_size=subset_frac, random_state=42, stratify=train_labels_full,
    )
    print(f"Training subset: {len(train_texts)} of {len(train_texts_full)} rows")
    print("Subset class distribution:", dict(sorted(Counter(train_labels_str).items())))

    train_labels = [label_to_id[l] for l in train_labels_str]
    test_labels = [label_to_id[l] for l in test_labels_str]

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(label_names)), y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print("Class weights:", dict(zip(label_names, class_weights.tolist())))

    print("Loading tokenizer and tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")

    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(label_names)
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    print(f"Training: {EPOCHS} epochs x {len(train_loader)} steps = {total_steps} total steps")

    model.train()
    start = time.perf_counter()
    step = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            if step % 50 == 0 or step == total_steps:
                elapsed = time.perf_counter() - start
                rate = elapsed / step
                remaining = rate * (total_steps - step)
                print(
                    f"  step {step}/{total_steps}  loss={loss.item():.4f}  "
                    f"elapsed={elapsed / 60:.1f}min  eta={remaining / 60:.1f}min",
                    flush=True,
                )
        print(f"Epoch {epoch + 1}/{EPOCHS} done, avg loss={epoch_loss / len(train_loader):.4f}")

    train_time = time.perf_counter() - start
    print(f"\nTraining finished in {train_time / 60:.1f} minutes")

    print("\nEvaluating on real test set...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("labels", None)
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.tolist())

    accuracy = accuracy_score(test_labels, all_preds)
    f1_macro = f1_score(test_labels, all_preds, average="macro")
    report = classification_report(test_labels, all_preds, target_names=label_names)

    print(f"\nAccuracy: {accuracy:.4f}  F1 (macro): {f1_macro:.4f}\n")
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\nF1 (macro): {f1_macro:.4f}\n\n")
        f.write(report)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    metadata = {
        "model": "distilbert-base-uncased (fine-tuned)",
        "subset_size": len(train_texts),
        "full_train_size": len(train_texts_full),
        "test_size": len(test_texts),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "class_weighted_loss": True,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "train_time_seconds": train_time,
        "labels": label_names,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nModel saved to: {MODEL_DIR}/")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
