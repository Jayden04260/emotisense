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

v2 (2026-08-18): the first run (see ROADMAP.md item 1, "What actually
happened") matched production accuracy but did NOT fix the negation
failure it was built to test - "I am not happy at all" still scored 96%
Joy. This version implements ROADMAP's own suggested next step:
NEGATION_OVERSAMPLE_FRACTION deliberately over-represents negation-
containing rows in the training subset (natural rate in the full
16,000-row set is 12.1%; see _split_negation_rows), instead of leaving
that pattern's exposure to chance stratified sampling. run_negation_probe
below tests the exact sentences that failed before, so "did this work"
is answered directly rather than inferred from overall accuracy moving.

Run from the project root with:

    python src/train_text_distilbert.py
"""

import json
import os
import re
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
NEGATION_PROBE_PATH = os.path.join(RESULTS_DIR, "distilbert_negation_probe.txt")

SUBSET_SIZE = 3000  # stratified sample of the 16,000-row training set - see module docstring
BATCH_SIZE = 8
EPOCHS = 2
MAX_LENGTH = 64  # this dataset is short sentences; covers virtually all of them

# Natural rate of negation-containing rows in the full training set is
# 12.1% (ROADMAP.md's own count, same regex here). Roughly tripling that
# within the training subset is a deliberate, meaningful oversample
# without making the subset unrealistically dominated by one pattern.
NEGATION_OVERSAMPLE_FRACTION = 0.35
NEGATION_RE = re.compile(r"\b(not|never|n't)\b", re.IGNORECASE)

# The exact sentences ROADMAP.md documented as failing on the v1 model,
# plus adversarial_probe.py's "Double negative" case for the same probe
# set used against the production sklearn model - so this is a genuine
# apples-to-apples check, not a new easier test made up after the fact.
NEGATION_PROBES = [
    ("Negation", "I am not happy at all"),
    ("Negation (inverse)", "I am not sad"),
    ("Double negative", "I wouldn't say I'm not pleased"),
]
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


def build_negation_oversampled_subset(texts, labels, subset_size, oversample_fraction):
    """
    Splits (texts, labels) into negation-containing and plain rows, then
    builds a subset of `subset_size` where negation rows make up
    `oversample_fraction` of the total (each half still stratified by
    label internally) - instead of a single stratified sample over
    everything, which would leave negation examples at their natural
    ~12% rate and give the model the same limited exposure v1 had.
    """
    is_negation = np.array([bool(NEGATION_RE.search(t)) for t in texts])
    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=object)

    negation_texts, negation_labels = texts[is_negation], labels[is_negation]
    plain_texts, plain_labels = texts[~is_negation], labels[~is_negation]
    print(
        f"Negation-containing rows: {len(negation_texts)} of {len(texts)} "
        f"({len(negation_texts) / len(texts):.1%})"
    )

    target_negation_count = min(int(subset_size * oversample_fraction), len(negation_texts))
    target_plain_count = subset_size - target_negation_count

    neg_frac = target_negation_count / len(negation_texts)
    negation_sub_texts, _, negation_sub_labels, _ = train_test_split(
        negation_texts, negation_labels, train_size=neg_frac, random_state=42, stratify=negation_labels
    )
    plain_frac = target_plain_count / len(plain_texts)
    plain_sub_texts, _, plain_sub_labels, _ = train_test_split(
        plain_texts, plain_labels, train_size=plain_frac, random_state=42, stratify=plain_labels
    )

    combined_texts = list(negation_sub_texts) + list(plain_sub_texts)
    combined_labels = list(negation_sub_labels) + list(plain_sub_labels)
    rng = np.random.RandomState(42)
    order = rng.permutation(len(combined_texts))
    combined_texts = [combined_texts[i] for i in order]
    combined_labels = [combined_labels[i] for i in order]

    print(
        f"Oversampled subset: {len(combined_texts)} rows, "
        f"{target_negation_count} negation ({target_negation_count / len(combined_texts):.1%}) + "
        f"{target_plain_count} plain"
    )
    return combined_texts, combined_labels


def run_negation_probe(model, tokenizer, label_names, device):
    """Runs NEGATION_PROBES through the fine-tuned model directly (not
    the sklearn vectorizer path - see module docstring) and returns a
    list of (name, text, predicted_label, confidence) results."""
    model.eval()
    results = []
    with torch.no_grad():
        for name, text in NEGATION_PROBES:
            encoding = tokenizer(text, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}
            logits = model(**encoding).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs))
            results.append((name, text, label_names[pred_id], float(probs[pred_id])))
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    train_texts_full, train_labels_full = load_data(TRAIN_PATH)
    test_texts, test_labels_str = load_data(TEST_PATH)

    label_names = sorted(set(train_labels_full))
    label_to_id = {l: i for i, l in enumerate(label_names)}
    print(f"Labels: {label_names}")

    # Negation-oversampled subset of the training set - see module
    # docstring ("v2") for why this replaced a plain stratified sample.
    train_texts, train_labels_str = build_negation_oversampled_subset(
        train_texts_full, train_labels_full, SUBSET_SIZE, NEGATION_OVERSAMPLE_FRACTION
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

    print("\nRunning negation probe (the exact sentences v1 failed on)...")
    probe_results = run_negation_probe(model, tokenizer, label_names, "cpu")
    probe_lines = ["Negation probe results (v2, negation-oversampled)", ""]
    for name, text, pred_label, confidence in probe_results:
        line = f"{name:22} {text!r:45} -> {pred_label:10} ({confidence * 100:5.1f}%)"
        print("  " + line)
        probe_lines.append(line)
    probe_lines.append("")
    probe_lines.append(
        "v1 (plain stratified subset, no oversampling) results: "
        '"I am not happy at all" -> joy (96.1%); "I am not sad" -> sadness (87.7%). '
        "See ROADMAP.md item 1 for the full v1 write-up."
    )
    with open(NEGATION_PROBE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(probe_lines) + "\n")

    metadata = {
        "model": "distilbert-base-uncased (fine-tuned, v2 negation-oversampled)",
        "subset_size": len(train_texts),
        "full_train_size": len(train_texts_full),
        "test_size": len(test_texts),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "class_weighted_loss": True,
        "negation_oversample_fraction": NEGATION_OVERSAMPLE_FRACTION,
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
    print(f"Negation probe saved to: {NEGATION_PROBE_PATH}")


if __name__ == "__main__":
    main()
