"""
v4 text model: DistilBERT fine-tuned on the FULL emotion dataset plus
GoEmotions, with a neutral class. See ROADMAP.md item 1 ("v4") for why.

Why v1-v3 (src/train_text_distilbert.py) couldn't fix negation, whatever the
training trick: the "Emotions dataset for NLP" (dair-ai/emotion) is labelled
by distant supervision - each tweet's label is the emotion word it contains,
not what it means. Negated statements are both rare (<10 of 16,000 for
"not happy"/"don't feel good"-style phrasings) and labelled by the negated
word itself: "i do not feel bad about it" is labelled sadness, "i feel
content if not happy" is labelled joy. A model trained on those labels is
being *taught* to ignore negation. Oversampling (v2) and a contrastive loss
(v3) can't overcome labels that point the other way.

The fix is data, not architecture: GoEmotions (Demszky et al., 2020) is
58k Reddit comments labelled by human raters, including a "neutral" class.
Its 27 labels are mapped onto this project's six via the standard
GoEmotions Ekman grouping (GO_TO_PROJECT below); examples whose labels
disagree after mapping are dropped.

Also trains on all 16,000 dair-ai rows (v1-v3 used a 3,000-row subset
because the old CPU-only machine needed ~3.5h per full epoch; on a GPU the
whole run takes a few minutes).

Run from the project root (a CUDA GPU is strongly recommended):

    pip install -r requirements-distilbert.txt
    python src/train_text_transformer.py

Writes results/distilbert_emotion_model/ (gitignored - push it to the Hub
with scripts/push_distilbert_fp16_to_hub.py) plus the small, committed
report/metadata/probe files listed below.
"""

import json
import os
import time

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:  # use the OS certificate store (needed behind TLS-inspecting antivirus/proxies)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_MODEL = "distilbert-base-uncased"
RESULTS_DIR = "results"
MODEL_DIR = os.path.join(RESULTS_DIR, "distilbert_emotion_model")
REPORT_PATH = os.path.join(RESULTS_DIR, "distilbert_classification_report.txt")
METADATA_PATH = os.path.join(RESULTS_DIR, "distilbert_model_metadata.json")
PROBE_PATH = os.path.join(RESULTS_DIR, "distilbert_negation_probe.txt")
LOCAL_TRAIN_PATH = "data/raw/train.txt"  # used instead of the Hub copy if present (may hold local additions)

LABELS = ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
EPOCHS = 3
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
MAX_LENGTH = 128
SEED = 42

# GoEmotions fine-grained label -> this project's label. The standard
# GoEmotions Ekman grouping, except love/caring/desire stay "love" (this
# project has a love class) and disgust folds into anger (it has no disgust).
GO_TO_PROJECT = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger", "disgust": "anger",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy", "gratitude": "joy",
    "optimism": "joy", "relief": "joy", "pride": "joy", "admiration": "joy",
    "love": "love", "caring": "love", "desire": "love",
    "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise", "curiosity": "surprise",
    "neutral": "neutral",
}

# Behavioural probes: the sentences earlier versions were documented as
# failing on (ROADMAP.md, README "Adversarial Input Exploration"), plus a
# few more of each kind. `acceptable` lists every label that would be a
# defensible read - deliberately lenient, since the point is catching
# confidently-wrong answers, not grading shades of emotion.
PROBES = [
    ("negation", "I am not happy at all", {"sadness", "anger"}),
    ("negation", "I am not sad", {"joy", "neutral"}),
    ("negation", "im not having the best day", {"sadness", "anger"}),
    ("negation", "I don't feel good about this", {"sadness", "fear", "anger"}),
    ("negation", "this isn't fun anymore", {"sadness", "anger"}),
    ("negation", "I'm not angry, just tired", {"sadness", "neutral"}),
    ("negation", "I'm not scared of anything", {"joy", "neutral"}),
    ("double negative", "I wouldn't say I'm not pleased", {"joy"}),
    ("neutral", "The meeting is at 3pm tomorrow", {"neutral"}),
    ("neutral", "ok", {"neutral"}),
    ("neutral", "I bought milk and bread from the shop", {"neutral"}),
    ("neutral", "The train leaves from platform 4", {"neutral"}),
    ("sarcasm", "Oh great, cancelled again. Just perfect.", {"anger", "sadness"}),
    ("plain", "I AM SO ANGRY RIGHT NOW", {"anger"}),
    ("plain", "I'm so happy for you!", {"joy", "love"}),
    ("plain", "I miss her so much it hurts", {"sadness", "love"}),
    ("plain", "I'm terrified of the exam tomorrow", {"fear"}),
    ("plain", "Wow, I did not expect that at all!", {"surprise"}),
]


def load_dair():
    d = load_dataset("dair-ai/emotion", "split")
    names = d["train"].features["label"].names
    splits = {s: (list(d[s]["text"]), [names[i] for i in d[s]["label"]]) for s in ["train", "validation", "test"]}
    if os.path.exists(LOCAL_TRAIN_PATH):
        texts, labels = [], []
        with open(LOCAL_TRAIN_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    text, label = line.strip().rsplit(";", 1)
                    texts.append(text)
                    labels.append(label)
        print(f"Using local {LOCAL_TRAIN_PATH} ({len(texts)} rows) instead of the Hub training split")
        splits["train"] = (texts, labels)
    return splits


def load_goemotions():
    g = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = g["train"].features["labels"].feature.names
    splits = {}
    for s in ["train", "validation", "test"]:
        texts, labels = [], []
        for text, ids in zip(g[s]["text"], g[s]["labels"]):
            mapped = {GO_TO_PROJECT[names[i]] for i in ids}
            if len(mapped) == 1:
                texts.append(text)
                labels.append(mapped.pop())
        splits[s] = (texts, labels)
    return splits


def main():
    torch.manual_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dair, go = load_dair(), load_goemotions()
    train_x = dair["train"][0] + go["train"][0]
    train_y = dair["train"][1] + go["train"][1]
    val_x = dair["validation"][0] + go["validation"][0]
    val_y = dair["validation"][1] + go["validation"][1]
    print(f"Training rows: {len(train_x)} ({len(dair['train'][0])} dair-ai + {len(go['train'][0])} GoEmotions)")

    label2id = {label: i for i, label in enumerate(LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def to_dataset(texts, labels):
        ds = Dataset.from_dict({"text": texts, "label": [label2id[x] for x in labels]})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH),
                      batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=label2id)
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "_distilbert_checkpoints"),
        num_train_epochs=EPOCHS, learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=128,
        warmup_steps=int(0.06 * EPOCHS * len(train_x) / BATCH_SIZE), weight_decay=0.01,
        bf16=use_cuda, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", save_total_limit=1, logging_steps=200, report_to=[], seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=to_dataset(train_x, train_y),
                      eval_dataset=to_dataset(val_x, val_y), data_collator=DataCollatorWithPadding(tokenizer))
    start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start
    model.eval()
    device = model.device

    def predict_proba(texts):
        out = []
        for i in range(0, len(texts), 256):
            enc = tokenizer(texts[i:i + 256], truncation=True, max_length=MAX_LENGTH, padding=True,
                            return_tensors="pt").to(device)
            with torch.no_grad():
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.vstack(out)

    # dair-ai test: the benchmark every earlier version reported on. It has no
    # neutral class, so the headline number takes argmax over the six shared
    # classes; predicting neutral there is never counted as correct.
    dair_x, dair_y = dair["test"]
    probs = predict_proba(dair_x)
    shared = [label2id[label] for label in LABELS if label != "neutral"]
    shared_names = [label for label in LABELS if label != "neutral"]
    dair_pred = [shared_names[i] for i in probs[:, shared].argmax(1)]
    dair_acc = accuracy_score(dair_y, dair_pred)
    dair_f1 = f1_score(dair_y, dair_pred, average="macro")

    go_x, go_y = go["test"]
    go_pred = [LABELS[i] for i in predict_proba(go_x).argmax(1)]
    go_acc = accuracy_score(go_y, go_pred)
    go_f1 = f1_score(go_y, go_pred, average="macro")

    print(f"\ndair-ai test:     accuracy={dair_acc:.4f}  macro-F1={dair_f1:.4f}")
    print(f"GoEmotions test:  accuracy={go_acc:.4f}  macro-F1={go_f1:.4f}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"dair-ai/emotion test (2,000 rows, 6 classes)\nAccuracy: {dair_acc:.4f}\nF1 (macro): {dair_f1:.4f}\n\n")
        f.write(classification_report(dair_y, dair_pred, digits=3))
        f.write(f"\nGoEmotions test (mapped, {len(go_y)} rows, 7 classes)\nAccuracy: {go_acc:.4f}\nF1 (macro): {go_f1:.4f}\n\n")
        f.write(classification_report(go_y, go_pred, digits=3))

    probe_probs = predict_proba([p[1] for p in PROBES])
    lines = ["Behavioural probes (v4: full dair-ai + GoEmotions, neutral class)", ""]
    passed = 0
    for (kind, text, acceptable), p in zip(PROBES, probe_probs):
        label = LABELS[int(p.argmax())]
        ok = label in acceptable
        passed += ok
        lines.append(f"{'PASS' if ok else 'FAIL'}  {kind:16} {text!r:45} -> {label:9} ({p.max() * 100:5.1f}%)")
    lines += ["", f"{passed}/{len(PROBES)} passed"]
    print("\n".join(lines))
    with open(PROBE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    metadata = {
        "model": f"{BASE_MODEL} (fine-tuned, v4: full dair-ai + GoEmotions, neutral class)",
        "train_rows": len(train_x),
        "train_rows_dair": len(dair["train"][0]),
        "train_rows_goemotions": len(go["train"][0]),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH, "learning_rate": LEARNING_RATE,
        "accuracy": float(dair_acc), "f1_macro": float(dair_f1),
        "goemotions_accuracy": float(go_acc), "goemotions_f1_macro": float(go_f1),
        "probes_passed": passed, "probes_total": len(PROBES),
        "train_time_seconds": train_time, "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "labels": LABELS,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"\nSaved model to {MODEL_DIR}/ and metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
