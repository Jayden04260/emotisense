"""
One-off audit: does the merged RAVDESS+CREMA-D audio model perform
comparably on both source datasets, or has it partly learned "which
dataset is this" (recording equipment/conditions) as a shortcut instead
of genuine emotion cues?

Replicates train_audio_model.py's exact feature-extraction order and
StratifiedGroupKFold split (same random_state, grouped by actor - see
"Fairness & Generalisation Audit" in the README for why) so the held-out
test set here is identical to the one accuracy was reported against, then
breaks that same test set down by source dataset.

RAVDESS filenames: 7 dash-separated numeric fields, e.g.
    03-01-05-01-01-01-01.wav
CREMA-D filenames: 4 underscore-separated fields, e.g.
    1001_DFA_ANG_XX.wav
The two conventions can't collide, so origin is read straight off the
filename - no tagging needed at sort time.

Run from the project root with:

    python tests/dataset_shortcut_audit.py
"""

import os
import re

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train_audio_model import actor_of, extract_features  # noqa: E402

AUDIO_DATA_PATH = "data/audio"
MODEL_PATH = "results/audio_emotion_model.pkl"

RAVDESS_PATTERN = re.compile(r"^\d{2}(-\d{2}){6}\.wav$", re.IGNORECASE)
CREMAD_PATTERN = re.compile(r"^\d{4}_[A-Z]{3}_[A-Z]{3}_(LO|MD|HI|XX)\.wav$", re.IGNORECASE)


def source_of(file_name):
    if RAVDESS_PATTERN.match(file_name):
        return "ravdess"
    if CREMAD_PATTERN.match(file_name):
        return "cremad"
    return "unknown"


def load_dataset_with_source(base_path):
    features, labels, sources, groups = [], [], [], []
    processed = 0

    for emotion in sorted(os.listdir(base_path)):
        emotion_folder = os.path.join(base_path, emotion)
        if not os.path.isdir(emotion_folder):
            continue

        for file_name in sorted(os.listdir(emotion_folder)):
            if not file_name.lower().endswith(".wav"):
                continue
            file_path = os.path.join(emotion_folder, file_name)
            try:
                feature_vector = extract_features(file_path)
                features.append(feature_vector)
                labels.append(emotion)
                sources.append(source_of(file_name))
                groups.append(actor_of(file_name))
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

            processed += 1
            if processed % 1000 == 0:
                print(f"  ...{processed} files processed", flush=True)

    return np.array(features), np.array(labels), np.array(sources), np.array(groups)


def main():
    print("Re-extracting features (same order as train_audio_model.py)...")
    X, y, source, groups = load_dataset_with_source(AUDIO_DATA_PATH)
    print(f"Loaded {len(X)} samples.\n")

    # Identical call (same random_state, grouped by actor) to
    # train_audio_model.py's split, so X_test here is the same held-out
    # set that model's reported accuracy came from.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    idx_train, idx_test = next(splitter.split(X, y, groups=groups))
    X_test, y_test, source_test = X[idx_test], y[idx_test], source[idx_test]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    print(f"Overall held-out accuracy: {overall_acc:.4f} (n={len(y_test)})\n")

    print("Accuracy broken down by source dataset:")
    for src in sorted(set(source_test)):
        mask = source_test == src
        acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"  {src:10} n={mask.sum():4}  accuracy={acc:.4f}")

    print()
    print("If the model had learned 'which dataset' as a shortcut instead")
    print("of genuine emotion cues, we'd expect a large accuracy gap between")
    print("the two rows above (e.g. near-perfect on one, poor on the other).")

    print()
    print("Confusion matrix, RAVDESS-origin test samples only:")
    classes = sorted(set(y_test))
    mask = source_test == "ravdess"
    if mask.sum() > 0:
        cm = confusion_matrix(y_test[mask], y_pred[mask], labels=classes)
        print("            " + " ".join(f"{c[:6]:>8}" for c in classes))
        for cls, row in zip(classes, cm):
            print(f"{cls[:10]:10}  " + " ".join(f"{v:8d}" for v in row))

    print()
    print("Confusion matrix, CREMA-D-origin test samples only:")
    mask = source_test == "cremad"
    if mask.sum() > 0:
        cm = confusion_matrix(y_test[mask], y_pred[mask], labels=classes)
        print("            " + " ".join(f"{c[:6]:>8}" for c in classes))
        for cls, row in zip(classes, cm):
            print(f"{cls[:10]:10}  " + " ".join(f"{v:8d}" for v in row))


if __name__ == "__main__":
    main()
