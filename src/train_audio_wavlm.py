"""
Audio model v2: frozen WavLM Base+ embeddings + logistic regression.

See README "Audio model v2" and emotion_logic.py's WavLM section. Same data
(RAVDESS + CREMA-D in data/audio/<emotion>/), same actor-grouped
evaluation (StratifiedGroupKFold, n_splits=5, random_state=42 - see
train_audio_model.py's actor_of for why grouping by speaker matters), so
the numbers compare directly with the classical models.

Steps:
1. Extract mean-pooled embeddings from every WavLM layer for every clip
   (cached to results/_wavlm_embeddings.npz - gitignored, re-used on rerun).
2. Pick the layer by grouped cross-validation. Emotion is mostly carried by
   the middle layers (prosody/voice quality), not the last ones (closer to
   phonetic content); this measures it rather than assuming.
3. Report 5-fold actor-grouped accuracy, with the layer chosen inside each
   fold on that fold's training actors only (no test-fold peeking).
4. Fit the final classifier on all clips and save it as
   results/audio_emotion_model_wavlm.pkl (a small dict: sklearn pipeline +
   layer index + encoder id).

Run from the project root (a CUDA GPU makes step 1 take minutes, not hours):

    python src/train_audio_wavlm.py [path/to/audio_dir]
"""

import json
import os
import sys
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from emotion_logic import WAVLM_MODEL_ID, wavlm_layer_embeddings
from train_audio_model import actor_of

AUDIO_DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/audio"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "audio_emotion_model_wavlm.pkl")
METADATA_PATH = os.path.join(RESULTS_DIR, "audio_wavlm_model_metadata.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "audio_wavlm_classification_report.txt")
CACHE_PATH = os.path.join(RESULTS_DIR, "_wavlm_embeddings.npz")
C = 0.1  # inverse L2 strength; 768-dim features from ~5.6k clips need real regularisation


def make_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def list_clips(base_path):
    paths, labels, groups = [], [], []
    for emotion in sorted(os.listdir(base_path)):
        folder = os.path.join(base_path, emotion)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".wav"):
                paths.append(os.path.join(folder, name))
                labels.append(emotion)
                groups.append(actor_of(name))
    return paths, np.array(labels), np.array(groups)


def extract_all(paths):
    """Returns (X, kept_mask). Clips that can't be embedded (e.g. the few
    completely silent CREMA-D recordings) are skipped, same as
    train_audio_model.py's loader does."""
    import torch

    if os.path.exists(CACHE_PATH):
        cached = np.load(CACHE_PATH, allow_pickle=True)
        if list(cached["paths"]) == paths:
            print(f"Using cached embeddings from {CACHE_PATH}")
            return cached["X"].astype(np.float32), cached["kept"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting WavLM embeddings for {len(paths)} clips on {device}...")
    start = time.perf_counter()
    X, kept = [], np.zeros(len(paths), dtype=bool)
    for i, path in enumerate(paths):
        try:
            X.append(wavlm_layer_embeddings(path, device=device))
            kept[i] = True
        except ValueError as exc:
            print(f"  skipping {path}: {exc}")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)}  ({time.perf_counter() - start:.0f}s)", flush=True)
    X = np.stack(X).astype(np.float32)
    np.savez(CACHE_PATH, X=X.astype(np.float16), paths=np.array(paths), kept=kept)
    return X, kept


def best_layer(X, labels, groups, idx):
    scores = [
        cross_val_score(make_classifier(), X[idx, layer], labels[idx], groups=groups[idx],
                        cv=GroupKFold(3), n_jobs=-1).mean()
        for layer in range(X.shape[1])
    ]
    return int(np.argmax(scores)), scores


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    paths, labels, groups = list_clips(AUDIO_DATA_PATH)
    if not paths:
        raise ValueError(f"No .wav files under {AUDIO_DATA_PATH}/<emotion>/ - see README 'Dataset Requirements'.")
    print(f"{len(paths)} clips, {len(set(groups))} actors, classes: {dict(sorted(Counter(labels).items()))}")
    X, kept = extract_all(paths)
    labels, groups = labels[kept], groups[kept]
    if not kept.all():
        print(f"Skipped {(~kept).sum()} clip(s); training on {kept.sum()}")

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X[:, 0], labels, groups))
    accs, f1s, layers, all_true, all_pred = [], [], [], [], []
    for k, (tr, te) in enumerate(folds):
        assert not set(groups[tr]) & set(groups[te]), "actor leaked between train and test"
        layer, _ = best_layer(X, labels, groups, tr)
        clf = make_classifier().fit(X[tr, layer], labels[tr])
        pred = clf.predict(X[te, layer])
        accs.append(accuracy_score(labels[te], pred))
        f1s.append(f1_score(labels[te], pred, average="macro"))
        layers.append(layer)
        all_true += list(labels[te])
        all_pred += list(pred)
        print(f"  fold {k + 1}: layer {layer}  accuracy={accs[-1]:.4f}  macro-F1={f1s[-1]:.4f}")
    print(f"5-fold actor-grouped accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}, macro-F1 {np.mean(f1s):.4f}")

    final_layer, layer_scores = best_layer(X, labels, groups, np.arange(len(labels)))
    final = make_classifier().fit(X[:, final_layer], labels)
    joblib.dump({"model": final, "layer": final_layer, "encoder": WAVLM_MODEL_ID}, MODEL_PATH, compress=3)

    report = classification_report(all_true, all_pred, digits=3)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"WavLM ({WAVLM_MODEL_ID}) layer embeddings + logistic regression\n")
        f.write(f"5-fold actor-grouped CV (predictions pooled across folds), {len(labels)} clips\n")
        f.write(f"Accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}\nF1 (macro): {np.mean(f1s):.4f}\n\n{report}")
    metadata = {
        "model": f"{WAVLM_MODEL_ID} (frozen) mean-pooled layer {final_layer} + StandardScaler + LogisticRegression(C={C})",
        "num_samples": int(len(labels)),
        "num_actors": int(len(set(groups))),
        "classes": sorted(set(labels)),
        "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), grouped by actor; layer chosen per fold on training actors only",
        "cv_accuracy_mean": float(np.mean(accs)), "cv_accuracy_std": float(np.std(accs)),
        "cv_f1_macro_mean": float(np.mean(f1s)), "cv_fold_accuracies": [float(a) for a in accs],
        "cv_fold_layers": layers,
        "final_layer": final_layer,
        "layer_cv_scores_all_data": [float(s) for s in layer_scores],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved {MODEL_PATH} (layer {final_layer}), {METADATA_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
