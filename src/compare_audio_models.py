"""
Compare several classical ML models for audio emotion detection, on top of
the same MFCC/ZCR/Chroma/RMS features used by train_audio_model.py.

This does NOT replace results/audio_emotion_model.pkl (the model
app/app.py actually serves) - it trains its own set of models side by
side purely for comparison, and saves them under results/models_comparison/
so you can plug a different one into the app later if you want to.

Feature extraction (the librosa pass over every WAV file) is the slow part
of this script, same as train_audio_model.py - expect it to take a while
the first time. To avoid re-extracting features once per model, this
script extracts them ONCE and reuses that same train/test split (same
random_state=42 as train_audio_model.py) for every model, so the
comparison is apples-to-apples with your existing baseline.

Run from the project root with:

    python src/compare_audio_models.py

Outputs (all under results/):
    audio_model_comparison.csv   - full metrics table
    audio_model_comparison.md    - the same table, formatted for a report
    audio_model_comparison_accuracy.png / _f1_macro.png
    models_comparison/audio_*.pkl
"""

import os
import re
import time
from collections import Counter

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from chart_utils import render_emphasis_bar_chart

AUDIO_DATA_PATH = "data/audio"
RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models_comparison")
COMPARISON_CSV = os.path.join(RESULTS_DIR, "audio_model_comparison.csv")
COMPARISON_MD = os.path.join(RESULTS_DIR, "audio_model_comparison.md")

# See train_audio_model.py for why this exists: a plain
# train_test_split(stratify=y) was found (tests/actor_leakage_audit.py)
# to put the same actor's voice in both train and test 100% of the time.
RAVDESS_PATTERN = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$", re.IGNORECASE)
CREMAD_PATTERN = re.compile(r"^(\d{4})_[A-Z]{3}_[A-Z]{3}_(LO|MD|HI|XX)\.wav$", re.IGNORECASE)


def actor_of(file_name):
    m = RAVDESS_PATTERN.match(file_name)
    if m:
        return f"ravdess-{m.group(7)}"
    m = CREMAD_PATTERN.match(file_name)
    if m:
        return f"cremad-{m.group(1)}"
    return f"unknown-{file_name}"


def extract_features(file_path, target_sr=22050, max_duration=5):
    """Identical to train_audio_model.py / app/app.py, duplicated rather
    than imported so this script has no import-time side effects and can
    be read/run standalone."""
    y, sr = librosa.load(file_path, sr=target_sr)

    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) == 0:
        raise ValueError("Audio is empty after trimming silence.")

    y = librosa.util.normalize(y)

    target_length = target_sr * max_duration
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, max(0, target_length - len(y))))

    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=target_sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms.T, axis=0)

    return np.hstack([mfcc_mean, zcr_mean, chroma_mean, rms_mean])


def load_audio_dataset(base_path):
    features, labels, groups = [], [], []
    processed = 0
    for emotion in sorted(os.listdir(base_path)):
        emotion_folder = os.path.join(base_path, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        for file_name in sorted(os.listdir(emotion_folder)):
            if file_name.lower().endswith(".wav"):
                file_path = os.path.join(emotion_folder, file_name)
                try:
                    features.append(extract_features(file_path))
                    labels.append(emotion)
                    groups.append(actor_of(file_name))
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                processed += 1
                if processed % 1000 == 0:
                    print(f"  ...{processed} files processed", flush=True)
    return np.array(features), np.array(labels), np.array(groups)


def slugify(name):
    return name.lower().replace(" ", "_")


def build_models():
    """The SVM here matches train_audio_model.py's configuration exactly
    (same scaler + linear SVC + class_weight="balanced") so it reproduces
    that script's baseline accuracy within this same run and gives the new
    models a fair, identical-features comparison point."""
    return {
        "SVM (baseline)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="linear", probability=True, class_weight="balanced")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            # n_jobs capped at 2 (not -1) - this project runs on a machine
            # with very little RAM, and each parallel worker holds its own
            # copy of the training data during tree building.
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=2
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def write_markdown_table(df, path, title):
    headers = ["Model", "Accuracy", "F1 Score (macro)", "Training Time (s)"]
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in df.iterrows():
        lines.append(
            "| {model} | {acc:.1%} | {f1:.1%} | {t:.2f} |".format(
                model=row["model"],
                acc=row["accuracy"],
                f1=row["f1_macro"],
                t=row["train_time_seconds"],
            )
        )
    lines.append("")
    lines.append(
        f"Best test accuracy: **{df.iloc[0]['model']}** ({df.iloc[0]['accuracy']:.1%}). "
        "All models were trained and evaluated on the same actor-grouped "
        "~80/20 split (StratifiedGroupKFold, random_state=42 - no speaker "
        "appears in both train and test, see 'Fairness & Generalisation "
        "Audit' in the README) over the same MFCC + ZCR + Chroma + RMS "
        "features, so accuracy differences come from the classifier, not "
        "the data split."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading audio dataset (this is the slow part - one librosa pass per file)...")
    X, y, groups = load_audio_dataset(AUDIO_DATA_PATH)
    if len(X) == 0:
        raise ValueError("No audio features were loaded. Check your folder structure and .wav files.")

    print(f"Loaded {len(X)} audio samples.")
    print("Class distribution:", dict(sorted(Counter(y).items())))

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    leaked = set(groups[train_idx]) & set(groups[test_idx])
    print(f"Actor-grouped split check: {len(leaked)} actors leaked between train/test (should be 0)")
    print(f"Training samples: {len(X_train)}  Test samples: {len(X_test)}  Feature dimension: {X.shape[1]}")

    results = []
    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()

        if name == "Gradient Boosting":
            # GradientBoostingClassifier has no class_weight parameter -
            # pass balanced sample weights instead for the same effect.
            sample_weight = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        train_time = time.perf_counter() - start

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        print(
            f"  accuracy={accuracy:.4f}  f1_macro={f1_macro:.4f}  "
            f"f1_weighted={f1_weighted:.4f}  train_time={train_time:.2f}s"
        )

        results.append(
            {
                "model": name,
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "train_time_seconds": train_time,
            }
        )

        joblib.dump(model, os.path.join(MODELS_DIR, f"audio_{slugify(name)}.pkl"), compress=3)

    df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(COMPARISON_CSV, index=False)
    write_markdown_table(df, COMPARISON_MD, "Audio model comparison")

    best_model = df.iloc[0]["model"]

    for metric, title in [("accuracy", "Test accuracy"), ("f1_macro", "Test macro F1")]:
        fig = render_emphasis_bar_chart(
            list(zip(df["model"], df[metric])), best_model, xlabel=f"{title} (%)"
        )
        fig.savefig(os.path.join(RESULTS_DIR, f"audio_model_comparison_{metric}.png"), dpi=150)

    print(f"\nComparison table saved to: {COMPARISON_CSV}")
    print(f"Comparison report saved to: {COMPARISON_MD}")
    print(
        "Comparison charts saved to: "
        f"{RESULTS_DIR}/audio_model_comparison_accuracy.png and "
        f"{RESULTS_DIR}/audio_model_comparison_f1_macro.png"
    )
    print(f"Fitted models saved under: {MODELS_DIR}/")
    print(
        f"\nBest by accuracy: {best_model} ({df.iloc[0]['accuracy']:.1%}). "
        "To make the app use it, copy "
        f"{MODELS_DIR}/audio_{slugify(best_model)}.pkl over results/audio_emotion_model.pkl."
    )


if __name__ == "__main__":
    main()
