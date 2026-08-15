import os
import json
import re
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

AUDIO_DATA_PATH = "data/audio"
RESULTS_DIR = "results"

REPORT_PATH = os.path.join(RESULTS_DIR, "audio_classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "audio_confusion_matrix.png")
MODEL_PATH = os.path.join(RESULTS_DIR, "audio_emotion_model.pkl")
METADATA_PATH = os.path.join(RESULTS_DIR, "audio_model_metadata.json")

# RAVDESS: NN-NN-NN-NN-NN-NN-NN.wav, actor is the last field.
# CREMA-D: ActorID_Sentence_Emotion_Level.wav, actor is the first field.
# Fairness audit (tests/actor_leakage_audit.py) found the previous plain
# train_test_split(..., stratify=y) put the same actor's voice in both
# train and test 100% of the time - the model could partly learn "whose
# voice is this" as a shortcut instead of generalisable emotion cues.
# Grouping by actor here (see StratifiedGroupKFold below) closes that.
RAVDESS_PATTERN = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$", re.IGNORECASE)
CREMAD_PATTERN = re.compile(r"^(\d{4})_[A-Z]{3}_[A-Z]{3}_(LO|MD|HI|XX)\.wav$", re.IGNORECASE)


def actor_of(file_name):
    m = RAVDESS_PATTERN.match(file_name)
    if m:
        return f"ravdess-{m.group(7)}"
    m = CREMAD_PATTERN.match(file_name)
    if m:
        return f"cremad-{m.group(1)}"
    return f"unknown-{file_name}"  # own group of size 1, never causes leakage


def extract_features(file_path, target_sr=22050, max_duration=5):
    y, sr = librosa.load(file_path, sr=target_sr)

    # Trim silence at beginning/end
    y, _ = librosa.effects.trim(y, top_db=20)

    # Avoid empty audio after trimming
    if len(y) == 0:
        raise ValueError("Audio is empty after trimming silence.")

    # Normalize volume
    y = librosa.util.normalize(y)

    # Make all clips same length
    target_length = target_sr * max_duration

    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, max(0, target_length - len(y))))

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=target_sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms.T, axis=0)

    feature_vector = np.hstack([mfcc_mean, zcr_mean, chroma_mean, rms_mean])
    return feature_vector


def load_audio_dataset(base_path):
    features = []
    labels = []
    groups = []
    processed = 0

    for emotion in sorted(os.listdir(base_path)):
        emotion_folder = os.path.join(base_path, emotion)

        if not os.path.isdir(emotion_folder):
            continue

        for file_name in sorted(os.listdir(emotion_folder)):
            if file_name.lower().endswith(".wav"):
                file_path = os.path.join(emotion_folder, file_name)

                try:
                    feature_vector = extract_features(file_path)
                    features.append(feature_vector)
                    labels.append(emotion)
                    groups.append(actor_of(file_name))
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

                processed += 1
                if processed % 250 == 0:
                    print(f"  ...{processed} files processed", flush=True)

    return np.array(features), np.array(labels), np.array(groups)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading audio dataset...")
    X, y, groups = load_audio_dataset(AUDIO_DATA_PATH)

    if len(X) == 0:
        raise ValueError("No audio features were loaded. Check your folder structure and .wav files.")

    print(f"Loaded {len(X)} audio samples.")

    class_counts = Counter(y)
    print("Class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count}")

    # StratifiedGroupKFold instead of a plain train_test_split(stratify=y):
    # keeps each actor entirely in train or entirely in test (no voice
    # leakage - see actor_of() above) while still balancing emotion
    # classes across the split as closely as the grouping constraint
    # allows. First fold's test half is used as the ~20% held-out set.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_actors = set(groups[train_idx])
    test_actors = set(groups[test_idx])
    leaked_actors = train_actors & test_actors
    print(f"\nActor-grouped split check: {len(leaked_actors)} actors leaked between train/test (should be 0)")

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X.shape[1]}")

    print("\nTraining improved audio model (scaled SVM)...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", probability=True, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)

    print("\nAudio Results:")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}\n")
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Class distribution:\n")
        for label, count in sorted(class_counts.items()):
            f.write(f"{label}: {count}\n")
        f.write(f"\nFeature dimension: {X.shape[1]}\n\n")
        f.write(report)

    # Confusion matrix
    classes = model.named_steps["svm"].classes_
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix - Improved Audio Emotion Detection")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.show(block=False)
    plt.pause(0.1)

    # Save model
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "num_samples": int(len(X)),
        "num_train_samples": int(len(X_train)),
        "num_test_samples": int(len(X_test)),
        "accuracy": float(accuracy),
        "classes": list(classes),
        "class_distribution": dict(class_counts),
        "feature_dimension": int(X.shape[1]),
        "feature_description": "40 MFCC mean + ZCR mean + chroma mean + RMS mean",
        "preprocessing": {
            "sample_rate": 22050,
            "max_duration_seconds": 5,
            "trim_silence": True,
            "normalize_audio": True,
            "fixed_length_padding": True,
            "scaling": "StandardScaler"
        },
        "model": "Pipeline(StandardScaler + SVC(kernel='linear', probability=True, class_weight='balanced'))",
        "split_method": "StratifiedGroupKFold(n_splits=5) grouped by actor - no speaker leakage between train/test",
        "leaked_actors_between_train_test": len(leaked_actors),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nAudio report saved to: {REPORT_PATH}")
    print(f"Audio confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Audio model saved to: {MODEL_PATH}")
    print(f"Audio metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()