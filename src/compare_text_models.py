"""
Compare several classical ML models for text emotion detection, on top of
the same TF-IDF features used by train_text_model.py.

This does NOT replace results/emotion_model.pkl (the model app/app.py
actually serves) - it trains its own set of models side by side purely for
comparison, and saves them under results/models_comparison/ so you can
plug a different one into the app later if you want to.

Run from the project root with:

    python src/compare_text_models.py

Outputs (all under results/):
    text_model_comparison.csv   - full metrics table
    text_model_comparison.md    - the same table, formatted for a report
    text_model_comparison.png   - accuracy + F1 bar charts
    models_comparison/text_*.pkl + text_tfidf_vectorizer.pkl
"""

import os
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from chart_utils import render_emphasis_bar_chart

TRAIN_PATH = "data/raw/train.txt"
TEST_PATH = "data/raw/test.txt"

RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models_comparison")
COMPARISON_CSV = os.path.join(RESULTS_DIR, "text_model_comparison.csv")
COMPARISON_MD = os.path.join(RESULTS_DIR, "text_model_comparison.md")
COMPARISON_CHART = os.path.join(RESULTS_DIR, "text_model_comparison.png")


def load_data(file_path):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.rsplit(";", 1)
            texts.append(text)
            labels.append(label)
    return texts, labels


def slugify(name):
    return name.lower().replace(" ", "_")


def build_models():
    """Logistic Regression is left exactly as train_text_model.py configures
    it, so it reproduces that script's ~87% baseline here and gives every
    other model a fair, identical-features comparison point. The newer
    models add class_weight="balanced" (or an equivalent sample_weight)
    since this dataset is meaningfully imbalanced (joy/sadness dominate,
    surprise is rare) and that flag costs nothing on a laptop-scale job.

    Linear SVM is wrapped in CalibratedClassifierCV (Platt/sigmoid scaling,
    5-fold internal CV) because LinearSVC has no predict_proba - without
    this, anything reading confidence scores off this model (see
    emotion_logic.predict_with_confidence's fallback) silently gets a fake
    one-hot 100%/0% vector instead of a real probability distribution."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=5000, class_weight="balanced"), cv=5
        ),
        "Random Forest": RandomForestClassifier(
            # 100 trees, uncapped depth, over 5000 TF-IDF features already
            # saves as tens of MB per model (see compress= on joblib.dump
            # below) - more trees barely moves accuracy on this dataset but
            # multiplies file size, which matters once these get synced to
            # OneDrive.
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def write_markdown_table(df, path, title):
    cols = ["model", "accuracy", "f1_macro", "train_time_seconds"]
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
        f"Best test accuracy: **{df.iloc[0]['model']}** "
        f"({df.iloc[0]['accuracy']:.1%}). Weighted F1 (accounts for class "
        "imbalance differently than macro F1) is also saved in the CSV "
        "alongside these columns."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    print("Vectorizing text (TF-IDF, max_features=5000)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "text_tfidf_vectorizer.pkl"), compress=3)

    results = []
    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()

        if name == "Gradient Boosting":
            # GradientBoostingClassifier has no class_weight parameter -
            # pass balanced sample weights instead for the same effect.
            sample_weight = compute_sample_weight("balanced", y_train)
            model.fit(X_train_vec, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train_vec, y_train)

        train_time = time.perf_counter() - start

        y_pred = model.predict(X_test_vec)
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

        joblib.dump(model, os.path.join(MODELS_DIR, f"text_{slugify(name)}.pkl"), compress=3)

    df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(COMPARISON_CSV, index=False)
    write_markdown_table(df, COMPARISON_MD, "Text model comparison")

    best_model = df.iloc[0]["model"]

    # Two small-multiple charts (accuracy, F1) rather than one dual-axis
    # chart - accuracy and F1 are different scales/meanings, and overlaying
    # two y-axes on one plot is the #1 chart anti-pattern (it invents a
    # correlation that isn't really there). Each uses the same
    # emphasis-on-the-winner style as the app's confidence chart.
    for metric, title in [("accuracy", "Test accuracy"), ("f1_macro", "Test macro F1")]:
        fig = render_emphasis_bar_chart(
            list(zip(df["model"], df[metric])), best_model, xlabel=f"{title} (%)"
        )
        fig.savefig(os.path.join(RESULTS_DIR, f"text_model_comparison_{metric}.png"), dpi=150)

    print(f"\nComparison table saved to: {COMPARISON_CSV}")
    print(f"Comparison report saved to: {COMPARISON_MD}")
    print(
        "Comparison charts saved to: "
        f"{RESULTS_DIR}/text_model_comparison_accuracy.png and "
        f"{RESULTS_DIR}/text_model_comparison_f1_macro.png"
    )
    print(f"Fitted models saved under: {MODELS_DIR}/")
    print(
        f"\nBest by accuracy: {best_model} ({df.iloc[0]['accuracy']:.1%}). "
        "To make the app use it, copy "
        f"{MODELS_DIR}/text_{slugify(best_model)}.pkl over results/emotion_model.pkl "
        f"and {MODELS_DIR}/text_tfidf_vectorizer.pkl over results/vectorizer.pkl."
    )


if __name__ == "__main__":
    main()
