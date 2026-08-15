import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Paths
train_path = "data/raw/train.txt"
val_path = "data/raw/val.txt"
test_path = "data/raw/test.txt"
music_path = "data/music.csv"

val_report_path = "results/val_classification_report.txt"
test_report_path = "results/test_classification_report.txt"
example_predictions_path = "results/example_predictions.txt"
confusion_matrix_path = "results/confusion_matrix.png"


def load_data(file_path):
    texts = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text, label = line.strip().split(";")
            texts.append(text)
            labels.append(label)

    return texts, labels


def recommend_music(emotion, music_df):
    songs = music_df[music_df["emotion"] == emotion]

    if songs.empty:
        return None

    return songs.sample(1).iloc[0]


# Make sure results folder exists
os.makedirs("results", exist_ok=True)

# Load datasets
print("Loading data...")
X_train, y_train = load_data(train_path)
X_val, y_val = load_data(val_path)
X_test, y_test = load_data(test_path)

# Vectorize text
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Validation results
print("\nValidation Results:")
y_val_pred = model.predict(X_val_vec)
val_report = classification_report(y_val, y_val_pred)
print(val_report)

with open(val_report_path, "w", encoding="utf-8") as f:
    f.write(val_report)

# Test results
print("\nTest Results:")
y_test_pred = model.predict(X_test_vec)
test_report = classification_report(y_test, y_test_pred)
print(test_report)

with open(test_report_path, "w", encoding="utf-8") as f:
    f.write(test_report)

# Save confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax)
plt.title("Confusion Matrix - Text Emotion Detection")
plt.tight_layout()

# Save image
plt.savefig(confusion_matrix_path)

plt.show(block=False)
plt.pause(0.1)

print(f"Confusion matrix saved to: {confusion_matrix_path}")

# Save model and vectorizer
joblib.dump(model, "results/emotion_model.pkl")
joblib.dump(vectorizer, "results/vectorizer.pkl")

print("\nModel and vectorizer saved in /results/")

# Load music dataset
music_df = pd.read_csv(music_path)

# Save example predictions
example_inputs = [
    "i feel amazing today",
    "i am really upset",
    "i am scared right now",
    "i love this so much",
]

with open(example_predictions_path, "w", encoding="utf-8") as f:
    for text in example_inputs:
        user_vec = vectorizer.transform([text])
        prediction = model.predict(user_vec)[0]
        recommendation = recommend_music(prediction, music_df)

        f.write(f"Input: {text}\n")
        f.write(f"Predicted emotion: {prediction}\n")

        if recommendation is not None:
            f.write(
                f"Recommended song: {recommendation['song']} - {recommendation['artist']}\n\n"
            )
        else:
            f.write("No song recommendation found.\n\n")

print("Example predictions saved in /results/")

# Live recommender
print("\nEmotion → Music Recommender 🎵")
while True:
    user_input = input("Enter text (or 'quit'): ").strip()

    if user_input.lower() == "quit":
        break

    if not user_input:
        print("Please enter some text.\n")
        continue

    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    recommendation = recommend_music(prediction, music_df)

    print(f"\nPredicted emotion: {prediction}")

    if recommendation is not None:
        print(f"Recommended song: {recommendation['song']} - {recommendation['artist']}\n")
    else:
        print("No song recommendation found for this emotion.\n")