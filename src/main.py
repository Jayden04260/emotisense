import os
import joblib
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

# Load models
text_model = joblib.load("results/emotion_model.pkl")
vectorizer = joblib.load("results/vectorizer.pkl")
audio_model = joblib.load("results/audio_emotion_model.pkl")

music_df = pd.read_csv("data/music.csv")

TEMP_AUDIO_PATH = "results/temp_recording.wav"


def recommend_music(emotion):
    songs = music_df[music_df["emotion"] == emotion]
    if songs.empty:
        return None
    return songs.sample(1).iloc[0]


def text_mode():
    print("\n--- TEXT MODE ---")
    while True:
        user_input = input("Enter text (or 'quit'): ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            print("Please enter some text.\n")
            continue

        vec = vectorizer.transform([user_input])
        emotion = text_model.predict(vec)[0]

        print(f"\n🎭 Emotion: {emotion}")

        song = recommend_music(emotion)
        if song is not None:
            print(f"🎵 Recommendation: {song['song']} - {song['artist']}\n")
        else:
            print("No recommendation found.\n")


def extract_features(file_path, target_sr=22050, max_duration=5):
    y, sr = librosa.load(file_path, sr=target_sr)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) == 0:
        raise ValueError("Audio is empty after trimming silence.")

    # Normalize volume
    y = librosa.util.normalize(y)

    # Force fixed length
    target_length = target_sr * max_duration

    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, max(0, target_length - len(y))))

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=target_sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms.T, axis=0)

    feature_vector = np.hstack([mfcc_mean, zcr_mean, chroma_mean, rms_mean])
    return feature_vector


def show_audio_prediction(features):
    emotion = audio_model.predict(features)[0]
    print(f"\n🎤 Predicted emotion: {emotion}")

    if hasattr(audio_model, "predict_proba"):
        probabilities = audio_model.predict_proba(features)[0]

        if hasattr(audio_model, "classes_"):
            class_names = audio_model.classes_
        elif hasattr(audio_model, "named_steps") and "svm" in audio_model.named_steps:
            class_names = audio_model.named_steps["svm"].classes_
        else:
            class_names = [f"class_{i}" for i in range(len(probabilities))]

        scores = list(zip(class_names, probabilities))
        scores.sort(key=lambda x: x[1], reverse=True)

        print("Confidence scores:")
        for label, score in scores:
            print(f"  {label}: {score:.2f}")

    # Map audio emotion labels to music emotion labels
    audio_to_music_emotion = {
        "happy": "joy",
        "sad": "sadness",
        "angry": "anger",
        "fear": "fear",
        "neutral": "joy",
    }

    mapped_emotion = audio_to_music_emotion.get(emotion, emotion)
    song = recommend_music(mapped_emotion)

    if song is not None:
        print(f"\n🎵 Recommendation: {song['song']} - {song['artist']}\n")
    else:
        print("\nNo song recommendation found.\n")

    return emotion


def record_audio(duration=6, sample_rate=22050, output_path=TEMP_AUDIO_PATH):
    print(f"\nRecording for {duration} seconds... Speak clearly now.")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    recording = recording.flatten()

    if np.max(np.abs(recording)) > 0:
        recording = recording / np.max(np.abs(recording))

    recording_int16 = np.int16(recording * 32767)

    write(output_path, sample_rate, recording_int16)
    print(f"Recording saved to: {output_path}")
    return output_path


def audio_file_mode():
    print("\n--- AUDIO FILE MODE ---")

    audio_base = "data/audio"
    all_audio_files = []

    # Collect audio files
    for emotion in os.listdir(audio_base):
        emotion_folder = os.path.join(audio_base, emotion)

        if not os.path.isdir(emotion_folder):
            continue

        for file_name in os.listdir(emotion_folder):
            if file_name.lower().endswith(".wav"):
                full_path = os.path.join(emotion_folder, file_name)
                all_audio_files.append({
                    "emotion": emotion,
                    "file_name": file_name,
                    "path": full_path
                })

    if not all_audio_files:
        print("No audio files found.\n")
        return

    emotions = sorted(list({item["emotion"] for item in all_audio_files}))

    while True:
        print("\nAvailable emotions:")
        for i, emotion in enumerate(emotions, start=1):
            count = sum(1 for item in all_audio_files if item["emotion"] == emotion)
            print(f"{i}. {emotion} ({count} files)")

        print("\nCommands:")
        print("  Enter an emotion number to view files")
        print("  Type 'quit' to go back")

        emotion_choice = input("Choose emotion: ").strip().lower()

        if emotion_choice == "quit":
            break

        if not emotion_choice.isdigit():
            print("Please enter a valid number or 'quit'.\n")
            continue

        emotion_index = int(emotion_choice) - 1

        if emotion_index < 0 or emotion_index >= len(emotions):
            print("Choice out of range.\n")
            continue

        selected_emotion = emotions[emotion_index]

        emotion_files = [
            item for item in all_audio_files if item["emotion"] == selected_emotion
        ]
        emotion_files.sort(key=lambda x: x["file_name"])

        while True:
            print(f"\n--- {selected_emotion.upper()} FILES ---")
            display_files = emotion_files[:10]

            for i, item in enumerate(display_files, start=1):
                print(f"{i}. {item['file_name']}")

            print("\nCommands:")
            print("  Enter a number to analyse a file")
            print("  Type 'play <number>' to listen to a file")
            print("  Type 'back' to choose another emotion")
            print("  Type 'quit' to go back")

            choice = input("Enter choice: ").strip().lower()

            if choice == "quit":
                return

            if choice == "back":
                break

            if choice.startswith("play "):
                number_part = choice.replace("play ", "").strip()

                if not number_part.isdigit():
                    print("Invalid play command.\n")
                    continue

                index = int(number_part) - 1

                if index < 0 or index >= len(display_files):
                    print("Choice out of range.\n")
                    continue

                selected = display_files[index]

                try:
                    data, samplerate = sf.read(selected["path"])
                    print(f"\nPlaying: {selected['file_name']} ({selected['emotion']})")
                    sd.play(data, samplerate)
                    sd.wait()
                    print("Playback finished.\n")
                except Exception as e:
                    print(f"Playback error: {e}\n")

                continue

            if not choice.isdigit():
                print("Please enter a valid number, 'play <number>', 'back', or 'quit'.\n")
                continue

            index = int(choice) - 1

            if index < 0 or index >= len(display_files):
                print("Choice out of range.\n")
                continue

            selected = display_files[index]

            try:
                features = extract_features(selected["path"]).reshape(1, -1)
                print(f"\nSelected file: {selected['file_name']}")
                print(f"Actual folder emotion: {selected['emotion']}")
                show_audio_prediction(features)
            except Exception as e:
                print(f"Error reading file: {e}\n")


def audio_microphone_mode():
    print("\n--- MICROPHONE MODE ---")
    while True:
        user_input = input("Press Enter to record, or type 'quit' to go back: ").strip().lower()

        if user_input == "quit":
            break

        try:
            recorded_file = record_audio(duration=6)
            features = extract_features(recorded_file).reshape(1, -1)
            show_audio_prediction(features)
        except Exception as e:
            print(f"Microphone/processing error: {e}\n")


def audio_mode():
    while True:
        print("\n--- AUDIO MODE ---")
        print("1 - Predict from audio file")
        print("2 - Record from microphone")
        print("3 - Back")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            audio_file_mode()
        elif choice == "2":
            audio_microphone_mode()
        elif choice == "3":
            break
        else:
            print("Invalid choice.\n")


def main():
    print("Emotion Detection System 🎯")

    while True:
        print("\nChoose mode:")
        print("1 - Text + Music Recommendation")
        print("2 - Audio Emotion Detection")
        print("3 - Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            text_mode()
        elif choice == "2":
            audio_mode()
        elif choice == "3":
            break
        else:
            print("Invalid choice\n")


if __name__ == "__main__":
    main()