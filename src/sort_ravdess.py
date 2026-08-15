import os
import shutil

# Wherever you extracted RAVDESS's Audio_Speech_Actors_01-24.zip to -
# update this to your own path before running. (Not a real path from
# anyone's machine - fill this in yourself; this script is a one-off you
# run once after downloading RAVDESS, not something the app itself calls.)
SOURCE_DIR = r"C:\path\to\Audio_Speech_Actors_01-24"

# Project output folder - same data/audio/<emotion>/ tree sort_cremad.py
# uses, so train_audio_model.py / compare_audio_models.py pick these up
# automatically alongside the existing CREMA-D files. Relative to wherever
# you run this from - run it from the project root, same as every other
# script under src/.
DEST_DIR = "data/audio"

# RAVDESS emotion mapping
EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
}

# Make destination folders
os.makedirs(DEST_DIR, exist_ok=True)
for emotion_name in EMOTION_MAP.values():
    os.makedirs(os.path.join(DEST_DIR, emotion_name), exist_ok=True)

copied_count = 0
skipped_count = 0

for root, _, files in os.walk(SOURCE_DIR):
    for file_name in files:
        if not file_name.lower().endswith(".wav"):
            continue

        parts = file_name.replace(".wav", "").split("-")

        if len(parts) < 3:
            skipped_count += 1
            continue

        vocal_channel = parts[1]   
        emotion_code = parts[2]

        # Keep only speech files
        if vocal_channel != "01":
            skipped_count += 1
            continue

        # Keep only selected emotions
        if emotion_code not in EMOTION_MAP:
            skipped_count += 1
            continue

        emotion_folder = EMOTION_MAP[emotion_code]
        source_path = os.path.join(root, file_name)
        dest_path = os.path.join(DEST_DIR, emotion_folder, file_name)

        shutil.copy2(source_path, dest_path)
        copied_count += 1

print(f"Done. Copied {copied_count} files.")
print(f"Skipped {skipped_count} files.")
print(f"Sorted files are in: {DEST_DIR}")