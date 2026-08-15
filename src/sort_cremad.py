import os
import shutil

# Wherever you extracted CREMA-D's AudioWAV.zip to - update this to your own
# path before running. (Not a real path from anyone's machine - fill this
# in yourself; this script is a one-off you run once after downloading
# CREMA-D, not something the app itself calls.)
SOURCE_DIR = r"C:\path\to\CREMA-D\AudioWAV"

# Project output folder - same data/audio/<emotion>/ tree sort_ravdess.py
# uses, so train_audio_model.py / compare_audio_models.py pick these up
# automatically alongside the existing RAVDESS files. Relative to wherever
# you run this from - run it from the project root, same as every other
# script under src/.
DEST_DIR = "data/audio"

# CREMA-D filenames: [ActorID]_[Sentence]_[Emotion]_[Level].wav
# Emotion codes: ANG, DIS, FEA, HAP, NEU, SAD - DIS (disgust) is dropped since
# the project's other datasets/classes don't include it.
EMOTION_MAP = {
    "ANG": "angry",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# CREMA-D filenames don't collide with RAVDESS's numeric-dash naming
# (e.g. "03-01-05-01-01-01-01.wav"), so files are copied in as-is with no
# renaming needed to tell the two sources apart later.
os.makedirs(DEST_DIR, exist_ok=True)
for emotion_name in set(EMOTION_MAP.values()):
    os.makedirs(os.path.join(DEST_DIR, emotion_name), exist_ok=True)

copied_count = 0
skipped_count = 0

for file_name in os.listdir(SOURCE_DIR):
    if not file_name.lower().endswith(".wav"):
        continue

    parts = file_name.replace(".wav", "").split("_")

    if len(parts) != 4:
        skipped_count += 1
        continue

    _actor_id, _sentence, emotion_code, _level = parts

    if emotion_code not in EMOTION_MAP:
        skipped_count += 1
        continue

    emotion_folder = EMOTION_MAP[emotion_code]
    source_path = os.path.join(SOURCE_DIR, file_name)
    dest_path = os.path.join(DEST_DIR, emotion_folder, file_name)

    shutil.copy2(source_path, dest_path)
    copied_count += 1

print(f"Done. Copied {copied_count} files.")
print(f"Skipped {skipped_count} files.")
print(f"Sorted files are in: {DEST_DIR}")
