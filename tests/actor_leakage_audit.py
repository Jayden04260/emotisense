"""
Fairness/generalisation audit (Phase 6 extension): does the audio
train/test split leak actors between train and test?

train_audio_model.py / compare_audio_models.py both call
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) -
stratified only by emotion label, with no grouping by actor. RAVDESS and
CREMA-D filenames both encode a per-actor speaker ID, and each actor
recorded many takes across all emotions, so a plain random split is very
likely to put the same actor's voice in both train and test - the model
could then partly learn "whose voice is this" as a shortcut instead of
generalisable emotion cues, inflating the reported accuracy.

train_test_split's assignment is fully determined by array length/order,
the stratify (y) array, and random_state - not by the actual feature
values - so this replicates the exact split using just filenames/labels,
no re-extraction of audio features needed. (Verified against the last
real training run's own log: "Loaded 7035 audio samples" for 7035 files
on disk - zero files were skipped, so this file listing is an exact
match for what load_audio_dataset produced, not an approximation.)

Run from the project root with:

    python tests/actor_leakage_audit.py
"""

import os
import re

import numpy as np
from sklearn.model_selection import train_test_split

AUDIO_DATA_PATH = "data/audio"

RAVDESS_PATTERN = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$", re.IGNORECASE)
CREMAD_PATTERN = re.compile(r"^(\d{4})_[A-Z]{3}_[A-Z]{3}_(LO|MD|HI|XX)\.wav$", re.IGNORECASE)


def actor_of(file_name):
    """Returns a globally-unique actor id like 'ravdess-12' or
    'cremad-1001', or None if the filename doesn't match either known
    convention."""
    m = RAVDESS_PATTERN.match(file_name)
    if m:
        return f"ravdess-{m.group(7)}"
    m = CREMAD_PATTERN.match(file_name)
    if m:
        return f"cremad-{m.group(1)}"
    return None


def list_files_and_labels(base_path):
    labels, actors = [], []
    for emotion in sorted(os.listdir(base_path)):
        emotion_folder = os.path.join(base_path, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        for file_name in sorted(os.listdir(emotion_folder)):
            if not file_name.lower().endswith(".wav"):
                continue
            labels.append(emotion)
            actors.append(actor_of(file_name))
    return np.array(labels), np.array(actors)


def main():
    y, actors = list_files_and_labels(AUDIO_DATA_PATH)
    print(f"Total files: {len(y)}")
    unknown = (actors == None).sum()  # noqa: E711 (numpy object array, need == not is)
    if unknown:
        print(f"WARNING: {unknown} files didn't match either filename convention - excluded below")

    known_mask = actors != None  # noqa: E711
    y_known, actors_known = y[known_mask], actors[known_mask]
    indices = np.arange(len(y_known))

    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_known
    )
    actors_train = set(actors_known[idx_train])
    actors_test = set(actors_known[idx_test])
    leaked = actors_train & actors_test

    print(f"\nDistinct actors in train: {len(actors_train)}")
    print(f"Distinct actors in test:  {len(actors_test)}")
    print(f"Actors appearing in BOTH: {len(leaked)}")
    print(f"Leakage rate: {len(leaked) / len(actors_test):.1%} of test actors also appear in train")

    test_rows_from_leaked_actors = sum(1 for a in actors_known[idx_test] if a in leaked)
    print(
        f"Test rows whose actor also appears in train: {test_rows_from_leaked_actors} "
        f"/ {len(idx_test)} ({test_rows_from_leaked_actors / len(idx_test):.1%})"
    )

    print("\nBreakdown by source dataset:")
    for prefix in ("ravdess", "cremad"):
        train_a = {a for a in actors_train if a.startswith(prefix)}
        test_a = {a for a in actors_test if a.startswith(prefix)}
        if not test_a:
            continue
        leaked_a = train_a & test_a
        print(
            f"  {prefix}: {len(test_a)} test actors, {len(leaked_a)} also in train "
            f"({len(leaked_a) / len(test_a):.1%})"
        )


if __name__ == "__main__":
    main()
