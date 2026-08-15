"""
One-off data-prep script that expanded data/music.csv beyond its original
35 hand-curated songs. See README "Music Recommendation" for the full
rationale; summary here.

Real songs/artists come from the ernestchu/lyrics-emotion-classification
dataset on Hugging Face (20.2k songs, genuinely diverse genres including
rap/R&B - confirmed by manual sampling, unlike a typical audio-feature
mood dataset which tends to skew mainstream pop/rock):

    https://huggingface.co/datasets/ernestchu/lyrics-emotion-classification

That dataset's own "class" column is undocumented (no label legend
anywhere, verified by checking its raw README directly) - a real dead
end, not a shortcut skipped. So labels here are derived independently via
the NRC Emotion Lexicon (Mohammad & Turney), a manually-annotated word
list mapping ~14k English words to 8 emotions. Free for non-commercial
research use; citation required wherever it's used - see data/music.csv's
citation note in the README's "Music Dataset" section. Download (select
"Non-Commercial Research Use"):

    https://saifmohammad.com/WebPages/AccessResource.htm

Method: tokenize each song's lyrics, count lexicon hits per emotion (only
the 5 that map onto this project's categories: joy, sadness, anger, fear,
surprise - NRC's trust/disgust/anticipation have no equivalent here), and
keep a song only if it has real evidence (>= MIN_MATCHED_WORDS lexicon
hits) *and* a clear top emotion (>= MIN_MARGIN over the runner-up).
Ranking within a passing pool is by matched-word count first, margin
second - margin alone was tried first and it silently preferred songs
with a single lucky lexicon word (100% "confidence" from 1 data point),
a bug caught by inspecting the actual output before this shipped.

"love" and "neutral" are untouched by this script and keep their
original 5 hand-picked songs each - none of NRC's 8 categories has an
equivalent for either.

Prerequisites (not included in this repo - see download links above):
    1. Download the 3 parquet splits (train/dev/test) of
       ernestchu/lyrics-emotion-classification and point LYRICS_DIR at
       the folder containing them, named lyrics_train.parquet /
       lyrics_dev.parquet / lyrics_test.parquet.
    2. Download and unzip the NRC Suite of Sentiment/Emotion Lexicons and
       point LEXICON_PATH at NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
       inside it.

Run from the project root with:

    python src/build_music_dataset.py

Appends directly to data/music.csv (skipping any song+artist pair
already present, case-insensitive) rather than overwriting it.
"""

import re
import string
from collections import defaultdict

import pandas as pd

# Update these two paths after downloading the prerequisites above.
LEXICON_PATH = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
LYRICS_DIR = "."

MUSIC_CSV_PATH = "data/music.csv"

TARGET_EMOTIONS = {"joy", "sadness", "anger", "fear", "surprise"}
MIN_MATCHED_WORDS = 10  # need real evidence, not 1 lucky word scoring a hollow 100%
MIN_MARGIN = 0.30  # top emotion must beat the runner-up by 30% (relative)
PER_EMOTION_TARGET = 25
MAX_SONGS_PER_ARTIST = 2  # so the expansion isn't dominated by whichever artist scores best


def load_lexicon(path):
    lexicon = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, emotion, flag = line.strip().split("\t")
            if emotion in TARGET_EMOTIONS and flag == "1":
                lexicon[word].add(emotion)
    return lexicon


def tokenize(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text.split()


def score_lyrics(tokens, lexicon):
    """Returns (top_emotion, top_score, margin, matched_word_count), or
    (None, 0.0, 0.0, 0) if no lexicon words were found at all."""
    counts = defaultdict(int)
    matched = 0
    for tok in tokens:
        emotions = lexicon.get(tok)
        if emotions:
            matched += 1
            for e in emotions:
                counts[e] += 1
    if matched == 0:
        return None, 0.0, 0.0, 0
    scores = {e: c / matched for e, c in counts.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_emotion, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = (top_score - second_score) / top_score if top_score else 0.0
    return top_emotion, top_score, margin, matched


def main():
    lexicon = load_lexicon(LEXICON_PATH)
    print(f"Loaded lexicon: {len(lexicon)} words across {TARGET_EMOTIONS}")

    df = pd.concat(
        [
            pd.read_parquet(f"{LYRICS_DIR}/lyrics_train.parquet"),
            pd.read_parquet(f"{LYRICS_DIR}/lyrics_dev.parquet"),
            pd.read_parquet(f"{LYRICS_DIR}/lyrics_test.parquet"),
        ],
        ignore_index=True,
    )
    print(f"Loaded {len(df)} songs total")

    candidates = defaultdict(list)
    for _, row in df.iterrows():
        lyrics = row.get("lyrics")
        if not isinstance(lyrics, str) or not lyrics.strip():
            continue
        tokens = tokenize(lyrics)
        emotion, score, margin, matched = score_lyrics(tokens, lexicon)
        if emotion is None or matched < MIN_MATCHED_WORDS or margin < MIN_MARGIN:
            continue
        candidates[emotion].append(
            {"song": row["track_name"], "artist": row["artist_name"], "score": score, "margin": margin, "matched": matched}
        )

    print("\nCandidate pool sizes (post min-evidence/margin filter):")
    for emotion in sorted(TARGET_EMOTIONS):
        print(f"  {emotion}: {len(candidates[emotion])}")

    selected_rows = []
    for emotion in sorted(TARGET_EMOTIONS):
        pool = sorted(candidates[emotion], key=lambda c: (-c["matched"], -c["margin"]))
        seen_artists = defaultdict(int)
        picked = []
        for c in pool:
            if seen_artists[c["artist"]] >= MAX_SONGS_PER_ARTIST:
                continue
            picked.append(c)
            seen_artists[c["artist"]] += 1
            if len(picked) >= PER_EMOTION_TARGET:
                break
        selected_rows.extend({"emotion": emotion, "song": c["song"], "artist": c["artist"]} for c in picked)
        print(f"\n{emotion.upper()} - {len(picked)} selected, sample of 5:")
        for c in picked[:5]:
            print(f"  {c['song']!r} - {c['artist']} (matched={c['matched']}, score={c['score']:.2f}, margin={c['margin']:.2f})")

    new_df = pd.DataFrame(selected_rows, columns=["emotion", "song", "artist"])
    existing = pd.read_csv(MUSIC_CSV_PATH)
    key = lambda d: d["song"].str.strip().str.lower() + "|" + d["artist"].str.strip().str.lower()
    dupe_mask = key(new_df).isin(set(key(existing)))
    new_df = new_df[~dupe_mask]

    combined = pd.concat([existing, new_df], ignore_index=True)[["emotion", "song", "artist"]]
    combined.to_csv(MUSIC_CSV_PATH, index=False)
    print(f"\n{MUSIC_CSV_PATH}: {len(existing)} -> {len(combined)} rows ({len(new_df)} added, {dupe_mask.sum()} duplicates skipped)")


if __name__ == "__main__":
    main()
