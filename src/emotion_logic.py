"""
Pure emotion-detection logic shared by app/app.py (the Streamlit UI) and
tests/test_emotion_logic.py (the automated test suite - see README
"Robustness Testing", Phase 6).

Deliberately has NO Streamlit import and no UI code - everything here is
plain functions/constants over numpy/pandas (and librosa, imported lazily
inside extract_audio_features only), so it can be unit tested directly
with pytest instead of needing a running Streamlit session
(st.session_state, a live script-run context, etc.). app.py imports
everything it needs from here rather than redefining it, so there is one
source of truth for the actual ML/business logic and the UI layer stays
thin.
"""

import functools
import re

import numpy as np

# --------------------------------------------------------------------------
# Real test-set accuracies (see README "Model Comparison") - used to weight
# each modality's contribution when fusing predictions in fuse_predictions().
# --------------------------------------------------------------------------
TEXT_MODEL_ACCURACY = 0.895
AUDIO_MODEL_ACCURACY = 0.528
TEXT_FUSION_WEIGHT = TEXT_MODEL_ACCURACY / (TEXT_MODEL_ACCURACY + AUDIO_MODEL_ACCURACY)
AUDIO_FUSION_WEIGHT = 1 - TEXT_FUSION_WEIGHT

# Maps the audio model's class labels onto the text model's label space for
# fusion/recommendation purposes. "neutral" is left as its own category
# rather than folded into "joy" - the text model has no neutral class of
# its own, so a genuinely neutral audio signal should show up as neutral
# in the blended result rather than silently inflating the joy score.
AUDIO_TO_TEXT_EMOTION = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "neutral": "neutral",
}

# Only used when picking a song, for canonical emotion labels that don't
# map 1:1 onto a data/music.csv "emotion" column value. Empty for now -
# "neutral" used to fall back to "joy" here since the music dataset had no
# neutral mood of its own, but data/music.csv now has a genuine "neutral"
# category (calm/ambient picks), so no remapping is needed.
CANONICAL_TO_MUSIC_EMOTION = {}

# "Lift my mood" recommendation mode maps negative emotions onto a more
# uplifting bucket instead of matching the detected mood directly - the
# distinction music psychology draws between mood-congruent listening (a
# sad song when sad) and mood-regulating listening (an upbeat song
# instead). Love and surprise are left unmapped since they aren't
# negative to begin with.
MOOD_SHIFT_MAP = {
    "sadness": "joy",
    "anger": "joy",
    "fear": "joy",
}

HISTORY_COLUMNS = [
    "timestamp",
    "mode",
    "input_summary",
    "predicted_emotion",
    "confidence",
    "recommended_song",
    "recommended_artist",
    # "agree" / "disagree" for Multimodal predictions, "" for Text/Audio.
    "modality_agreement",
]

# Phase 6: a soft cap on text input length. Not a crash risk (TF-IDF
# vectorisation handles arbitrarily long text fine) but pasting in an
# entire document is clearly not "a sentence or two describing how you
# feel" - truncating with a visible note is friendlier than silently
# vectorising 50,000 characters or letting one pathological input make a
# single prediction noticeably slower than every other one.
MAX_TEXT_LENGTH = 2000

# Phase 6: guard against accidentally uploading something enormous - WAV
# is uncompressed, so even a few minutes of audio is tens of MB; anything
# past this is almost certainly not "a short clip of someone talking".
MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

# --------------------------------------------------------------------------
# Chat mode (templated conversational responses) - deliberately NOT an LLM
# call. A curated pool of empathetic acknowledgements per canonical
# emotion, picked at random (with light repeat-avoidance) rather than
# generated, so the "AI assistant" feel stays honest about being a
# classifier + templates, doesn't need an API key/network dependency, and
# is fully unit-testable/deterministic given a seeded rng - consistent
# with the rest of this project's no-external-LLM, offline-first design.
# Multiple phrasings per emotion exist purely so a chat session doesn't
# feel robotic on repeat use, same motivation as recommend_music()'s
# repeat-avoidance for songs.
# --------------------------------------------------------------------------

CHAT_ACKNOWLEDGEMENTS = {
    "joy": [
        "That's lovely to hear - sounds like things are going well for you.",
        "I'm glad you're feeling good, that kind of energy is worth holding onto.",
        "That's a great mood to be in.",
        "Sounds like you're in a genuinely good place right now.",
    ],
    "sadness": [
        "That sounds heavy - I'm sorry you're carrying that right now.",
        "It's okay to feel low sometimes. Thanks for telling me.",
        "That sounds tough. Take it easy on yourself.",
        "I hear you - that kind of sadness can be a lot to sit with.",
    ],
    "anger": [
        "That sounds really frustrating - I can understand why you'd feel that way.",
        "Sounds like something's genuinely gotten to you. That's fair.",
        "I hear the frustration in that.",
        "That kind of thing would wind anyone up.",
    ],
    "fear": [
        "That sounds unsettling - it's okay to feel uneasy about it.",
        "I can hear the worry in that. That's a lot to sit with.",
        "Sounds like something's got you on edge.",
        "That kind of uncertainty is genuinely hard to shake.",
    ],
    "love": [
        "That's really warm to hear - sounds like something means a lot to you.",
        "That's a nice feeling to have. Hold onto that.",
        "Sounds like real affection there.",
        "I can hear how much that matters to you.",
    ],
    "surprise": [
        "Whoa, that sounds like it caught you off guard.",
        "That's unexpected - how are you sitting with it?",
        "Didn't see that coming, huh?",
        "Sounds like quite the curveball.",
    ],
    "neutral": [
        "Sounds like a pretty even, steady kind of moment.",
        "Nothing dramatic going on right now, just steady. That's alright too.",
        "Sounds calm out there for you right now.",
        "A quiet, even moment - not every day needs to be a big one.",
    ],
}

# Only the sadness/anger/fear bucket has a meaningfully different
# transition line in "lift my mood" mode (matching MOOD_SHIFT_MAP above) -
# joy/love/surprise/neutral aren't remapped by recommend_music() in shift
# mode, so their transition into the recommendation stays the same either
# way.
CHAT_TRANSITION_MATCH = "Here's something that fits that mood:"
CHAT_TRANSITION_SHIFT = "Thought you could use a bit of a lift, so here's something brighter instead:"

CHAT_LOW_CONFIDENCE_PREFIXES = [
    "I'm not fully sure I'm reading this right, but it sounds a little like: ",
    "Take this with a grain of salt, but this reads a bit like: ",
    "Hard to tell for certain from that alone, but it leans toward: ",
]

# --------------------------------------------------------------------------
# Negation-aware confidence checking. TF-IDF is bag-of-words, so it has no
# way to represent "not happy" as anything other than "not" + "happy" as
# two independent, unrelated tokens - it can't know "not" flips the
# sentiment of the word after it. Rather than guessing which emotion the
# input "really" is (risky - a wrong guess is just as misleading as the
# original wrong-but-confident answer, see README "Adversarial Input
# Exploration"), this detects the specific pattern (a negation word
# immediately followed by a word with known sentiment) and surfaces it as
# an explicit, specific warning instead of a blind "any negation word
# present" heuristic, which would fire on plenty of negations that don't
# actually flip anything relevant ("I did not go to the shop").
#
# Word-level sentiment scores come from AFINN-en-165 (Nielsen, 2011;
# Apache-2.0, Technical University of Denmark) - see data/
# afinn_sentiment_lexicon.txt and the README's "Dataset Requirements" for
# attribution. Chosen over the NRC Emotion Lexicon used elsewhere in this
# project (see src/build_music_dataset.py) specifically because NRC's
# license prohibits redistribution - AFINN's Apache-2.0 license allows
# bundling the actual file into this repo as a real runtime dependency.
# --------------------------------------------------------------------------

AFINN_LEXICON_PATH = "data/afinn_sentiment_lexicon.txt"
NEGATION_WORD_WINDOW = 3  # how many words after a negation cue to scan for a scored word


@functools.lru_cache(maxsize=1)
def _load_afinn_lexicon():
    """Loads AFINN-en-165 into a {word: int_score} dict, cached after the
    first call (it's ~3,400 lines - trivial memory-wise, but no reason to
    re-read the file on every single prediction). Returns {} - not a
    crash - if the file isn't present, so confidence_warning()/
    generate_conversational_reply() degrade to their non-negation-aware
    behaviour rather than breaking the app over an optional enhancement."""
    lexicon = {}
    try:
        with open(AFINN_LEXICON_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                word, _, score = line.partition("\t")
                if score:
                    lexicon[word.lower()] = int(score)
    except FileNotFoundError:
        pass
    return lexicon


def _tokenize_words(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


# Apostrophe'd contractions ("isn't", "wasn't", "doesn't", "won't", "can't",
# ...) are already caught by the token.endswith("n't") check below - this
# set is only for the same contractions typed without the apostrophe
# (extremely common in casual text), which don't end in "n't" and so
# wouldn't otherwise match.
NEGATION_CONTRACTIONS_NO_APOSTROPHE = {
    "dont", "cant", "wont", "isnt", "arent", "wasnt", "werent",
    "doesnt", "didnt", "hasnt", "havent", "hadnt",
    "wouldnt", "couldnt", "shouldnt", "aint",
}


def _is_negation_token(token):
    return (
        token in {"not", "never", "no", "cannot", "without"}
        or token in NEGATION_CONTRACTIONS_NO_APOSTROPHE
        or token.endswith("n't")
    )


def detect_negated_sentiment(text):
    """Scans text for a negation cue word immediately followed (within
    NEGATION_WORD_WINDOW words) by a word with a known AFINN sentiment
    score. Returns (negation_word, sentiment_word, score) for the first
    match, or None if no such pattern is found (including when the
    lexicon file isn't available)."""
    lexicon = _load_afinn_lexicon()
    if not lexicon:
        return None
    tokens = _tokenize_words(text)
    for i, tok in enumerate(tokens):
        if not _is_negation_token(tok):
            continue
        for j in range(i + 1, min(i + 1 + NEGATION_WORD_WINDOW, len(tokens))):
            score = lexicon.get(tokens[j])
            if score:
                return tok, tokens[j], score
    return None


def generate_conversational_reply(label, probs, mood_mode="match", exclude_text=None, text=None, rng=None):
    """Returns a single templated, empathetic reply string for the Chat
    tab - an acknowledgement of the detected emotion plus a transition
    into the song recommendation that follows it in the UI.

    exclude_text, if given, is the *previous full reply string* this
    function returned (acknowledgement + transition, exactly what a
    caller has on hand to pass back in - see app.py's Chat tab) so
    consecutive chat turns with the same detected emotion don't repeat
    verbatim. Matched via startswith rather than equality, since
    exclude_text is the full reply but the pool holds acknowledgement
    text only - same exclude_song pattern as recommend_music(), and falls
    back to the full pool the same way if excluding it would leave
    nothing to pick from.

    text is the original user input (separate from exclude_text) - passed
    straight through to confidence_warning() so its negation-aware check
    (see detect_negated_sentiment) can hedge on "not happy"-style inputs
    even when the raw confidence score alone wouldn't have triggered a
    warning. When confidence_warning() would fire (negation-based or
    otherwise), the reply is prefixed with a hedging phrase rather than
    stating the emotion as settled fact - the Chat tab's equivalent of the
    warning banner shown elsewhere.
    """
    rng = rng if rng is not None else np.random.default_rng()

    pool = CHAT_ACKNOWLEDGEMENTS.get(label, CHAT_ACKNOWLEDGEMENTS["neutral"])
    candidates = [t for t in pool if not (exclude_text or "").startswith(t)] or pool
    acknowledgement = candidates[rng.integers(len(candidates))]

    if confidence_warning(probs, text=text) is not None:
        prefix = CHAT_LOW_CONFIDENCE_PREFIXES[rng.integers(len(CHAT_LOW_CONFIDENCE_PREFIXES))]
        acknowledgement = prefix + acknowledgement[0].lower() + acknowledgement[1:]

    transition = CHAT_TRANSITION_SHIFT if (mood_mode == "shift" and label in MOOD_SHIFT_MAP) else CHAT_TRANSITION_MATCH
    return f"{acknowledgement} {transition}"


def extract_audio_features(file_path, target_sr=22050, max_duration=5):
    """Extracts the same 54-dim feature vector (40 MFCC + ZCR + 12 chroma
    + RMS) used everywhere in this project: train_audio_model.py,
    compare_audio_models.py, and app.py's Audio/Multimodal tabs.

    librosa is imported lazily (inside the function, not at module level)
    so that importing emotion_logic.py - and therefore testing every
    other function in this file - doesn't require librosa to be
    installed. It's still a hard requirement to actually call this
    function, exactly as it always has been.

    Raises ValueError if the clip is silent after trimming - this is
    treated as an expected, user-facing validation failure (see
    app.py's friendlier messaging for it) rather than a crash.
    """
    import librosa

    y, sr = librosa.load(file_path, sr=target_sr)

    # librosa.effects.trim computes its dB threshold relative to the
    # signal's own peak amplitude, which is degenerate for pure digital
    # silence (max == 0) - it does not reliably collapse an all-zero
    # signal to length 0. Catch that case explicitly before trimming.
    if not np.any(np.abs(y) > 1e-4):
        raise ValueError("Audio is empty after trimming silence.")

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


def model_classes(model):
    """Return the class labels for either a bare estimator or a Pipeline."""
    if hasattr(model, "classes_"):
        return model.classes_
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return step.classes_
    raise AttributeError("Could not find classes_ on model.")


def predict_with_confidence(model, X):
    """Return (predicted_label, {label: probability})."""
    classes = model_classes(model)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        # Fallback: one-hot the predicted class if the model can't score
        # probabilities at all.
        pred = model.predict(X)[0]
        probs = np.array([1.0 if c == pred else 0.0 for c in classes])
    label = classes[int(np.argmax(probs))]
    return label, dict(zip(classes, probs))


def map_audio_probs_to_text_space(probs: dict) -> dict:
    """Remap the audio model's raw class probabilities (happy/sad/angry/
    fear/neutral) onto the text model's label space (AUDIO_TO_TEXT_EMOTION),
    merging weight into a shared key wherever multiple audio labels would
    map to the same canonical emotion."""
    mapped = {}
    for label, prob in probs.items():
        target = AUDIO_TO_TEXT_EMOTION.get(label, label)
        mapped[target] = mapped.get(target, 0.0) + prob
    return mapped


def fuse_predictions(text_probs: dict, audio_probs: dict) -> dict:
    """Decision-level fusion: blend the text and audio models' probability
    distributions into one, weighted by each model's own test accuracy
    (TEXT_FUSION_WEIGHT / AUDIO_FUSION_WEIGHT). Both distributions already
    sum to 1 and the two weights sum to 1, so the fused result sums to 1
    too - no renormalisation needed."""
    audio_in_text_space = map_audio_probs_to_text_space(audio_probs)
    fused = {}
    for label, prob in text_probs.items():
        fused[label] = fused.get(label, 0.0) + TEXT_FUSION_WEIGHT * prob
    for label, prob in audio_in_text_space.items():
        fused[label] = fused.get(label, 0.0) + AUDIO_FUSION_WEIGHT * prob
    return fused


def recommend_music(probs: dict, music_df, mode="match", exclude_song=None, rng=None):
    """Confidence-weighted song pick.

    Rather than only ever sampling from the single top-predicted emotion's
    songs, every emotion in `probs` contributes candidate songs weighted by
    its own predicted probability - so a meaningfully-likely secondary
    emotion (e.g. 27% joy behind a 55% sadness call) has a real, smaller
    chance of surfacing a song too, instead of the recommendation
    hard-committing to one label the way a plain argmax lookup would.

    mode="match" (default) weights candidates toward the detected mood.
    mode="shift" first remaps negative emotions to a more uplifting bucket
    via MOOD_SHIFT_MAP - "lift my mood" instead of "match my mood".

    exclude_song, if given, is a (song, artist) tuple left out of the
    candidate pool so the same track isn't repeated back-to-back; if that
    was the only candidate, the exclusion is dropped rather than returning
    nothing.
    """
    weights_by_emotion = {}
    for label, prob in probs.items():
        target = CANONICAL_TO_MUSIC_EMOTION.get(label, label)
        if mode == "shift":
            target = MOOD_SHIFT_MAP.get(target, target)
        weights_by_emotion[target] = weights_by_emotion.get(target, 0.0) + prob

    def candidate_rows(skip_excluded):
        rows, weights = [], []
        for _, row in music_df.iterrows():
            weight = weights_by_emotion.get(row["emotion"], 0.0)
            if weight <= 0:
                continue
            if skip_excluded and exclude_song is not None and (row["song"], row["artist"]) == exclude_song:
                continue
            rows.append(row)
            weights.append(weight)
        return rows, weights

    rows, weights = candidate_rows(skip_excluded=True)
    if not rows:
        # The exclusion removed the only match (or nothing matched at
        # all) - fall back to the unfiltered pool rather than recommending
        # nothing just to avoid a repeat.
        rows, weights = candidate_rows(skip_excluded=False)
    if not rows:
        return None

    rng = rng if rng is not None else np.random.default_rng()
    weights = np.array(weights, dtype=float)
    choice_idx = rng.choice(len(rows), p=weights / weights.sum())
    return rows[choice_idx]


def normalize_history_columns(history_df):
    """Backfill any HISTORY_COLUMNS missing from an older history
    DataFrame/CSV (e.g. modality_agreement, added after some rows were
    already logged) so old history files keep loading instead of
    erroring out, and reorder to the canonical column order."""
    for col in HISTORY_COLUMNS:
        if col not in history_df.columns:
            history_df[col] = ""
    return history_df[HISTORY_COLUMNS]


def validate_text_input(text: str):
    """Phase 6 input validation for the Text/Multimodal tabs.

    Returns (cleaned_text, note). cleaned_text is stripped and truncated
    to MAX_TEXT_LENGTH if needed; note is a short user-facing string
    explaining the truncation, or None if nothing needed changing. Does
    NOT reject empty input - callers already check that separately since
    the right response there is "type something", not a truncation note.
    """
    cleaned = text.strip()
    if len(cleaned) > MAX_TEXT_LENGTH:
        cleaned = cleaned[:MAX_TEXT_LENGTH]
        return cleaned, (
            f"Only the first {MAX_TEXT_LENGTH:,} characters were used - "
            "that's already several paragraphs' worth for a single "
            "prediction."
        )
    return cleaned, None


def confidence_warning(probs: dict, text: str = None, low_threshold=0.40, margin_threshold=0.10):
    """Phase 6 low-confidence / out-of-distribution flag.

    Returns a short human-readable warning string if this prediction
    looks genuinely uncertain, or None if it looks like a confident,
    unambiguous call. Three independent signals, any one is enough to
    warn:

    - text (optional - only the Text/Multimodal/Chat callers have raw
      text to pass; Audio doesn't) contains a negation word immediately
      followed by a word with a known sentiment score (see
      detect_negated_sentiment) - regardless of how confident the model
      claims to be, since TF-IDF has no way to represent that negation
      and a high score here is exactly the kind of case that's
      confidently wrong (see README "Adversarial Input Exploration").
    - top confidence below low_threshold: the model isn't sure of
      anything in particular.
    - the gap between the top two classes is below margin_threshold: the
      model is genuinely torn between two candidates, even when the top
      score alone might look moderately confident.
    """
    if text:
        negated = detect_negated_sentiment(text)
        if negated is not None and probs:
            neg_word, sent_word, score = negated
            top_label = max(probs, key=probs.get)
            polarity = "positive" if score > 0 else "negative"
            return (
                f"\"{neg_word} ... {sent_word}\" negates a {polarity}-scored word "
                f"(AFINN score {score:+d}) - the text model is bag-of-words and "
                "can't represent that flip, so this confident-looking read "
                f"({probs[top_label] * 100:.0f}% {top_label.capitalize()}) may "
                "well have the sentiment backwards. Take it with real caution."
            )
    if not probs:
        return None
    sorted_probs = sorted(probs.values(), reverse=True)
    top = sorted_probs[0]
    second = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = top - second

    if top < low_threshold:
        return (
            f"Low confidence ({top * 100:.0f}%) - this input may be "
            "genuinely ambiguous, or unlike anything the model saw during "
            "training. Take this prediction with a grain of salt."
        )
    if margin < margin_threshold:
        top_label = max(probs, key=probs.get)
        runner_up = sorted(probs, key=probs.get, reverse=True)[1]
        return (
            f"Close call - {top_label.capitalize()} and {runner_up.capitalize()} "
            f"are only {margin * 100:.0f} points apart, so this prediction "
            "could easily have gone either way."
        )
    return None


# --------------------------------------------------------------------------
# Negation-aware adjustment. confidence_warning() above only hedges - it
# never changes the label, because guessing the *specific* emotion a
# negation implies is unreliable (see its docstring). But AFINN's score
# *sign* alone is a coarser, more defensible signal: negating a
# positive-scored word ("not happy") reliably points toward the
# negative-emotion side of the model's own distribution, even without
# knowing which negative emotion specifically. So rather than guessing a
# label from scratch, this reallocates probability mass the classifier
# already assigned, shifting the positive-bucket share into the negative
# bucket in the same proportions the model already used between
# sadness/anger/fear.
#
# The reverse direction is NOT applied: negating a negative-scored word
# ("not sad") doesn't reliably mean the opposite specific emotion (joy) -
# it just as often means "fine"/neutral rather than "delighted". Guessing
# joy there would be exactly the kind of overconfident wrong answer this
# whole feature exists to avoid, so that case is instead collapsed to a
# neutral read rather than picking any specific positive emotion.
# --------------------------------------------------------------------------

POSITIVE_BUCKET_EMOTIONS = {"joy", "love"}
NEGATIVE_BUCKET_EMOTIONS = {"sadness", "anger", "fear"}


def apply_negation_adjustment(probs: dict, text: str):
    """Returns (probs, note). If detect_negated_sentiment(text) finds
    nothing (or there's no positive-bucket mass to move), returns the
    original probs unchanged and note=None. Otherwise returns an adjusted
    probs dict (same keys, still summing to 1) and a short human-readable
    note explaining what changed and why - callers should show this note
    to the user in place of (not in addition to) confidence_warning()'s
    negation branch, since this actively corrects the read rather than
    just hedging on it."""
    if not probs:
        return probs, None
    negated = detect_negated_sentiment(text) if text else None
    if negated is None:
        return probs, None

    neg_word, sent_word, score = negated

    if score < 0:
        note = (
            f"\"{neg_word} {sent_word}\" negates a negative-scored word "
            f"(AFINN score {score:+d}) - that doesn't reliably point to any "
            "one specific positive emotion, so this is read as neutral "
            "rather than guessing which one."
        )
        return {"neutral": 1.0}, note

    positive_mass = sum(probs.get(e, 0.0) for e in POSITIVE_BUCKET_EMOTIONS)
    if positive_mass <= 0:
        return probs, None

    original_label = max(probs, key=probs.get)
    adjusted = dict(probs)
    for e in POSITIVE_BUCKET_EMOTIONS:
        if e in adjusted:
            adjusted[e] = 0.0

    negative_keys = [e for e in NEGATIVE_BUCKET_EMOTIONS if e in adjusted] or list(NEGATIVE_BUCKET_EMOTIONS)
    negative_total = sum(probs.get(e, 0.0) for e in negative_keys)
    if negative_total > 0:
        for e in negative_keys:
            adjusted[e] = adjusted.get(e, 0.0) + positive_mass * (probs.get(e, 0.0) / negative_total)
    else:
        share = positive_mass / len(negative_keys)
        for e in negative_keys:
            adjusted[e] = adjusted.get(e, 0.0) + share

    new_label = max(adjusted, key=adjusted.get)
    note = (
        f"\"{neg_word} {sent_word}\" negates a positive-scored word "
        f"(AFINN score {score:+d}) - the positive-emotion share of this "
        "prediction was shifted toward the negative emotions instead, "
        f"changing the read from {original_label.capitalize()} to "
        f"{new_label.capitalize()}."
    )
    return adjusted, note
