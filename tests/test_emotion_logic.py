"""
Automated test suite for src/emotion_logic.py - the pure ML/business logic
behind EmotiSense's Text, Audio, and Multimodal modes (see README
"Robustness Testing").

Run from the project root with:

    pytest

Some tests need the real production model files under results/
(emotion_model.pkl, vectorizer.pkl, audio_emotion_model.pkl) - those are
marked and skip automatically if the files aren't there yet (e.g. a fresh
clone before running the training scripts), rather than failing the whole
suite.
"""

import numpy as np
import pandas as pd
import pytest

import emotion_logic as el


# --------------------------------------------------------------------------
# fuse_predictions / map_audio_probs_to_text_space
# --------------------------------------------------------------------------

def test_fuse_predictions_sums_to_one():
    text_probs = {"joy": 0.7, "sadness": 0.1, "anger": 0.05, "fear": 0.05, "love": 0.05, "surprise": 0.05}
    audio_probs = {"happy": 0.1, "sad": 0.6, "angry": 0.1, "fear": 0.1, "neutral": 0.1}
    fused = el.fuse_predictions(text_probs, audio_probs)
    assert sum(fused.values()) == pytest.approx(1.0)


def test_fuse_predictions_uses_real_accuracy_weights():
    # Pure text signal (audio all zero except one class) should scale by
    # exactly TEXT_FUSION_WEIGHT / AUDIO_FUSION_WEIGHT, not a 50/50 split.
    text_probs = {"joy": 1.0}
    audio_probs = {"neutral": 1.0}
    fused = el.fuse_predictions(text_probs, audio_probs)
    assert fused["joy"] == pytest.approx(el.TEXT_FUSION_WEIGHT)
    assert fused["neutral"] == pytest.approx(el.AUDIO_FUSION_WEIGHT)


def test_fuse_predictions_keeps_neutral_separate_from_joy():
    # A confidently neutral audio signal must not be silently folded into
    # joy during fusion - only recommend_music does that, and only at the
    # very end when picking a song.
    text_probs = {"anger": 0.17, "fear": 0.17, "joy": 0.17, "love": 0.16, "sadness": 0.16, "surprise": 0.17}
    audio_probs = {"angry": 0.0, "fear": 0.0, "happy": 0.0, "neutral": 1.0, "sad": 0.0}
    fused = el.fuse_predictions(text_probs, audio_probs)
    assert fused["neutral"] == pytest.approx(el.AUDIO_FUSION_WEIGHT)
    assert max(fused, key=fused.get) == "neutral"


def test_map_audio_probs_to_text_space_merges_correctly():
    mapped = el.map_audio_probs_to_text_space(
        {"happy": 0.3, "sad": 0.2, "angry": 0.2, "fear": 0.1, "neutral": 0.2}
    )
    assert mapped == {"joy": 0.3, "sadness": 0.2, "anger": 0.2, "fear": 0.1, "neutral": 0.2}


# --------------------------------------------------------------------------
# recommend_music
# --------------------------------------------------------------------------

@pytest.fixture
def tiny_music_df():
    return pd.DataFrame(
        [
            {"emotion": "joy", "song": "Joy Song A", "artist": "Artist A"},
            {"emotion": "joy", "song": "Joy Song B", "artist": "Artist B"},
            {"emotion": "sadness", "song": "Sad Song A", "artist": "Artist C"},
            {"emotion": "anger", "song": "Anger Song A", "artist": "Artist D"},
            {"emotion": "neutral", "song": "Neutral Song A", "artist": "Artist E"},
        ]
    )


def test_recommend_music_only_picks_from_nonzero_probability_emotions(tiny_music_df):
    rng = np.random.default_rng(0)
    for _ in range(50):
        song = el.recommend_music({"joy": 1.0}, tiny_music_df, rng=rng)
        assert song["emotion"] == "joy"


def test_recommend_music_confidence_weighting_reflects_probabilities(tiny_music_df):
    probs = {"joy": 0.9, "sadness": 0.1}
    rng = np.random.default_rng(42)
    picks = [el.recommend_music(probs, tiny_music_df, rng=rng)["emotion"] for _ in range(500)]
    joy_share = picks.count("joy") / len(picks)
    # Should track ~0.9 with some sampling noise, not hard-commit to
    # exactly the top label every time.
    assert 0.80 <= joy_share <= 0.98
    assert "sadness" in picks


def test_recommend_music_neutral_picks_a_neutral_song(tiny_music_df):
    # data/music.csv has a real "neutral" category (calm/ambient picks) -
    # no fallback to joy needed, unlike the old behaviour.
    rng = np.random.default_rng(1)
    song = el.recommend_music({"neutral": 1.0}, tiny_music_df, rng=rng)
    assert song["emotion"] == "neutral"


def test_recommend_music_shift_mode_never_recommends_negative_bucket(tiny_music_df):
    probs = {"sadness": 0.8, "anger": 0.2}
    rng = np.random.default_rng(2)
    for _ in range(100):
        song = el.recommend_music(probs, tiny_music_df, mode="shift", rng=rng)
        assert song["emotion"] == "joy"


def test_recommend_music_shift_mode_leaves_positive_emotions_alone(tiny_music_df):
    rng = np.random.default_rng(3)
    song = el.recommend_music({"love": 1.0}, tiny_music_df, mode="shift", rng=rng)
    # tiny_music_df has no "love" rows, so this should return None rather
    # than silently substituting a different emotion's song.
    assert song is None


def test_recommend_music_exclude_song_avoids_immediate_repeat(tiny_music_df):
    pure_joy = {"joy": 1.0}
    rng = np.random.default_rng(4)
    first = el.recommend_music(pure_joy, tiny_music_df, rng=rng)
    exclude = (first["song"], first["artist"])
    for i in range(30):
        rng = np.random.default_rng(100 + i)
        nxt = el.recommend_music(pure_joy, tiny_music_df, exclude_song=exclude, rng=rng)
        assert (nxt["song"], nxt["artist"]) != exclude


def test_recommend_music_exclude_song_falls_back_when_it_is_the_only_candidate(tiny_music_df):
    only_anger = tiny_music_df[tiny_music_df["emotion"] == "anger"]
    exclude = (only_anger.iloc[0]["song"], only_anger.iloc[0]["artist"])
    song = el.recommend_music({"anger": 1.0}, only_anger, exclude_song=exclude, rng=np.random.default_rng(5))
    assert song is not None  # falls back rather than returning nothing
    assert (song["song"], song["artist"]) == exclude


def test_recommend_music_returns_none_when_nothing_matches(tiny_music_df):
    song = el.recommend_music({"surprise": 1.0}, tiny_music_df, rng=np.random.default_rng(6))
    assert song is None


# --------------------------------------------------------------------------
# generate_conversational_reply (Chat mode - see README "Core Features")
# --------------------------------------------------------------------------

def test_generate_conversational_reply_uses_a_real_template_for_every_known_emotion():
    for label in el.CHAT_ACKNOWLEDGEMENTS:
        probs = {label: 1.0}
        reply = el.generate_conversational_reply(label, probs, rng=np.random.default_rng(0))
        assert any(reply.startswith(t) for t in el.CHAT_ACKNOWLEDGEMENTS[label])


def test_generate_conversational_reply_falls_back_to_neutral_pool_for_unknown_label():
    # canonical labels the models can actually produce are all covered,
    # but this should degrade gracefully rather than raising if that ever
    # changes (e.g. a future model with a class not in the template pool).
    reply = el.generate_conversational_reply("disgust", {"disgust": 1.0}, rng=np.random.default_rng(0))
    assert any(reply.startswith(t) for t in el.CHAT_ACKNOWLEDGEMENTS["neutral"])


def test_generate_conversational_reply_avoids_immediate_repeat():
    pure_joy = {"joy": 1.0}
    rng = np.random.default_rng(1)
    first = el.generate_conversational_reply("joy", pure_joy, rng=rng)
    for i in range(30):
        rng = np.random.default_rng(100 + i)
        nxt = el.generate_conversational_reply("joy", pure_joy, exclude_text=first, rng=rng)
        assert nxt != first


def test_generate_conversational_reply_matches_mood_by_default():
    reply = el.generate_conversational_reply("sadness", {"sadness": 1.0}, rng=np.random.default_rng(2))
    assert el.CHAT_TRANSITION_MATCH in reply
    assert el.CHAT_TRANSITION_SHIFT not in reply


def test_generate_conversational_reply_shift_mode_changes_transition_for_negative_emotions():
    reply = el.generate_conversational_reply("anger", {"anger": 1.0}, mood_mode="shift", rng=np.random.default_rng(3))
    assert el.CHAT_TRANSITION_SHIFT in reply


def test_generate_conversational_reply_shift_mode_leaves_positive_emotions_alone():
    # joy isn't in MOOD_SHIFT_MAP, so shift mode shouldn't change its transition.
    reply = el.generate_conversational_reply("joy", {"joy": 1.0}, mood_mode="shift", rng=np.random.default_rng(4))
    assert el.CHAT_TRANSITION_MATCH in reply
    assert el.CHAT_TRANSITION_SHIFT not in reply


def test_generate_conversational_reply_hedges_on_low_confidence():
    torn_probs = {"joy": 0.34, "sadness": 0.33, "anger": 0.33}
    reply = el.generate_conversational_reply("joy", torn_probs, rng=np.random.default_rng(5))
    assert any(reply.startswith(p) for p in el.CHAT_LOW_CONFIDENCE_PREFIXES)


def test_generate_conversational_reply_does_not_hedge_on_confident_prediction():
    confident_probs = {"joy": 0.95, "sadness": 0.05}
    reply = el.generate_conversational_reply("joy", confident_probs, rng=np.random.default_rng(6))
    assert not any(reply.startswith(p) for p in el.CHAT_LOW_CONFIDENCE_PREFIXES)


# --------------------------------------------------------------------------
# detect_negated_sentiment / confidence_warning's negation-aware check
# (see README "Adversarial Input Exploration" / ROADMAP.md item 1 - TF-IDF
# has no way to represent "not happy", so a negation word immediately
# before a scored sentiment word is exactly the case worth flagging
# regardless of how confident the model claims to be)
# --------------------------------------------------------------------------

def test_detect_negated_sentiment_finds_negation_immediately_before_scored_word():
    result = el.detect_negated_sentiment("I am not happy at all")
    assert result is not None
    neg_word, sent_word, score = result
    assert neg_word == "not"
    assert sent_word == "happy"
    assert score > 0  # "happy" is a positive-scored word in AFINN


def test_detect_negated_sentiment_handles_contractions():
    result = el.detect_negated_sentiment("honestly this isn't great")
    assert result is not None
    neg_word, sent_word, score = result
    assert neg_word.endswith("n't")
    assert sent_word == "great"


def test_detect_negated_sentiment_finds_negative_scored_word_too():
    # "not sad" negates a NEGATIVE-scored word - still worth flagging,
    # regardless of which direction the polarity flip goes.
    result = el.detect_negated_sentiment("I am not sad")
    assert result is not None
    _, sent_word, score = result
    assert sent_word == "sad"
    assert score < 0


def test_detect_negated_sentiment_returns_none_when_no_negation_present():
    assert el.detect_negated_sentiment("I am so happy today") is None


def test_detect_negated_sentiment_handles_contractions_without_apostrophe():
    # Casual typing often drops the apostrophe ("dont" not "don't") - the
    # negation cue should still be recognised.
    result = el.detect_negated_sentiment("i dont feel too bad")
    assert result is not None
    neg_word, sent_word, score = result
    assert neg_word == "dont"
    assert sent_word == "bad"
    assert score < 0


def test_detect_negated_sentiment_returns_none_when_negation_has_no_nearby_scored_word():
    # "not" is present, but nothing sentiment-bearing follows within the
    # word window - shouldn't false-positive on every negation in general.
    assert el.detect_negated_sentiment("I did not go to the shop yesterday because of the weather") is None


def test_confidence_warning_flags_negated_sentiment_even_at_high_confidence():
    # The exact real bug report this was built for: a confident-looking
    # score on text containing "not happy" should still be flagged.
    warning = el.confidence_warning({"joy": 0.95, "sadness": 0.05}, text="I am not happy at all")
    assert warning is not None
    assert "not" in warning and "happy" in warning


def test_confidence_warning_does_not_flag_confident_prediction_without_negation():
    warning = el.confidence_warning({"joy": 0.95, "sadness": 0.05}, text="I am so happy today")
    assert warning is None


def test_generate_conversational_reply_hedges_on_negated_sentiment_via_text():
    reply = el.generate_conversational_reply(
        "joy", {"joy": 0.95, "sadness": 0.05}, text="I am not happy at all", rng=np.random.default_rng(7)
    )
    assert any(reply.startswith(p) for p in el.CHAT_LOW_CONFIDENCE_PREFIXES)


# --------------------------------------------------------------------------
# apply_negation_adjustment
# --------------------------------------------------------------------------

def test_apply_negation_adjustment_no_op_without_negation():
    probs = {"joy": 0.7, "sadness": 0.3}
    adjusted, note = el.apply_negation_adjustment(probs, "I am so happy today")
    assert adjusted == probs
    assert note is None


def test_apply_negation_adjustment_no_op_without_text():
    probs = {"joy": 0.7, "sadness": 0.3}
    adjusted, note = el.apply_negation_adjustment(probs, None)
    assert adjusted == probs
    assert note is None


def test_apply_negation_adjustment_shifts_positive_bucket_into_negative_bucket():
    # "not happy" negates a positive word - joy's mass should move into
    # sadness/anger/fear, proportional to their existing shares.
    probs = {"joy": 0.6, "sadness": 0.3, "anger": 0.1}
    adjusted, note = el.apply_negation_adjustment(probs, "I am not happy at all")
    assert note is not None
    assert adjusted["joy"] == 0.0
    assert adjusted["sadness"] == pytest.approx(0.3 + 0.6 * (0.3 / 0.4))
    assert adjusted["anger"] == pytest.approx(0.1 + 0.6 * (0.1 / 0.4))
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert max(adjusted, key=adjusted.get) == "sadness"


def test_apply_negation_adjustment_splits_evenly_when_negative_bucket_is_empty():
    probs = {"joy": 1.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0}
    adjusted, note = el.apply_negation_adjustment(probs, "I am not happy at all")
    assert note is not None
    assert adjusted["joy"] == 0.0
    assert adjusted["sadness"] == pytest.approx(1 / 3)
    assert adjusted["anger"] == pytest.approx(1 / 3)
    assert adjusted["fear"] == pytest.approx(1 / 3)


def test_apply_negation_adjustment_no_op_when_no_positive_mass_to_move():
    probs = {"sadness": 0.8, "anger": 0.2}
    adjusted, note = el.apply_negation_adjustment(probs, "I am not happy at all")
    assert adjusted == probs
    assert note is None


def test_apply_negation_adjustment_collapses_to_neutral_for_negated_negative_word():
    # "not sad" negates a NEGATIVE word - doesn't reliably imply joy, so
    # this should read as neutral rather than guessing a positive emotion.
    probs = {"joy": 0.1, "sadness": 0.8, "anger": 0.1}
    adjusted, note = el.apply_negation_adjustment(probs, "I am not sad")
    assert adjusted == {"neutral": 1.0}
    assert note is not None
    assert "neutral" in note.lower()


# --------------------------------------------------------------------------
# normalize_history_columns
# --------------------------------------------------------------------------

def test_normalize_history_columns_backfills_missing_column():
    old_history = pd.DataFrame(
        {
            "timestamp": ["t1"],
            "mode": ["Text"],
            "input_summary": ["hi"],
            "predicted_emotion": ["joy"],
            "confidence": [90.0],
            "recommended_song": ["Happy"],
            "recommended_artist": ["Pharrell Williams"],
            # modality_agreement intentionally missing, as an old
            # pre-Phase-5 history file would be.
        }
    )
    normalized = el.normalize_history_columns(old_history)
    assert list(normalized.columns) == el.HISTORY_COLUMNS
    assert normalized["modality_agreement"].iloc[0] == ""


def test_normalize_history_columns_is_a_no_op_when_already_complete():
    complete = pd.DataFrame([{col: "x" for col in el.HISTORY_COLUMNS}])
    normalized = el.normalize_history_columns(complete)
    assert list(normalized.columns) == el.HISTORY_COLUMNS
    assert len(normalized) == 1


# --------------------------------------------------------------------------
# validate_text_input (Phase 6 input validation)
# --------------------------------------------------------------------------

def test_validate_text_input_leaves_short_text_untouched():
    cleaned, note = el.validate_text_input("  I had a great day  ")
    assert cleaned == "I had a great day"
    assert note is None


def test_validate_text_input_truncates_long_text_with_a_note():
    long_text = "x" * (el.MAX_TEXT_LENGTH + 500)
    cleaned, note = el.validate_text_input(long_text)
    assert len(cleaned) == el.MAX_TEXT_LENGTH
    assert note is not None


# --------------------------------------------------------------------------
# confidence_warning (Phase 6 low-confidence / OOD flag)
# --------------------------------------------------------------------------

def test_confidence_warning_none_for_a_confident_unambiguous_call():
    assert el.confidence_warning({"joy": 0.9, "sadness": 0.1}) is None


def test_confidence_warning_flags_low_top_confidence():
    warning = el.confidence_warning({"joy": 0.3, "sadness": 0.28, "anger": 0.24, "fear": 0.18})
    assert warning is not None
    assert "Low confidence" in warning


def test_confidence_warning_flags_a_close_call_even_with_moderate_top_score():
    # Top score alone (45%) is above the low-confidence threshold, but the
    # top two classes are nearly tied - this should still warn.
    warning = el.confidence_warning({"joy": 0.45, "sadness": 0.44, "anger": 0.11})
    assert warning is not None
    assert "Close call" in warning


def test_confidence_warning_empty_probs_does_not_crash():
    assert el.confidence_warning({}) is None


# --------------------------------------------------------------------------
# model_classes / predict_with_confidence (using real sklearn objects, not
# mocks - a bare estimator and a Pipeline, mirroring the shapes
# load_text_pipeline()/load_audio_model() actually return)
# --------------------------------------------------------------------------

def test_predict_with_confidence_probabilities_sum_to_one():
    from sklearn.linear_model import LogisticRegression

    X = np.array([[0, 0], [0, 1], [5, 5], [5, 4]])
    y = np.array(["a", "a", "b", "b"])
    model = LogisticRegression().fit(X, y)

    label, probs = el.predict_with_confidence(model, np.array([[4.8, 4.9]]))
    assert label == "b"
    assert sum(probs.values()) == pytest.approx(1.0)
    assert set(probs.keys()) == {"a", "b"}


def test_model_classes_finds_classifier_inside_a_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X = np.array([[0, 0], [0, 1], [5, 5], [5, 4]])
    y = np.array(["a", "a", "b", "b"])
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True))]).fit(X, y)

    classes = el.model_classes(pipeline)
    assert set(classes) == {"a", "b"}


def test_model_classes_raises_a_clear_error_for_an_unfitted_object():
    with pytest.raises(AttributeError):
        el.model_classes(object())


# --------------------------------------------------------------------------
# extract_audio_features (needs real librosa - skipped automatically if
# it isn't installed, e.g. in a minimal CI environment)
# --------------------------------------------------------------------------

librosa = pytest.importorskip("librosa")


def _write_wav(path, samples, sample_rate=22050):
    from scipy.io.wavfile import write as write_wav

    samples_int16 = np.int16(np.clip(samples, -1.0, 1.0) * 32767)
    write_wav(str(path), sample_rate, samples_int16)


def test_extract_audio_features_returns_the_expected_54_dim_vector(tmp_path):
    sample_rate = 22050
    duration_seconds = 2
    t = np.linspace(0, duration_seconds, sample_rate * duration_seconds, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)  # a plain 220Hz tone, not silence
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path, tone, sample_rate)

    features = el.extract_audio_features(str(wav_path))
    # 40 MFCC + 1 ZCR + 12 chroma + 1 RMS = 54
    assert features.shape == (54,)
    assert np.all(np.isfinite(features))


def test_extract_audio_features_raises_value_error_on_silence(tmp_path):
    sample_rate = 22050
    silence = np.zeros(sample_rate * 2)
    wav_path = tmp_path / "silence.wav"
    _write_wav(wav_path, silence, sample_rate)

    with pytest.raises(ValueError):
        el.extract_audio_features(str(wav_path))


# --------------------------------------------------------------------------
# Integration smoke tests against the real production models, if present.
# Skip (not fail) if results/ hasn't been populated yet - these confirm
# the actual shipped artefacts behave sanely, on top of the pure-logic
# unit tests above.
# --------------------------------------------------------------------------

def _load_real_text_pipeline():
    import joblib
    from conftest import RESULTS_DIR

    model_path = RESULTS_DIR / "emotion_model.pkl"
    vectorizer_path = RESULTS_DIR / "vectorizer.pkl"
    if not (model_path.exists() and vectorizer_path.exists()):
        pytest.skip("results/emotion_model.pkl or vectorizer.pkl not found - run train_text_model.py first")
    return joblib.load(model_path), joblib.load(vectorizer_path)


def test_real_production_text_model_predicts_sensibly_on_a_clear_example():
    model, vectorizer = _load_real_text_pipeline()
    vec = vectorizer.transform(["I am absolutely furious that my flight got cancelled"])
    label, probs = el.predict_with_confidence(model, vec)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert label == "anger"
