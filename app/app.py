"""
EmotiSense - Multimodal Emotion Detection & Music Recommendation
==================================================================

A Streamlit front end for the emotion-project ML pipeline (text + audio
emotion classifiers trained in src/train_text_model.py and
src/train_audio_model.py).

Run from the project root with:

    streamlit run app/app.py

The app expects the trained artefacts already produced by the training
scripts to exist under results/ (emotion_model.pkl, vectorizer.pkl,
audio_emotion_model.pkl) and data/music.csv for recommendations.
"""

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Paths (resolved relative to this file, so it works regardless of the
# working directory `streamlit run` is launched from)
# --------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
import spotify_oauth as sp_oauth  # noqa: E402 (must follow sys.path insert)
from chart_utils import (  # noqa: E402 (must follow sys.path insert)
    ACCENT_DARK,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    render_emphasis_bar_chart,
    render_trend_line_chart,
)
from emotion_logic import (  # noqa: E402 (must follow sys.path insert)
    AUDIO_MODEL_ACCURACY,
    AUDIO_TO_TEXT_EMOTION,
    CANONICAL_TO_MUSIC_EMOTION,
    HISTORY_COLUMNS,
    MAX_AUDIO_UPLOAD_BYTES,
    MOOD_SHIFT_MAP,
    TEXT_FUSION_WEIGHT,
    TEXT_MODEL_ACCURACY,
    AUDIO_FUSION_WEIGHT,
    apply_negation_adjustment,
    confidence_warning,
    extract_audio_features,
    fuse_predictions,
    generate_conversational_reply,
    map_audio_probs_to_text_space,
    model_classes,
    normalize_history_columns,
    predict_with_confidence,
    recommend_music,
    validate_text_input,
)

TEXT_MODEL_PATH = RESULTS_DIR / "emotion_model.pkl"
VECTORIZER_PATH = RESULTS_DIR / "vectorizer.pkl"
AUDIO_MODEL_PATH = RESULTS_DIR / "audio_emotion_model.pkl"
MUSIC_PATH = DATA_DIR / "music.csv"

# Phase 5 dashboard - model-comparison outputs already produced by
# src/compare_text_models.py / src/compare_audio_models.py, plus the
# per-production-model evaluation artefacts from train_text_model.py /
# train_audio_model.py. All optional: the dashboard tab shows whichever
# of these exist and explains how to generate whichever don't.
TEXT_COMPARISON_CSV = RESULTS_DIR / "text_model_comparison.csv"
AUDIO_COMPARISON_CSV = RESULTS_DIR / "audio_model_comparison.csv"
TEXT_CONFUSION_MATRIX = RESULTS_DIR / "test_confusion_matrix.png"
AUDIO_CONFUSION_MATRIX = RESULTS_DIR / "audio_confusion_matrix.png"
TEXT_CLASSIFICATION_REPORT = RESULTS_DIR / "test_classification_report.txt"
AUDIO_CLASSIFICATION_REPORT = RESULTS_DIR / "audio_classification_report.txt"
HISTORY_PATH = RESULTS_DIR / "prediction_history.csv"

# --------------------------------------------------------------------------
# Spotify enrichment (optional) - looks up a real, playable Spotify track
# for the song/artist the curated CSV recommends, so the result is a
# clickable/embeddable link rather than just text. The CSV (and its
# emotion mapping) stays the actual recommendation logic; Spotify's only
# job is resolving "song title + artist" to a real track.
#
# Uses the Client Credentials flow (app-only auth - no Spotify user login
# needed, just a free Developer app). Completely optional: with no
# credentials configured, or if any call fails (offline, bad credentials,
# rate limited, Spotify's API down), everything below silently does
# nothing and the app behaves exactly as it did before this feature - the
# text-only recommendation card is never affected.
#
# Credentials are read from Streamlit secrets (.streamlit/secrets.toml,
# see .streamlit/secrets.toml.example) or environment variables, never
# hardcoded here. See the README's "Spotify Integration (optional)"
# section for setup steps.
# --------------------------------------------------------------------------

def _get_secret(key):
    try:
        return st.secrets.get(key, "")
    except Exception:
        # No secrets.toml at all, or st.secrets not available (e.g. an
        # older Streamlit version) - treat as "not configured", not a
        # crash. Spotify enrichment is optional.
        return ""


SPOTIFY_CLIENT_ID = _get_secret("SPOTIFY_CLIENT_ID") or os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = _get_secret("SPOTIFY_CLIENT_SECRET") or os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_ENABLED = bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)

# --------------------------------------------------------------------------
# Spotify user login (Authorization Code + PKCE) - a separate, optional
# layer on top of the Client Credentials setup above. Reuses the same
# Developer app/credentials, but additionally lets a specific person log
# into their own Spotify account so EmotiSense can act on it: cross-
# reference their recently-played tracks, build a playlist from their
# prediction history, control playback, and save tracks to Liked Songs.
# See the "My Spotify" tab and src/spotify_oauth.py.
#
# Spotify's Feb 2025 security update requires the redirect URI to use the
# loopback IP literal "127.0.0.1", not "localhost" - the default below
# matches Streamlit's default local port. Override via SPOTIFY_REDIRECT_URI
# in secrets.toml/env if you run on a different port.
# --------------------------------------------------------------------------

SPOTIFY_REDIRECT_URI = _get_secret("SPOTIFY_REDIRECT_URI") or os.environ.get(
    "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501"
)
SPOTIFY_TOKENS_PATH = PROJECT_ROOT / ".streamlit" / "spotify_user_tokens.json"
# Holds the PKCE verifier + expected state between "Connect Spotify
# account" and the redirect back - can't use st.session_state for this
# (see the redirect-handling block below for why).
SPOTIFY_PENDING_OAUTH_PATH = PROJECT_ROOT / ".streamlit" / "spotify_pending_oauth.json"

# Emoji shown next to each predicted label.
EMOTION_EMOJI = {
    "anger": "\U0001F620", "angry": "\U0001F620",
    "fear": "\U0001F628",
    "joy": "\U0001F60A", "happy": "\U0001F60A",
    "love": "\U0001F970",
    "sadness": "\U0001F622", "sad": "\U0001F622",
    "surprise": "\U0001F632",
    "neutral": "\U0001F610",
}

# Brand palette lives in src/chart_utils.py (ACCENT, MUTED_BAR, SURFACE,
# GRID, INK_* imported above) so every chart in the project - this app and
# the model-comparison scripts - shares one validated, colorblind-safe
# color system instead of each file inventing its own.

BRAND_NAME = "EmotiSense"
BRAND_TAGLINE = "Multimodal emotion detection & mood-aware music recommendation"


# --------------------------------------------------------------------------
# Cached model / data loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_text_pipeline_cached(model_mtime: float, vectorizer_mtime: float):
    model = joblib.load(TEXT_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def load_text_pipeline():
    # st.cache_resource keys on the function's arguments, not on the
    # target files' contents - a retrained model.pkl with the exact same
    # path would otherwise keep serving the old in-memory object after a
    # code-only redeploy (Streamlit Cloud reruns the script but doesn't
    # always restart the process), until someone notices and manually
    # reboots the app. Passing each file's mtime busts the cache whenever
    # the file on disk actually changes.
    return _load_text_pipeline_cached(TEXT_MODEL_PATH.stat().st_mtime, VECTORIZER_PATH.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def _load_audio_model_cached(model_mtime: float):
    return joblib.load(AUDIO_MODEL_PATH)


def load_audio_model():
    return _load_audio_model_cached(AUDIO_MODEL_PATH.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_music_df():
    return pd.read_csv(MUSIC_PATH)


@st.cache_data(show_spinner=False)
def load_comparison_csv(path: Path):
    """Loads a model-comparison CSV produced by src/compare_text_models.py
    or src/compare_audio_models.py for the Dashboard tab. Returns None if
    that script hasn't been run yet - the dashboard treats this as
    optional, not an error."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_text_file(path: Path):
    """Best-effort read of an optional results/ text file (a saved
    classification report) for the Dashboard tab. Returns None if it
    doesn't exist or can't be read."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def artefacts_missing():
    missing = []
    for label, path in [
        ("Text model", TEXT_MODEL_PATH),
        ("Text vectorizer", VECTORIZER_PATH),
        ("Audio model", AUDIO_MODEL_PATH),
        ("Music dataset", MUSIC_PATH),
    ]:
        if not path.exists():
            missing.append(f"{label} ({path.relative_to(PROJECT_ROOT)})")
    return missing


def get_spotify_token():
    """Client Credentials flow - app-only auth, no Spotify user login
    needed. Tokens last ~3600s, so a successful fetch is cached in
    st.session_state (keyed with its own expiry) rather than refetched on
    every rerun. Deliberately NOT using @st.cache_data here - that caches
    return values by TTL regardless of what the value is, so a single
    transient failure (offline, Spotify briefly down, etc.) would cache
    None for the full TTL and silently disable every embed for up to an
    hour even after the underlying problem clears. Only caching successes
    means a failure just gets retried on the next call instead."""
    if not SPOTIFY_ENABLED:
        return None
    cached = st.session_state.get("_spotify_app_token")
    if cached and cached["expires_at"] > time.time():
        return cached["token"]
    try:
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=5,
        )
        response.raise_for_status()
        token = response.json().get("access_token")
    except Exception:
        return None
    if token:
        st.session_state["_spotify_app_token"] = {"token": token, "expires_at": time.time() + 3500}
    return token


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_spotify_search(song: str, artist: str, token: str):
    """The actual cached search call, keyed on the (already-stripped)
    song/artist plus the token in use. Only called once a real token is
    in hand (see find_spotify_track below), so a cached None here always
    means a genuine "not found" from the API - never "no token yet" -
    and is safe to trust for the full TTL. Including the token in the
    cache key means the cache naturally invalidates whenever the token
    itself is refreshed (~hourly), which is harmless - just an occasional
    extra lookup, not a correctness issue."""
    try:
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": f"track:{song} artist:{artist}", "type": "track", "limit": 1},
            timeout=5,
        )
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
        if not items:
            return None
        track = items[0]
        return {
            "id": track["id"],
            "url": track["external_urls"]["spotify"],
            "name": track["name"],
            "artist": ", ".join(a["name"] for a in track.get("artists", [])),
        }
    except Exception:
        return None


def find_spotify_track(song: str, artist: str):
    """Best-effort lookup of a real Spotify track matching a song/artist
    from the curated CSV. Returns None on any failure (including "not
    found") - this is a bonus link/embed on top of a recommendation that
    already works perfectly well as plain text without it."""
    token = get_spotify_token()
    if not token:
        return None
    song, artist = sp_oauth.strip_featured_artists(song), sp_oauth.strip_featured_artists(artist)
    return _cached_spotify_search(song, artist, token)


# --------------------------------------------------------------------------
# extract_audio_features, model_classes, predict_with_confidence,
# map_audio_probs_to_text_space, fuse_predictions, and recommend_music all
# moved to src/emotion_logic.py (Phase 6) so they're unit-testable without
# a running Streamlit session - imported above, not redefined here.
# --------------------------------------------------------------------------


def render_result_card(label, probs, song):
    emoji = EMOTION_EMOJI.get(label, "")
    confidence_pct = probs[label] * 100
    if song is not None:
        song_line = f"\U0001F3B5 Recommended: {song['song']} - {song['artist']}"
    else:
        song_line = "No song recommendation found for this emotion."

    html = f"""
    <div class="es-result-card">
        <p class="es-result-label">{emoji} {label.capitalize()}
            <span style="font-size:1rem; color:{INK_MUTED}; font-weight:400;">
                ({confidence_pct:.1f}% confidence)
            </span>
        </p>
        <p class="es-song">{song_line}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_spotify_embed(song):
    """Best-effort: render a real, playable Spotify embed for the
    recommended song if SPOTIFY_CLIENT_ID/SECRET are configured and a
    matching track is found. Does nothing at all otherwise - the plain
    text recommendation in render_result_card() above already stands on
    its own without this."""
    if song is None or not SPOTIFY_ENABLED:
        return
    track = find_spotify_track(song["song"], song["artist"])
    if track is None:
        return
    st.markdown(
        f'<iframe style="border-radius:12px" '
        f'src="https://open.spotify.com/embed/track/{track["id"]}" '
        'width="100%" height="152" frameBorder="0" allowfullscreen="" '
        'allow="autoplay; clipboard-write; encrypted-media; fullscreen; '
        'picture-in-picture" loading="lazy"></iframe>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Spotify user login helpers - session/token management for the
# Authorization Code + PKCE flow (src/spotify_oauth.py). Separate from
# the Client Credentials helpers above: these act on the specific
# person's own account, not just the app.
# --------------------------------------------------------------------------

def get_user_tokens():
    """Returns a valid (non-expired) token dict for the logged-in
    Spotify user, transparently refreshing it (from the on-disk refresh
    token) if it's expired, or None if nobody's connected / the
    connection is no longer valid. Backed by a small JSON file rather
    than st.cache_data because it must survive both Streamlit reruns and
    full app restarts - re-logging in every restart would defeat the
    point of a "connect once" flow."""
    tokens = st.session_state.get("spotify_user_tokens")
    if tokens is None:
        tokens = sp_oauth.load_tokens(SPOTIFY_TOKENS_PATH)
        if tokens:
            st.session_state["spotify_user_tokens"] = tokens
    if not tokens:
        return None
    if sp_oauth.is_expired(tokens):
        if not SPOTIFY_ENABLED:
            return None
        refreshed = sp_oauth.refresh_tokens(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, tokens.get("refresh_token"))
        if refreshed is None:
            # The refresh token itself was rejected - most likely the
            # user revoked EmotiSense's access from their Spotify
            # account settings. Treat them as logged out rather than
            # retrying forever.
            disconnect_spotify_user()
            return None
        sp_oauth.save_tokens(SPOTIFY_TOKENS_PATH, refreshed)
        st.session_state["spotify_user_tokens"] = refreshed
        tokens = refreshed
    return tokens


def disconnect_spotify_user():
    sp_oauth.clear_tokens(SPOTIFY_TOKENS_PATH)
    sp_oauth.clear_tokens(SPOTIFY_PENDING_OAUTH_PATH)
    for key in ("spotify_user_tokens", "spotify_user_info"):
        st.session_state.pop(key, None)


def show_spotify_user_actions(song, context):
    """Play / Queue / Save-to-Liked-Songs buttons for a single
    recommendation - shown under a result card only once the user has
    connected their own account in the "My Spotify" tab. Looked up fresh
    via the user's own access token (not the separate Client Credentials
    lookup used for the embed above) so these work even for someone
    who's only set up login, not the app token, or vice versa.

    `context` (e.g. "text"/"audio"/"multimodal") keeps Streamlit widget
    keys unique when the same song is recommended from more than one tab
    in the same session."""
    if song is None:
        return
    tokens = get_user_tokens()
    if tokens is None:
        return
    track = sp_oauth.search_track(tokens["access_token"], song["song"], song["artist"])
    if track is None:
        return

    key_suffix = f"{context}_{song['song']}_{song['artist']}"
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Play", key=f"play_{key_suffix}"):
            ok, reason = sp_oauth.play_track(tokens["access_token"], track["uri"])
            if ok:
                st.toast("Playing on your Spotify.")
            else:
                st.warning(reason or "Couldn't start playback.")
    with col2:
        if st.button("➕ Queue", key=f"queue_{key_suffix}"):
            ok, reason = sp_oauth.queue_track(tokens["access_token"], track["uri"])
            if ok:
                st.toast("Added to your Spotify queue.")
            else:
                st.warning(reason or "Couldn't queue track.")
    with col3:
        if st.button("❤️ Save", key=f"save_{key_suffix}"):
            if sp_oauth.save_track(tokens["access_token"], track["id"]):
                st.toast("Saved to your Liked Songs.")
            else:
                st.warning("Couldn't save track - try again in a moment.")


# --------------------------------------------------------------------------
# History persistence
# --------------------------------------------------------------------------

def load_history():
    if HISTORY_PATH.exists():
        try:
            return normalize_history_columns(pd.read_csv(HISTORY_PATH))
        except Exception:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def append_history(mode, input_summary, emotion, confidence, song, modality_agreement=""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "input_summary": input_summary,
        "predicted_emotion": emotion,
        "confidence": round(confidence * 100, 1),
        "recommended_song": song["song"] if song is not None else "",
        "recommended_artist": song["artist"] if song is not None else "",
        "modality_agreement": modality_agreement,
    }
    history = load_history()
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)
    st.session_state["history"] = history


def clear_history():
    st.session_state["history"] = pd.DataFrame(columns=HISTORY_COLUMNS)
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


# --------------------------------------------------------------------------
# Chart: emphasis pattern (shared with the model-comparison scripts via
# src/chart_utils.py) - the predicted class is the accent color, every
# other class fades to neutral gray. Keeps the encoding to a single,
# always-legible hue instead of a rainbow of categorical colors, and it
# stays correct no matter how many / which classes a model has.
# --------------------------------------------------------------------------

def show_confidence_chart(probabilities: dict, predicted_label: str):
    """Render the chart into the app and free the matplotlib figure
    immediately after - otherwise repeated predictions across a long
    Streamlit session leak open figures."""
    # No emoji in the chart labels: matplotlib's default font can't render
    # most color emoji glyphs (they show as missing-glyph boxes). Emoji are
    # used in the HTML result card and tab labels instead, where the
    # browser renders them natively.
    items = [(lbl.capitalize(), prob) for lbl, prob in probabilities.items()]
    highlight = predicted_label.capitalize()
    fig = render_emphasis_bar_chart(items, highlight, xlabel="Confidence (%)")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# --------------------------------------------------------------------------
# Streamlit page setup + light branding
# --------------------------------------------------------------------------

st.set_page_config(
    page_title=BRAND_NAME,
    page_icon="\U0001F3AD",
    layout="centered",
)

st.markdown(
    f"""
    <style>
    .es-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 0.25rem;
    }}
    .es-logo {{
        font-size: 2.2rem;
        line-height: 1;
    }}
    .es-title {{
        font-size: 2rem;
        font-weight: 700;
        color: {INK_PRIMARY};
        margin: 0;
    }}
    .es-tagline {{
        color: {INK_SECONDARY};
        margin-top: -0.3rem;
        font-size: 0.95rem;
    }}
    .es-result-card {{
        border: 1px solid {GRID};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0 1rem 0;
        background-color: {SURFACE};
    }}
    .es-result-label {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {ACCENT_DARK};
        margin: 0;
    }}
    .es-song {{
        font-size: 1rem;
        color: {INK_SECONDARY};
    }}
    .es-disclaimer {{
        font-size: 0.8rem;
        color: {INK_MUTED};
        border-top: 1px solid {GRID};
        padding-top: 0.6rem;
        margin-top: 1.5rem;
    }}
    </style>
    <div class="es-header">
        <div class="es-logo">\U0001F3AD</div>
        <div>
            <p class="es-title">{BRAND_NAME}</p>
            <p class="es-tagline">{BRAND_TAGLINE}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = artefacts_missing()
if missing:
    st.error(
        "Missing trained model files, so predictions can't run yet:\n\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n\nTrain the models first with `python src/train_text_model.py` "
        "and `python src/train_audio_model.py`, then reload this page."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = load_history()

music_df = load_music_df()

# --------------------------------------------------------------------------
# Spotify login redirect handling - Streamlit reruns this whole script on
# every page load, including the one Spotify sends the browser back to
# after the user approves (or denies) access, so this has to run near the
# top of every run rather than only inside the "My Spotify" tab.
#
# The verifier/expected-state can't be read from st.session_state here -
# the redirect to accounts.spotify.com and back is a real, full-page
# cross-origin navigation, which tears down the browser's connection to
# the Streamlit server and starts a brand new session on return. Anything
# stored in st.session_state before the redirect is already gone by the
# time this runs. SPOTIFY_PENDING_OAUTH_PATH (a small local file, same
# pattern as SPOTIFY_TOKENS_PATH) survives that navigation instead.
# --------------------------------------------------------------------------

query_params = st.query_params
if "code" in query_params and SPOTIFY_ENABLED:
    returned_state = query_params.get("state")
    pending = sp_oauth.load_tokens(SPOTIFY_PENDING_OAUTH_PATH)
    expected_state = (pending or {}).get("state")
    verifier = (pending or {}).get("verifier")
    if returned_state and expected_state and returned_state == expected_state and verifier:
        exchanged = sp_oauth.exchange_code_for_tokens(
            SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI,
            query_params.get("code"), verifier,
        )
        if exchanged:
            sp_oauth.save_tokens(SPOTIFY_TOKENS_PATH, exchanged)
            st.session_state["spotify_user_tokens"] = exchanged
        else:
            st.session_state["spotify_oauth_error"] = "token exchange failed"
    else:
        st.session_state["spotify_oauth_error"] = "state mismatch - please try connecting again"
    sp_oauth.clear_tokens(SPOTIFY_PENDING_OAUTH_PATH)
    st.query_params.clear()
    st.rerun()
elif "error" in query_params:
    st.session_state["spotify_oauth_error"] = query_params.get("error")
    st.query_params.clear()

with st.sidebar:
    st.markdown("### Recommendation style")
    mood_mode_choice = st.radio(
        "Song recommendations should:",
        ["Match my mood", "Lift my mood"],
        key="mood_mode_choice",
    )
    st.caption(
        "“Lift my mood” swaps sadness/anger/fear recommendations "
        "for an uplifting pick instead of one that matches the detected "
        "mood - the difference between mood-congruent and mood-regulating "
        "listening. Applies to every mode below."
    )

mood_mode = "shift" if mood_mode_choice == "Lift my mood" else "match"

tab_chat, tab_text, tab_audio, tab_multimodal, tab_dashboard, tab_spotify, tab_history = st.tabs(
    [
        "\U0001F4AC Chat",
        "\U0001F5E8️ Text",
        "\U0001F3A4 Audio",
        "\U0001F9E9 Multimodal",
        "\U0001F4CA Dashboard",
        "\U0001F517 My Spotify",
        "\U0001F4CB History",
    ]
)

# --------------------------------------------------------------------------
# Chat mode - a conversational front end over the same text model/
# recommend_music() the Text tab uses, so instead of a form that "blurts
# out" a label+confidence, the assistant acknowledges what you said in
# plain language and offers a song, all inside a running st.chat_message
# history. Deliberately template-based (see CHAT_ACKNOWLEDGEMENTS in
# emotion_logic.py), not a real LLM call - no new dependency, no API key,
# no network requirement, and every reply is exactly as testable/
# deterministic (given a seeded rng) as the rest of this project's logic.
# --------------------------------------------------------------------------

with tab_chat:
    st.subheader("Chat")
    st.caption(
        "A conversational front end over the same Text model/recommendation "
        "engine as the Text tab - see 'Text emotion detection' there for the "
        "model details. Replies are picked from a curated set of templates, "
        "not generated by an LLM."
    )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I'm here to help you find something to listen to. "
                    "How are you feeling today?"
                ),
                "song": None,
            }
        ]

    for msg_idx, msg in enumerate(st.session_state["chat_messages"]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("label") is not None:
                emoji = EMOTION_EMOJI.get(msg["label"], "")
                st.caption(f"{emoji} Detected: {msg['label'].capitalize()} ({msg['confidence'] * 100:.0f}% confidence)")
            if msg.get("warning"):
                st.info(msg["warning"])
            song = msg.get("song")
            if song is not None:
                song_line = f"\U0001F3B5 {song['song']} - {song['artist']}"
                st.markdown(song_line)
                show_spotify_embed(song)
                show_spotify_user_actions(song, context=f"chat_{msg_idx}")

    chat_input = st.chat_input("Tell me how you're feeling...")
    if chat_input:
        st.session_state["chat_messages"].append({"role": "user", "content": chat_input, "song": None})

        cleaned_text, length_note = validate_text_input(chat_input)
        model, vectorizer = load_text_pipeline()
        vec = vectorizer.transform([cleaned_text])
        label, probs = predict_with_confidence(model, vec)
        probs, negation_note = apply_negation_adjustment(probs, cleaned_text)
        if negation_note:
            label = max(probs, key=probs.get)
        song = recommend_music(
            probs,
            music_df,
            mode=mood_mode,
            exclude_song=st.session_state.get("last_song_chat"),
        )
        if song is not None:
            st.session_state["last_song_chat"] = (song["song"], song["artist"])

        reply = generate_conversational_reply(
            label,
            probs,
            mood_mode=mood_mode,
            exclude_text=st.session_state.get("last_chat_reply"),
            text=None if negation_note else cleaned_text,
        )
        st.session_state["last_chat_reply"] = reply
        if length_note:
            reply = f"{reply}\n\n({length_note})"

        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": reply,
                "label": label,
                "confidence": probs[label],
                "song": song,
                "warning": negation_note or confidence_warning(probs),
            }
        )

        append_history(
            mode="Chat",
            input_summary=cleaned_text[:80],
            emotion=label,
            confidence=probs[label],
            song=song,
        )

        st.rerun()

    if len(st.session_state["chat_messages"]) > 1 and st.button("Clear conversation", key="clear_chat"):
        st.session_state["chat_messages"] = st.session_state["chat_messages"][:1]
        st.rerun()

# --------------------------------------------------------------------------
# Text mode
# --------------------------------------------------------------------------

with tab_text:
    st.subheader("Text emotion detection")
    st.caption("TF-IDF + Linear SVM - trained/compared in src/train_text_model.py and src/compare_text_models.py")

    text_input = st.text_area(
        "How are you feeling? Describe it in a sentence or two.",
        placeholder="e.g. I've had such a good day, everything went right!",
        height=100,
    )

    if st.button("Analyse text", type="primary", key="analyse_text"):
        if not text_input.strip():
            st.warning("Type something first.")
        else:
            cleaned_text, length_note = validate_text_input(text_input)
            if length_note:
                st.caption(length_note)

            model, vectorizer = load_text_pipeline()
            vec = vectorizer.transform([cleaned_text])
            label, probs = predict_with_confidence(model, vec)
            probs, negation_note = apply_negation_adjustment(probs, cleaned_text)
            if negation_note:
                label = max(probs, key=probs.get)
            song = recommend_music(
                probs,
                music_df,
                mode=mood_mode,
                exclude_song=st.session_state.get("last_song_text"),
            )
            if song is not None:
                st.session_state["last_song_text"] = (song["song"], song["artist"])

            render_result_card(label, probs, song)
            show_spotify_embed(song)
            show_spotify_user_actions(song, context="text")

            show_confidence_chart(probs, label)

            warning = negation_note or confidence_warning(probs)
            if warning:
                st.info(warning)

            append_history(
                mode="Text",
                input_summary=cleaned_text[:80],
                emotion=label,
                confidence=probs[label],
                song=song,
            )

# --------------------------------------------------------------------------
# Audio mode
# --------------------------------------------------------------------------

with tab_audio:
    st.subheader("Audio emotion detection")
    st.caption(
        "MFCC + Zero-Crossing-Rate + Chroma + RMS features -> Random "
        "Forest - trained/compared in src/train_audio_model.py and "
        "src/compare_audio_models.py"
    )

    audio_source = st.radio(
        "Choose an input method",
        ["Upload a WAV file", "Record from microphone"],
        horizontal=True,
    )

    audio_path = None
    input_summary = None

    if audio_source == "Upload a WAV file":
        uploaded = st.file_uploader("Upload a .wav recording", type=["wav"])
        if uploaded is not None and uploaded.size > MAX_AUDIO_UPLOAD_BYTES:
            st.error(
                f"That file is {uploaded.size / (1024 * 1024):.1f} MB - larger "
                f"than the {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB limit. "
                "Try a shorter clip (this app only looks at the first 5 "
                "seconds anyway)."
            )
        elif uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(uploaded.getbuffer())
            tmp.close()
            audio_path = tmp.name
            input_summary = uploaded.name
            st.audio(uploaded)

    else:
        duration = st.slider("Recording length (seconds)", 3, 10, 6)
        st.caption("Requires a working microphone on the machine running this app.")
        if st.button("\U0001F534 Record now", key="record_button"):
            try:
                import sounddevice as sd
                from scipy.io.wavfile import write as write_wav

                sample_rate = 22050
                with st.spinner(f"Recording for {duration} seconds..."):
                    recording = sd.rec(
                        int(duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                    )
                    sd.wait()

                recording = recording.flatten()
                if np.max(np.abs(recording)) > 0:
                    recording = recording / np.max(np.abs(recording))
                recording_int16 = np.int16(recording * 32767)

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                write_wav(tmp.name, sample_rate, recording_int16)
                tmp.close()

                st.session_state["last_recording"] = tmp.name
                st.success("Recording captured.")
            except Exception as e:
                st.error(
                    f"Couldn't access a microphone ({e}). If this app is "
                    "running on a hosted server, that's expected - there's "
                    "no physical mic for it to use. Try uploading a WAV "
                    "file instead, or run the app locally to record with "
                    "your own microphone."
                )

        if st.session_state.get("last_recording"):
            audio_path = st.session_state["last_recording"]
            input_summary = f"Microphone recording ({duration}s)"
            st.audio(audio_path)

    if audio_path and st.button("Analyse audio", type="primary", key="analyse_audio"):
        try:
            with st.spinner("Extracting features and predicting..."):
                features = extract_audio_features(audio_path).reshape(1, -1)
                audio_model = load_audio_model()
                label, probs = predict_with_confidence(audio_model, features)
                canonical_probs = map_audio_probs_to_text_space(probs)
                song = recommend_music(
                    canonical_probs,
                    music_df,
                    mode=mood_mode,
                    exclude_song=st.session_state.get("last_song_audio"),
                )
                if song is not None:
                    st.session_state["last_song_audio"] = (song["song"], song["artist"])

            render_result_card(label, probs, song)
            show_spotify_embed(song)
            show_spotify_user_actions(song, context="audio")

            show_confidence_chart(probs, label)

            warning = confidence_warning(probs)
            if warning:
                st.info(warning)

            append_history(
                mode="Audio",
                input_summary=input_summary or "Audio clip",
                emotion=label,
                confidence=probs[label],
                song=song,
            )
        except ValueError as e:
            # extract_audio_features raises ValueError for expected,
            # user-fixable problems (currently: silence after trimming) -
            # worth a friendlier, more specific message than the generic
            # except below.
            st.warning(f"Couldn't analyse this clip: {e} Try a louder or longer recording.")
        except Exception as e:
            st.error(f"Could not process this audio clip: {e}")

# --------------------------------------------------------------------------
# Multimodal mode (Phase 3) - runs the text and audio models independently
# and blends their predictions with fuse_predictions() above, rather than
# training a single joint model. A true joint model would need training
# data where the same sample has both a matching transcript and audio
# labelled with the same emotion; the text and audio datasets here are
# separate corpora with no such overlap, so decision-level fusion is the
# approach that works with what this project actually has.
# --------------------------------------------------------------------------

with tab_multimodal:
    st.subheader("Multimodal emotion detection")
    st.caption(
        "Runs the text and audio models independently, then blends their "
        "predictions into one result - weighted by each model's own test "
        f"accuracy (Text {TEXT_FUSION_WEIGHT:.0%} / Audio {AUDIO_FUSION_WEIGHT:.0%})."
    )

    mm_text_input = st.text_area(
        "What are you feeling? Describe it in a sentence or two.",
        placeholder="e.g. I've had such a good day, everything went right!",
        height=100,
        key="mm_text_input",
    )

    mm_audio_source = st.radio(
        "Choose an audio input method",
        ["Upload a WAV file", "Record from microphone"],
        horizontal=True,
        key="mm_audio_source",
    )

    mm_audio_path = None
    mm_audio_summary = None

    if mm_audio_source == "Upload a WAV file":
        mm_uploaded = st.file_uploader(
            "Upload a .wav recording", type=["wav"], key="mm_file_uploader"
        )
        if mm_uploaded is not None and mm_uploaded.size > MAX_AUDIO_UPLOAD_BYTES:
            st.error(
                f"That file is {mm_uploaded.size / (1024 * 1024):.1f} MB - "
                f"larger than the {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB "
                "limit. Try a shorter clip (this app only looks at the first "
                "5 seconds anyway)."
            )
        elif mm_uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(mm_uploaded.getbuffer())
            tmp.close()
            mm_audio_path = tmp.name
            mm_audio_summary = mm_uploaded.name
            st.audio(mm_uploaded)

    else:
        mm_duration = st.slider("Recording length (seconds)", 3, 10, 6, key="mm_duration")
        st.caption("Requires a working microphone on the machine running this app.")
        if st.button("\U0001F534 Record now", key="mm_record_button"):
            try:
                import sounddevice as sd
                from scipy.io.wavfile import write as write_wav

                sample_rate = 22050
                with st.spinner(f"Recording for {mm_duration} seconds..."):
                    recording = sd.rec(
                        int(mm_duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                    )
                    sd.wait()

                recording = recording.flatten()
                if np.max(np.abs(recording)) > 0:
                    recording = recording / np.max(np.abs(recording))
                recording_int16 = np.int16(recording * 32767)

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                write_wav(tmp.name, sample_rate, recording_int16)
                tmp.close()

                st.session_state["mm_last_recording"] = tmp.name
                st.success("Recording captured.")
            except Exception as e:
                st.error(
                    f"Couldn't access a microphone ({e}). If this app is "
                    "running on a hosted server, that's expected - there's "
                    "no physical mic for it to use. Try uploading a WAV "
                    "file instead, or run the app locally to record with "
                    "your own microphone."
                )

        if st.session_state.get("mm_last_recording"):
            mm_audio_path = st.session_state["mm_last_recording"]
            mm_audio_summary = f"Microphone recording ({mm_duration}s)"
            st.audio(mm_audio_path)

    mm_ready = bool(mm_text_input.strip()) and mm_audio_path
    if not mm_ready:
        st.info("Provide both a text description and an audio clip to run the combined analysis.")

    if mm_ready and st.button("Analyse combined", type="primary", key="analyse_multimodal"):
        try:
            mm_cleaned_text, mm_length_note = validate_text_input(mm_text_input)
            if mm_length_note:
                st.caption(mm_length_note)

            with st.spinner("Running both models and blending the results..."):
                text_model, vectorizer = load_text_pipeline()
                text_vec = vectorizer.transform([mm_cleaned_text])
                text_label, text_probs = predict_with_confidence(text_model, text_vec)

                features = extract_audio_features(mm_audio_path).reshape(1, -1)
                audio_model = load_audio_model()
                audio_label, audio_probs = predict_with_confidence(audio_model, features)

                fused_probs = fuse_predictions(text_probs, audio_probs)
                fused_probs, negation_note = apply_negation_adjustment(fused_probs, mm_cleaned_text)
                fused_label = max(fused_probs, key=fused_probs.get)
                song = recommend_music(
                    fused_probs,
                    music_df,
                    mode=mood_mode,
                    exclude_song=st.session_state.get("last_song_multimodal"),
                )
                if song is not None:
                    st.session_state["last_song_multimodal"] = (song["song"], song["artist"])

            render_result_card(fused_label, fused_probs, song)
            show_spotify_embed(song)
            show_spotify_user_actions(song, context="multimodal")
            show_confidence_chart(fused_probs, fused_label)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Text model alone",
                    text_label.capitalize(),
                    f"{text_probs[text_label] * 100:.1f}% confidence",
                )
            with col2:
                st.metric(
                    "Audio model alone",
                    audio_label.capitalize(),
                    f"{audio_probs[audio_label] * 100:.1f}% confidence",
                )

            # Compare in canonical (text) label space, not raw labels - e.g.
            # text "joy" and audio "happy" agree even though the strings
            # differ, since AUDIO_TO_TEXT_EMOTION maps "happy" -> "joy".
            mm_audio_canonical = AUDIO_TO_TEXT_EMOTION.get(audio_label, audio_label)
            modality_agreement = "agree" if text_label == mm_audio_canonical else "disagree"

            if modality_agreement == "agree":
                st.success(
                    f"Text and audio agree - both independently point to "
                    f"**{text_label.capitalize()}**."
                )
            else:
                st.warning(
                    "Text and audio disagree - text alone said "
                    f"**{text_label.capitalize()}** ({text_probs[text_label] * 100:.1f}%), "
                    f"audio alone said **{audio_label.capitalize()}** "
                    f"({audio_probs[audio_label] * 100:.1f}%). The blended result "
                    "above leans toward text since it's the more accurate model "
                    "overall, but still factors in the audio signal."
                )

            warning = negation_note or confidence_warning(fused_probs)
            if warning:
                st.info(warning)

            append_history(
                mode="Multimodal",
                input_summary=f"{mm_cleaned_text[:50]} + {mm_audio_summary or 'audio clip'}",
                emotion=fused_label,
                confidence=fused_probs[fused_label],
                song=song,
                modality_agreement=modality_agreement,
            )
        except ValueError as e:
            # extract_audio_features raises ValueError for expected,
            # user-fixable problems (currently: silence after trimming).
            st.warning(f"Couldn't analyse this clip: {e} Try a louder or longer recording.")
        except Exception as e:
            st.error(f"Could not process this combined input: {e}")

# --------------------------------------------------------------------------
# Dashboard (Phase 5) - surfaces the project's real ML evaluation results
# and this app's own usage inside EmotiSense itself, rather than leaving
# them in results/ files and the README that most users of the app will
# never open.
# --------------------------------------------------------------------------

with tab_dashboard:
    st.subheader("Model performance")
    st.caption(
        "Pulled live from results/ - generated by src/compare_text_models.py "
        "and src/compare_audio_models.py (comparison tables/charts) and "
        "src/train_text_model.py / src/train_audio_model.py (confusion "
        "matrices, classification reports) for the models actually running "
        "in this app."
    )

    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.metric(
            "Text model (production)",
            "Linear SVM",
            f"{TEXT_MODEL_ACCURACY * 100:.1f}% accuracy",
        )
    with perf_col2:
        st.metric(
            "Audio model (production)",
            "Random Forest",
            f"{AUDIO_MODEL_ACCURACY * 100:.1f}% accuracy",
        )
    st.caption(
        "These are the real test-set accuracies used to weight Multimodal "
        f"fusion (Text {TEXT_FUSION_WEIGHT:.0%} / Audio {AUDIO_FUSION_WEIGHT:.0%} "
        "- see the Multimodal tab)."
    )

    text_comparison = load_comparison_csv(TEXT_COMPARISON_CSV)
    audio_comparison = load_comparison_csv(AUDIO_COMPARISON_CSV)

    st.markdown("#### Text model comparison")
    if text_comparison is None:
        st.info("Run `python src/compare_text_models.py` to generate this comparison.")
    else:
        st.dataframe(text_comparison, use_container_width=True, hide_index=True)
        best_text_model = text_comparison.sort_values("accuracy", ascending=False).iloc[0]["model"]
        fig = render_emphasis_bar_chart(
            list(zip(text_comparison["model"], text_comparison["accuracy"])),
            best_text_model,
            xlabel="Test accuracy (%)",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Audio model comparison")
    if audio_comparison is None:
        st.info("Run `python src/compare_audio_models.py` to generate this comparison.")
    else:
        st.dataframe(audio_comparison, use_container_width=True, hide_index=True)
        best_audio_model = audio_comparison.sort_values("accuracy", ascending=False).iloc[0]["model"]
        fig = render_emphasis_bar_chart(
            list(zip(audio_comparison["model"], audio_comparison["accuracy"])),
            best_audio_model,
            xlabel="Test accuracy (%)",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Confusion matrices & classification reports")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.caption("Text model")
        if TEXT_CONFUSION_MATRIX.exists():
            st.image(str(TEXT_CONFUSION_MATRIX), use_container_width=True)
        else:
            st.info("Not generated yet - run `python src/train_text_model.py`.")
        text_report = read_text_file(TEXT_CLASSIFICATION_REPORT)
        if text_report:
            with st.expander("Classification report"):
                st.code(text_report)
    with cm_col2:
        st.caption("Audio model")
        if AUDIO_CONFUSION_MATRIX.exists():
            st.image(str(AUDIO_CONFUSION_MATRIX), use_container_width=True)
        else:
            st.info("Not generated yet - run `python src/train_audio_model.py`.")
        audio_report = read_text_file(AUDIO_CLASSIFICATION_REPORT)
        if audio_report:
            with st.expander("Classification report"):
                st.code(audio_report)

    st.divider()
    st.subheader("Usage analytics")
    st.caption("Based on your own prediction_history.csv - every analysis you've run through this app.")

    history_df = st.session_state["history"]
    if history_df.empty:
        st.info("No predictions yet - try Text, Audio, or Multimodal mode first.")
    else:
        total_predictions = len(history_df)
        avg_confidence = history_df["confidence"].mean()
        top_emotion = history_df["predicted_emotion"].mode().iloc[0]
        top_mode = history_df["mode"].mode().iloc[0]

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total predictions", total_predictions)
        with stat_col2:
            st.metric("Average confidence", f"{avg_confidence:.1f}%")
        with stat_col3:
            st.metric("Most common emotion", str(top_emotion).capitalize())
        with stat_col4:
            st.metric("Most used mode", top_mode)

        mode_counts = history_df["mode"].value_counts()
        fig = render_emphasis_bar_chart(
            list(mode_counts.items()),
            mode_counts.idxmax(),
            xlabel="Number of predictions",
            value_scale=1.0,
            value_suffix="",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        emotion_counts = history_df["predicted_emotion"].value_counts()
        fig = render_emphasis_bar_chart(
            list(emotion_counts.items()),
            emotion_counts.idxmax(),
            xlabel="Number of predictions",
            value_scale=1.0,
            value_suffix="",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if total_predictions >= 2:
            fig = render_trend_line_chart(history_df["confidence"].tolist(), ylabel="Confidence (%)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.divider()
    st.subheader("Multimodal agreement")
    st.caption(
        "How often the text and audio models have independently landed on "
        "the same emotion vs. disagreed, across every Multimodal-mode "
        "prediction you've run (see the agree/disagree callout on the "
        "Multimodal tab)."
    )
    mm_history = history_df[history_df["mode"] == "Multimodal"] if not history_df.empty else history_df
    mm_history = mm_history[mm_history["modality_agreement"].isin(["agree", "disagree"])]
    if mm_history.empty:
        st.info("No Multimodal-mode predictions logged yet.")
    else:
        agreement_counts = mm_history["modality_agreement"].value_counts()
        agree_n = int(agreement_counts.get("agree", 0))
        disagree_n = int(agreement_counts.get("disagree", 0))
        agree_pct = agree_n / (agree_n + disagree_n) * 100
        st.metric("Agreement rate", f"{agree_pct:.0f}%", f"{agree_n} agree / {disagree_n} disagree")
        fig = render_emphasis_bar_chart(
            [("Agree", agree_n), ("Disagree", disagree_n)],
            "Agree",
            xlabel="Number of predictions",
            value_scale=1.0,
            value_suffix="",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# --------------------------------------------------------------------------
# My Spotify - Authorization Code + PKCE login and the 4 account-linked
# features it unlocks (recently-played cross-reference, playlist
# creation from history; Play/Queue/Save buttons live under each
# recommendation in the Text/Audio/Multimodal tabs instead, via
# show_spotify_user_actions() above).
# --------------------------------------------------------------------------

with tab_spotify:
    st.subheader("Connect your Spotify account")
    st.caption(
        "Optional, and separate from the album-art/embed lookups elsewhere "
        "in the app (those use an app-only token and don't need your "
        "login). Logging in here lets EmotiSense act on your own Spotify "
        "account: compare your recent listening to what it's detected, "
        "build a playlist from your prediction history, and play, queue, "
        "or save recommended tracks directly."
    )

    if not SPOTIFY_ENABLED:
        st.info(
            "Set up SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET first - see "
            "the README's 'Spotify Integration' section. Login reuses the "
            "same Developer app as the album-art feature."
        )
    else:
        oauth_error = st.session_state.pop("spotify_oauth_error", None)
        if oauth_error:
            st.error(f"Spotify login didn't complete ({oauth_error}). Try connecting again below.")

        user_tokens = get_user_tokens()

        if user_tokens is None:
            verifier, challenge = sp_oauth.generate_pkce_pair()
            state = sp_oauth.generate_state()
            sp_oauth.save_tokens(SPOTIFY_PENDING_OAUTH_PATH, {"verifier": verifier, "state": state})
            auth_url = sp_oauth.build_authorize_url(SPOTIFY_CLIENT_ID, SPOTIFY_REDIRECT_URI, challenge, state)
            # A plain st.link_button always opens in a new tab (it renders
            # target="_blank"), which leaves the original tab stuck on
            # "not connected" and makes the user go hunt for the new one.
            # A same-tab anchor, styled to match, sends Spotify's redirect
            # back to the tab the user is already looking at.
            st.markdown(
                f'<a href="{auth_url}" target="_self" style="display:inline-block;'
                'padding:0.5rem 1rem;background-color:#FF4B4B;color:white;'
                'border-radius:0.5rem;text-decoration:none;font-weight:600;">'
                '\U0001F517 Connect Spotify account</a>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Opens Spotify's login, then redirects back to {SPOTIFY_REDIRECT_URI}. "
                "That exact URI needs to be registered under your Spotify Developer "
                "app's Settings -> Redirect URIs first (Spotify now requires the "
                "literal `127.0.0.1`, not `localhost` - see the README)."
            )
        else:
            user_info = st.session_state.get("spotify_user_info")
            if user_info is None:
                user_info = sp_oauth.get_current_user(user_tokens["access_token"])
                st.session_state["spotify_user_info"] = user_info

            info_col, disconnect_col = st.columns([3, 1])
            with info_col:
                if user_info:
                    st.success(f"Connected as **{user_info['display_name']}**")
                else:
                    st.success("Connected to Spotify")
            with disconnect_col:
                if st.button("Disconnect"):
                    disconnect_spotify_user()
                    st.rerun()

            st.divider()

            # --- Feature 1: recently-played cross-reference ---
            st.markdown("#### Your recent listening vs. EmotiSense's history")
            st.caption(
                "Spotify no longer exposes audio-features (valence/energy) to "
                "personal apps, so this can't classify the mood of an arbitrary "
                "track you've played. Instead it checks your recently-played "
                "tracks against the 30 curated songs in data/music.csv and shows "
                "which of those you've actually been listening to."
            )
            if st.button("Check recently played", key="check_recent"):
                recent = sp_oauth.get_recently_played(user_tokens["access_token"], limit=50)
                if not recent:
                    st.info("No recently-played tracks returned by Spotify.")
                else:
                    curated = {
                        (str(row["song"]).strip().lower(), str(row["artist"]).strip().lower()): row["emotion"]
                        for _, row in music_df.iterrows()
                    }
                    matches = []
                    for track in recent:
                        key = (track["name"].strip().lower(), track["artist"].strip().lower())
                        if key in curated:
                            matches.append({
                                "Played at": track["played_at"],
                                "Track": f"{track['name']} - {track['artist']}",
                                "Curated emotion tag": curated[key],
                            })
                    if matches:
                        st.write(f"Found {len(matches)} curated track(s) in your last {len(recent)} plays:")
                        st.dataframe(pd.DataFrame(matches), hide_index=True, use_container_width=True)
                    else:
                        st.info(
                            f"None of your last {len(recent)} played tracks are in the "
                            "curated list - expected unless you've specifically played "
                            "one of EmotiSense's recommendations."
                        )
                    with st.expander("Show all recently played tracks"):
                        st.dataframe(
                            pd.DataFrame(recent)[["played_at", "name", "artist"]]
                            .rename(columns={"played_at": "Played at", "name": "Track", "artist": "Artist"}),
                            hide_index=True,
                            use_container_width=True,
                        )

            st.divider()

            # --- Feature 2: auto-playlist creation from history ---
            st.markdown("#### Build a playlist from your prediction history")
            history_for_playlist = st.session_state.get("history", pd.DataFrame(columns=HISTORY_COLUMNS))
            logged_emotions = (
                sorted(e for e in history_for_playlist["predicted_emotion"].dropna().unique())
                if not history_for_playlist.empty else []
            )
            if not logged_emotions:
                st.info("No prediction history yet - analyse some text or audio first, then come back here.")
            else:
                emotion_choice = st.selectbox(
                    "Build a playlist from history logged as:", logged_emotions, key="playlist_emotion"
                )
                playlist_rows = history_for_playlist[history_for_playlist["predicted_emotion"] == emotion_choice]
                candidate_songs = playlist_rows[["recommended_song", "recommended_artist"]].dropna().drop_duplicates()
                candidate_songs = candidate_songs[candidate_songs["recommended_song"] != ""]
                st.caption(f"{len(candidate_songs)} unique recommended track(s) logged under '{emotion_choice}'.")
                if len(candidate_songs) == 0:
                    st.info("No recommended songs logged for this emotion yet.")
                elif st.button(f"Create '{BRAND_NAME} - {emotion_choice.capitalize()} Mix' playlist", key="create_playlist"):
                    with st.spinner("Creating playlist on Spotify..."):
                        user_id = user_info["id"] if user_info else None
                        if not user_id:
                            st.error("Couldn't determine your Spotify user ID - try disconnecting and reconnecting.")
                        else:
                            uris, not_found = [], []
                            for _, row in candidate_songs.iterrows():
                                track = sp_oauth.search_track(
                                    user_tokens["access_token"], row["recommended_song"], row["recommended_artist"]
                                )
                                if track:
                                    uris.append(track["uri"])
                                else:
                                    not_found.append(f"{row['recommended_song']} - {row['recommended_artist']}")
                            if not uris:
                                st.error("Couldn't find any of these tracks on Spotify.")
                            else:
                                playlist_id = sp_oauth.create_playlist(
                                    user_tokens["access_token"],
                                    user_id,
                                    name=f"{BRAND_NAME} - {emotion_choice.capitalize()} Mix",
                                    description=f"Auto-generated from your {BRAND_NAME} prediction history.",
                                )
                                if playlist_id and sp_oauth.add_tracks_to_playlist(
                                    user_tokens["access_token"], playlist_id, uris
                                ):
                                    st.success(
                                        f"Created '{BRAND_NAME} - {emotion_choice.capitalize()} Mix' with "
                                        f"{len(uris)} track(s) in your Spotify account."
                                    )
                                    if not_found:
                                        st.caption(f"Couldn't find on Spotify (skipped): {', '.join(not_found)}")
                                else:
                                    st.error("Playlist creation failed - try again in a moment.")

            st.divider()
            st.caption(
                "Playback control and queueing (needs Spotify Premium) and "
                "'Save to Liked Songs' appear as buttons directly under each "
                "recommendation in the Text/Audio/Multimodal tabs once you're "
                "connected here."
            )

# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------

with tab_history:
    st.subheader("Prediction history")
    history = st.session_state["history"]

    if history.empty:
        st.info("No predictions yet - try Text or Audio mode.")
    else:
        st.dataframe(
            history.sort_index(ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear history"):
            clear_history()
            st.rerun()

st.markdown(
    """
    <div class="es-disclaimer">
        EmotiSense predicts patterns statistically associated with emotion in text and
        speech - it is not a definitive assessment of how someone actually feels.
        Predictions can be affected by background noise, accent, wording, and the
        biases of the training data.
    </div>
    """,
    unsafe_allow_html=True,
)
