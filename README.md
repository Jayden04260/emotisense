# Emotion Detection System 🎯

## Overview

The Emotion Detection System is a multimodal machine learning project developed in Python that detects human emotions from both **text input** and **audio input**.

The system includes:

 Text Emotion Detection using TF-IDF and Linear SVM
 Audio Emotion Detection using speech feature extraction and Random Forest
 Multimodal Emotion Detection that blends the text and audio models' predictions into one result
 Chat Mode - a conversational front end over the text model, templated
  responses rather than a raw label+confidence display
 Music Recommendation System based on predicted emotional state
 In-app Dashboard surfacing real model evaluation results and usage analytics
 Interactive Command-Line Interface (CLI) for real-time user interaction
 EmotiSense - a Streamlit web app front end for all of the above (see below)


# Core Features

## Chat Mode

 A conversational front end over the same text model/`recommend_music()`
  the Text tab uses - instead of a form that predicts and displays a
  label+confidence, the assistant acknowledges what you said in plain
  language (via `st.chat_message`/`st.chat_input`) and offers a song,
  inside a running conversation history
 Replies are picked from a curated pool of empathetic acknowledgements
  per emotion (`CHAT_ACKNOWLEDGEMENTS` in `src/emotion_logic.py`), with
  light repeat-avoidance so a session doesn't feel robotic - **not an
  LLM call**: no new dependency, no API key, no network requirement, and
  fully deterministic/unit-testable given a seeded rng, consistent with
  the rest of this project's offline-first design
 When `confidence_warning()` would fire, the reply is hedged ("I'm not
  fully sure I'm reading this right, but...") rather than stating the
  detected emotion as settled fact
 Each assistant turn still shows the actual detected emotion + confidence
  as a caption, and the full song card (Spotify embed, Play/Queue/Save if
  you're logged in) - conversational framing on top of the real
  prediction, not instead of it
 Logged to prediction history as mode "Chat", so Dashboard's usage
  analytics and per-mode breakdown include it automatically

## Text Mode

 User enters written text
 TF-IDF vectorisation processes text
 Linear SVM predicts emotion
 Emotion-linked music recommendation is displayed

## Audio File Mode

 User selects or plays audio files
 Audio preprocessing includes:

   Silence trimming
   Volume normalisation
   Fixed-length standardisation
 Features extracted:

   MFCC
   Zero Crossing Rate
   Chroma
   RMS Energy
 Gradient Boosting predicts emotion
 Song recommendation provided

## Microphone Mode

 Live voice recording
 Real-time speech emotion prediction
 Confidence score reporting
 Song recommendation output

## Multimodal Mode

 User provides both a text description AND an audio clip (upload or microphone)
 Text and audio models each predict independently, in their own label space
 Predictions are blended via decision-level fusion, weighted by each
  model's own real test accuracy (Text ~63% weight / Audio ~37% weight,
  from the 89.2% / 52.8% accuracies in "Model Comparison" below - the
  weighting shifts automatically whenever either model's measured
  accuracy changes, since it's computed from `TEXT_MODEL_ACCURACY`/
  `AUDIO_MODEL_ACCURACY` in `src/emotion_logic.py`, not hardcoded)
 A genuinely neutral audio signal stays "neutral" in the blended result
  rather than being folded into another emotion
 Combined result plus each model's standalone prediction are both shown,
  with an explicit callout when the two modalities agree or disagree
 Emotion-linked music recommendation is displayed for the blended result

This is decision-level fusion (combining two independently-trained models'
outputs), not a single jointly-trained multimodal model - a true joint
model would need training data where the same sample has both a matching
transcript and audio labelled with the same emotion, and the text and
audio datasets this project uses are separate corpora with no such
overlap.

## Music Recommendation

 Applies to all three modes (Text, Audio, Multimodal) via one shared
  `recommend_music()` function
 Confidence-weighted picks: every emotion with nonzero predicted
  probability can contribute a song, weighted by that probability - so a
  meaningfully-likely secondary emotion (e.g. 27% joy behind a 55%
  sadness call) has a real, smaller chance of surfacing a song too,
  instead of always hard-committing to the single top label
 Sidebar toggle between "Match my mood" (a song fitting the detected
  emotion) and "Lift my mood" (an uplifting pick instead, for
  sadness/anger/fear) - the distinction music psychology draws between
  mood-congruent and mood-regulating listening
 Won't recommend the same song twice in a row for the same mode
 `data/music.csv` has grown from its original 12 songs to 160: joy,
  sadness, anger, fear, and surprise each have 30 (expanded via
  `src/build_music_dataset.py` - real songs, lyric-based labelling, see
  "Music Dataset"), while love and neutral remain the original 5
  hand-picked songs each, since the labelling method used for the other
  five doesn't have an equivalent category for those two
 Optional Spotify enrichment shows a real, playable embed under the
  recommendation if configured (see "Spotify Integration" below) - the
  CSV above is still what decides *which* song, Spotify just resolves it
  to a real track

## Spotify Login (Optional)

A second, separate Spotify feature from the plain embed above: logging
into your *own* Spotify account (not just the app-only token) from the
"My Spotify" tab unlocks four things -

 Recently-played cross-reference: checks your last 50 played tracks
  against the curated songs in `data/music.csv` (160, see "Music
  Recommendation") and shows which of EmotiSense's own recommendations
  you've actually been listening to.
  (Spotify no longer exposes audio-features/valence-energy data to
  personal apps, so this can't classify the mood of an arbitrary track
  you played - only match against the curated list.)
 Auto-playlist creation: builds a real Spotify playlist in your account
  from every unique song EmotiSense has recommended for a given emotion
  in your prediction history (e.g. a "EmotiSense - Joy Mix").
 Direct playback control: Play / Queue buttons under each
  recommendation start or queue the track on one of your active Spotify
  devices. **Requires Spotify Premium** - Spotify's playback-control
  endpoints reject free accounts (a 403 is shown as a friendly message
  rather than failing silently).
 Save to Liked Songs: a Save button adds the recommended track straight
  to your Liked Songs.

Uses the Authorization Code + PKCE flow (a full user login via Spotify's
own sign-in page, not just the app-only Client Credentials setup) - see
"Spotify Integration" in the Installation Guide for setup, including the
extra Redirect URI step this needs. Like the plain embed, this is
entirely optional: skip it and every other feature works exactly the
same without it.

## Dashboard Mode

 Model performance: the real text/audio model-comparison tables and
  accuracy charts (from `src/compare_text_models.py` /
  `src/compare_audio_models.py`), plus each production model's confusion
  matrix and classification report, if those have been generated -
  surfaces the project's own ML evaluation inside the app itself instead
  of leaving it in the `results/` folder and README
 Usage analytics: total predictions, average confidence, most common
  emotion and most-used mode, a breakdown chart per mode and per emotion,
  and a confidence-over-time trend line, all computed live from your own
  `prediction_history.csv`
 Multimodal agreement rate: how often the text and audio models have
  independently landed on the same emotion vs. disagreed, across every
  Multimodal-mode prediction you've run (see the agree/disagree callout
  on the Multimodal tab) - logged going forward via a new
  `modality_agreement` history column; predictions logged before this
  feature existed are simply excluded from this stat rather than
  guessed at
 Every section here gracefully explains what to run if the underlying
  file doesn't exist yet, instead of erroring


# Technologies Used

 Python 3.11
 scikit-learn
 pandas
 numpy
 librosa
 matplotlib
 sounddevice
 soundfile
 scipy
 joblib
 streamlit (EmotiSense web app)
 requests (optional Spotify enrichment - see "Spotify Integration")
 pytest (automated test suite - see "Robustness Testing")


# Project Structure

emotion-project/
 app/
    app.py (EmotiSense Streamlit web app)

 .streamlit/
    secrets.toml.example (copy to secrets.toml and fill in for optional
     Spotify enrichment/login - secrets.toml itself is gitignored)
    spotify_user_tokens.json (created automatically the first time you
     connect a Spotify account in the "My Spotify" tab - gitignored, as
     sensitive as a password)
    spotify_pending_oauth.json (short-lived, created/deleted automatically
     during the Connect Spotify flow - gitignored)

 data/
    audio/
    text_emotion.csv
    music.csv

 results/
    emotion_model.pkl
    vectorizer.pkl
    audio_emotion_model.pkl
    test_confusion_matrix.png
    audio_confusion_matrix.png
    test_classification_report.txt
    audio_classification_report.txt
    prediction_history.csv (created automatically by EmotiSense)

 src/
    main.py
    train_text_model.py
    train_text_distilbert.py (DistilBERT fine-tuning experiment - see
     ROADMAP.md item 1; not the production text model, see caveats there)
    train_audio_model.py
    sort_ravdess.py
    sort_cremad.py (sorts CREMA-D into the same data/audio/<emotion>/ layout, alongside RAVDESS)
    build_music_dataset.py (one-off: expanded data/music.csv via lyrics + the NRC Emotion Lexicon - see "Music Dataset")
    compare_text_models.py (Naive Bayes / Linear SVM / Random Forest / Gradient Boosting vs the baseline)
    compare_audio_models.py (Random Forest / Gradient Boosting vs the baseline SVM)
    chart_utils.py (shared chart styling used by EmotiSense and the comparison scripts)
    emotion_logic.py (pure ML/business logic - fusion, recommendation,
     feature extraction - shared by app.py and the test suite)
    spotify_oauth.py (Spotify user login - Authorization Code + PKCE,
     token storage, and the API calls behind the "My Spotify" tab)

 tests/
    conftest.py
    test_emotion_logic.py (automated test suite - see "Robustness Testing")
    test_spotify_oauth.py (automated test suite for src/spotify_oauth.py)
    adversarial_probe.py (adversarial input exploration, not a pass/fail test)
    dataset_shortcut_audit.py (checks the audio model for a RAVDESS/CREMA-D shortcut - see "Model Comparison")
    actor_leakage_audit.py (checks the audio train/test split for
     speaker leakage - see "Fairness & Generalisation Audit")

 requirements.txt
 README.md
 .gitignore


# Installation Guide

## 1. Clone or Download Project

Place project folder inside your working directory.



## 2. Create Virtual Environment

powershell
python -m venv venv


## 3. Activate Virtual Environment

powershell
.\venv\Scripts\activate



## 4. Install Dependencies

powershell
pip install -r requirements.txt



## 5. Spotify Integration (Optional)

EmotiSense works completely fine without this step - skip it and every
recommendation just shows as plain text, exactly as it always has. Do this
only if you want a real, playable Spotify embed under each recommendation,
and/or the "My Spotify" login tab (recently-played cross-reference,
auto-playlist creation, playback control, saving to Liked Songs).

1. Go to <https://developer.spotify.com/dashboard> and log in (or create a
   free Spotify account first if you don't have one).
2. Click "Create app". Any name/description works.
3. Under Redirect URIs, add exactly `http://127.0.0.1:8501`. Spotify's
   Feb 2025 security update rejects the hostname `localhost` now - it must
   be the literal IP `127.0.0.1`, on whatever port `streamlit run` is using
   (8501 by default). This single Redirect URI covers both the plain
   embed (which doesn't actually use it) and the "My Spotify" login tab
   (which does).
4. Open the new app's Settings and copy the Client ID and Client Secret.
5. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   (same folder) and paste your Client ID/Secret in. This file is
   gitignored - your credentials are never committed or shared.

powershell
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
notepad .streamlit\secrets.toml


If you're running Streamlit on a different port, also uncomment and set
`SPOTIFY_REDIRECT_URI` in `secrets.toml` to match - it must exactly equal
one of the Redirect URIs registered on the app in step 3.

If `.streamlit/secrets.toml` doesn't exist, the credentials in it don't
work, or Spotify's API is unreachable, EmotiSense silently falls back to
the plain-text recommendation and hides the "My Spotify" login tab's
features behind a setup message - it never crashes over any of this being
optional.

The first time you click "Connect Spotify account" in the "My Spotify"
tab, you'll be redirected to Spotify's own login/consent page, then back
to the app. The resulting login token is stored locally in
`.streamlit/spotify_user_tokens.json` (gitignored, never sent anywhere but
Spotify's API) so you don't have to log in again every time you restart
the app; use the "Disconnect" button in that tab to remove it. Playback
control and queueing additionally require your Spotify account to have
**Premium** - Spotify's API rejects those specific calls for free
accounts.



# Running the Project

## Launch Main System (CLI)

powershell
python .\src\main.py


## Launch the Web App (EmotiSense)

EmotiSense is a Streamlit front end over the same trained models - Text,
Audio, and Multimodal (combined) modes, a clear predicted emotion with a
confidence percentage, a probability chart across every emotion class,
upload-or-record audio input, and a persistent history of past
predictions.

Train the models first (see "Training Models" below) so that
`results/emotion_model.pkl`, `results/vectorizer.pkl` and
`results/audio_emotion_model.pkl` exist, then run:

powershell
streamlit run .\app\app.py


This opens the app in your browser at http://localhost:8501. Microphone
recording works the same way it does in the CLI - it uses your machine's
own microphone through `sounddevice`, so it only works when you run the app
locally (not if it's ever deployed to a remote server without mic access).

### Desktop shortcut (skip the terminal entirely)

`Launch EmotiSense.bat` does the above for you - double-click it (or a
shortcut to it) instead of opening a terminal and typing the command
above. It always runs from its own folder regardless of where a shortcut
to it lives, and leaves a console window open so you can see logs or stop
the app with Ctrl+C. `emotisense.ico` is a matching icon if you want to
set it on a desktop shortcut (right-click the shortcut -> Properties ->
Change Icon).


# Training Models

## Train Text Model

powershell
python .\src\train_text_model.py

## Train Audio Model

powershell
python .\src\train_audio_model.py


# Model Comparison

`src/compare_text_models.py` and `src/compare_audio_models.py` benchmark a
handful of classical ML alternatives on the exact same features used in
production. This comparison is how EmotiSense's current text and audio
models were chosen - both scripts have already done their job once, and
EmotiSense now runs on the winners.

powershell
python .\src\compare_text_models.py
python .\src\compare_audio_models.py


Each script trains every model on the same train/test split, times it, and
writes a comparison table + bar charts to `results/` (`text_model_comparison.*`,
`audio_model_comparison.*`) plus the fitted models themselves under
`results/models_comparison/` in case you want to try swapping in a
different one later - the script's own output tells you exactly which
files to copy over `results/emotion_model.pkl` / `results/vectorizer.pkl`
(or `results/audio_emotion_model.pkl` for audio) to do that. The original
baseline models this project shipped with (Logistic Regression for text,
SVM for audio) are kept as `results/emotion_model.pkl.bak` and
`results/vectorizer.pkl.bak` in case you ever want to roll back.

## Text models

Real results from this project's text dataset (16,000 train / 2,000 test
rows), from the run that promoted the current production model. Linear
SVM has no native `predict_proba`, so it's wrapped in
`CalibratedClassifierCV` (5-fold Platt/sigmoid scaling) here and in
production - without that wrapper, every confidence score the app shows
for text predictions would silently be a fake 100%/0% one-hot value
instead of a real probability distribution (see "Robustness Testing" for
why this matters, and `tests/test_spotify_oauth.py`'s sibling coverage
philosophy for the general idea):

| Model | Accuracy | F1 Score (macro) | Training Time (s) |
|---|---|---|---|
| Linear SVM (now in production, calibrated) | 89.2% | 83.9% | 2.03 |
| Logistic Regression (original baseline) | 86.9% | 80.6% | 1.12 |
| Gradient Boosting | 86.0% | 83.1% | 48.34 |
| Random Forest | 85.9% | 81.3% | 6.37 |
| Naive Bayes | 71.6% | 48.2% | 0.02 |

Linear SVM beat the original Logistic Regression baseline on both accuracy
and F1, so it's now the model `app/app.py` actually serves.

## Audio models

Real results from this project's combined audio dataset - RAVDESS plus
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (sorted in by
`src/sort_cremad.py`, same `data/audio/<emotion>/` folder layout
`sort_ravdess.py` uses), 7,035 samples total (5,632 train / 1,403 test, 54
features per sample, 5 emotion classes - angry/fear/happy/sad at 1,463
each, neutral at 1,183), split with `StratifiedGroupKFold` grouped by
actor (see "Fairness & Generalisation Audit" below for why a plain
`train_test_split` isn't used here):

| Model | Accuracy | F1 Score (macro) | Training Time (s) |
|---|---|---|---|
| Random Forest (now in production) | 52.8% | 50.8% | 7.23 |
| Gradient Boosting | 51.4% | 49.9% | 97.07 |
| SVM (original baseline) | 49.7% | 48.6% | 39.62 |

Random Forest wins this comparison - a genuinely different winner than
the Gradient Boosting result this project shipped with before the actor
leak (see below) was fixed, not just a smaller number under the same
ranking. That said, accuracy dropped substantially from the RAVDESS-only
figure this project shipped with originally (64.2%). That's a real,
measured result from adding CREMA-D's more varied speakers/recording
conditions, not a bug: `tests/dataset_shortcut_audit.py` exists
specifically to check whether the *original* 64.2% partly reflected the
model learning "which dataset/recording setup is this" rather than
genuine emotion cues (RAVDESS alone is 24 actors reading two fixed
scripted sentences in one studio - an easy pattern for a shallow model to
latch onto). Run it to see the accuracy broken down by source dataset. If
you'd rather keep the higher-scoring RAVDESS-only model instead, it's
backed up at `results/audio_emotion_model.pkl.ravdess_only_gb.bak`.


# Robustness Testing

Phase 6 of this project's post-submission roadmap. Four parts: an
automated test suite, input validation in the app itself, a low-confidence
warning, and a deliberate adversarial probe of both models. The core
ML/business logic (fusion, recommendation, feature extraction, etc.) lives
in `src/emotion_logic.py` - it has no Streamlit dependency, specifically
so it can be unit tested directly instead of needing a running app.

## Automated Test Suite

powershell
pytest


`tests/test_emotion_logic.py` covers the fusion math (`fuse_predictions`
sums to 1, uses the real accuracy weights, keeps a neutral audio signal
separate from joy rather than silently merging them), the recommendation
engine (`recommend_music`'s confidence weighting, "lift my mood" shifting,
repeat-avoidance, and fallback behaviour), the Chat tab's templated
replies (`generate_conversational_reply`'s per-emotion template pool,
repeat-avoidance, mood-mode-aware transition line, and low-confidence
hedging), history file backward compatibility
(`normalize_history_columns`), the new input-validation and
confidence-warning helpers below, and `extract_audio_features` itself
(shape of the real feature vector, and that a silent clip correctly raises
instead of producing garbage features). A couple of tests load the real
production models from `results/` for an end-to-end sanity check, and skip
automatically (not fail) if those files aren't there yet.

`tests/test_spotify_oauth.py` covers `src/spotify_oauth.py`'s PKCE
generation, authorize-URL construction, token-expiry math, and local
token-file persistence - everything that doesn't need a real network
call. The functions that actually call Spotify's API are deliberately
left untested here (only checked live, via the app's own graceful
degradation) since mocking Spotify's real behaviour would just be
checking this project's own assumptions about it, not the behaviour
itself.

`tests/dataset_shortcut_audit.py` is a one-off diagnostic, not part of
the pass/fail suite: it re-splits the audio dataset exactly as
`train_audio_model.py` does, then checks whether the production model's
accuracy differs sharply between RAVDESS-origin and CREMA-D-origin test
samples - a large gap would mean the model partly learned "which
dataset/recording setup is this" rather than genuine emotion cues. See
"Model Comparison" for why this matters now that both datasets are
combined.

## Fairness & Generalisation Audit

An extension of Phase 6's adversarial testing, prompted by a real
question: does the audio model's test accuracy reflect genuine emotion
recognition, or could it be partly recognising *who* is speaking?

`tests/actor_leakage_audit.py` checks this directly. RAVDESS and CREMA-D
filenames both encode a per-actor speaker ID, and each actor recorded
many takes across every emotion. The original split -
`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` -
stratified only by emotion label, with no actor grouping at all. Running
the audit against that split found:

 **100% of test-set actors also appeared in the training set** - every
  single one, across both RAVDESS (24/24) and CREMA-D (91/91)
 Every held-out test row's speaker had therefore already been heard
  during training, in a different clip

This means the audio model's accuracy figures could have been partly
"I recognise this voice" rather than purely "this sounds angry" - a real
methodological gap, not a hypothetical one. Fix: `train_audio_model.py`
and `compare_audio_models.py` now split with `StratifiedGroupKFold`,
grouped by actor (see `actor_of()` in `train_audio_model.py`), so no
speaker's clips can appear in both train and test. Both scripts print and
save a leakage check (`leaked_actors_between_train_test` in
`results/audio_model_metadata.json`) confirming 0 on every run since.

Concretely, the corrected, leakage-free numbers are the ones in "Model
Comparison" above (Random Forest 52.8%, Gradient Boosting 51.4%, SVM
49.7%) - notably, the *winner changed* (Random Forest, not Gradient
Boosting) once actor leakage was removed, not just the accuracy number,
which is itself informative: Gradient Boosting's earlier edge may have
come partly from fitting the leaked speaker patterns more aggressively
than Random Forest did, rather than from truly better emotion
generalisation.

`tests/dataset_shortcut_audit.py` was re-run against the corrected split
too, since its own numbers were computed under the old leaky one.
Corrected: CREMA-D-origin test samples score 54.1% (n=1,223), RAVDESS-
origin score 43.9% (n=180) - both moved from the old leaky-split numbers
(51.4%/34.2%), and the gap between them *narrowed* under the honest
split. Still no shortcut-learning smoking gun (neither source is
near-perfect or near-zero), and RAVDESS's real accuracy turns out to be
meaningfully better than the leaky split suggested once its actors are no
longer split between train and test.

Run either audit yourself:

powershell
python .\tests\actor_leakage_audit.py
python .\tests\dataset_shortcut_audit.py


## Input Validation & Graceful Degradation

 Text input longer than 2,000 characters is truncated with a visible
  note rather than vectorising an entire pasted document
 Uploaded audio files over 25 MB are rejected with a clear message
  instead of being processed (the app only ever looks at the first 5
  seconds of any clip anyway)
 A silent clip (after trimming) is caught and shown as "Couldn't analyse
  this clip - try a louder or longer recording" instead of a raw
  exception
 Every other unexpected audio-processing error still shows a message
  instead of crashing the app, exactly as it already did

## Low-Confidence / Out-of-Distribution Warning

Every prediction (Text, Audio, Multimodal, Chat) is now checked by
`confidence_warning()` and shown an extra note when any of:

 the input contains a negation word ("not"/"never"/"n't"/etc.)
  immediately followed by a word with a known sentiment score (see
  "Negation-Aware Warning" below) - checked *regardless* of how
  confident the model claims to be, since this is exactly the failure
  mode that produces a confident-but-wrong answer
 the top confidence is below 40% - the model isn't sure of anything in
  particular, or
 the top two candidates are within 10 percentage points of each other -
  the model is genuinely torn, even if the top score alone looks
  moderate

This doesn't change the prediction shown - it's a "take this with a grain
of salt" signal layered on top, so a low-information, negated, or
out-of-vocabulary input reads as uncertain instead of presenting the same
false confidence as a clear-cut case.

### Negation-Aware Warning

Added after a real bug report: Chat mode replied "That's lovely to hear"
to "im not having the best day", because the text model's TF-IDF features
have no way to represent that "not" negates "best" - bag-of-words sees
two unrelated tokens, not a flipped meaning. Rather than guessing which
emotion the input "really" is (a wrong guess is just as misleading as the
original wrong-but-confident answer), `detect_negated_sentiment()` in
`src/emotion_logic.py` looks for the *specific pattern* - a negation word
immediately followed (within 3 words) by a word with a known sentiment
score - and surfaces it as an explicit warning naming the actual words
involved, rather than a blanket "any negation word present" heuristic
(which would also fire on negations that don't flip anything relevant,
e.g. "I did not go to the shop").

Word-level sentiment scores come from
[AFINN-en-165](https://github.com/fnielsen/afinn) (Nielsen, 2011;
Apache-2.0, Technical University of Denmark) - bundled directly in this
repo at `data/afinn_sentiment_lexicon.txt` (~3,400 words, unlike the NRC
Emotion Lexicon used for `data/music.csv` - see "Music Dataset" below -
AFINN's Apache-2.0 license explicitly permits redistribution, so it can
ship as a real runtime file here rather than only being usable as a
one-off build-time reference).

Example: "I am not happy at all" scores confidently as Joy from the SVM
alone (TF-IDF sees "happy" and nothing tells it "not" matters) - but
`detect_negated_sentiment` finds "not" immediately before "happy" (AFINN
score +3) and surfaces: `"not ... happy" negates a positive-scored word
(AFINN score +3) - the text model is bag-of-words and can't represent
that flip, so this confident-looking read (96% Joy) may well have the
sentiment backwards. Take it with real caution.`

## Adversarial Input Exploration

powershell
python .\tests\adversarial_probe.py


Runs both production models against a battery of deliberately tricky
inputs and reports what happens - not a pass/fail test, an evaluation.
Real results against this project's production text model, re-run after
the Linear SVM was wrapped in `CalibratedClassifierCV` (see "Model
Comparison") - the numbers below shifted from an earlier version of this
table because of that change, some for the better and one for the worse
(see the new gap called out below):

| Input | Predicted | Confidence | Flagged? |
|---|---|---|---|
| Sarcasm ("Oh great, cancelled again. Just perfect.") | Joy | 87.6% | No |
| Negation ("I am not happy at all") | Joy | 92.2% | No |
| Double negative ("I wouldn't say I'm not pleased") | Joy | 96.7% | No |
| Very short ("ok") | Joy | 99.4% | No |
| ALL CAPS shouting ("I AM SO ANGRY RIGHT NOW") | Anger | 97.2% | No |
| Mixed sentiment (happy for you, sad for me) | Sadness | 71.6% | No |
| Non-English (Spanish) / Gibberish / Emoji-only / Punctuation-only | Joy | 57.3% | **No** |
| Neutral/factual ("The meeting is at 3pm tomorrow") | Fear | 32.4% | Yes |

Takeaways:

 TF-IDF has no concept of word order or negation, so sarcasm and
  negation are genuine, unfixed blind spots - "not happy" and "happy"
  share the same strongly-weighted token, and the model has no mechanism
  to catch the flip. This is a known, well-documented limitation of
  bag-of-words models in general, not a bug specific to this project -
  fixing it would need a sequence-aware model (see "Future Improvements").
  Calibration made this *look* worse (92.2%/96.7% vs. an earlier 72.6%),
  but that's the model's genuine confidence surfacing more honestly, not
  a new problem the calibration introduced.
 **New gap opened by calibration:** gibberish, emoji-only, non-English,
  and punctuation-only text all still vectorise to the same all-zero
  TF-IDF input (none of their tokens exist in the vocabulary) - but where
  that used to correctly score as low-confidence, the calibrated model
  now reports 57.3% for all of them, comfortably above both of
  `confidence_warning()`'s thresholds (40% floor, 10-point margin), so
  **none of them get flagged any more**. Calibration fit a sigmoid curve
  to the model's decision-function output, and the near-zero decision
  score this particular all-zero input produces happens to map to a
  higher probability than before - a real regression in the
  out-of-distribution safety net worth a follow-up (e.g. detecting "zero
  TF-IDF vocabulary overlap" directly, rather than relying on the
  confidence score to indirectly imply it).
 A genuinely neutral, unemotional sentence is still correctly flagged as
  low-confidence rather than confidently misclassified.

Real results against the production audio model (Random Forest, trained
on the merged RAVDESS+CREMA-D dataset with the actor-grouped split - see
"Fairness & Generalisation Audit"), probed with synthetic non-speech
signals:

| Input | Predicted | Confidence | Flagged? |
|---|---|---|---|
| Pure 220Hz tone (2s) | Fear | 32.5% | Yes |
| Pure 880Hz tone (2s) | Fear | 33.0% | Yes |
| White noise (2s) | Sad | 32.0% | Yes |
| Very short clip (0.3s tone) | Fear | 35.5% | Yes |
| Quiet tone near silence threshold | Fear | 34.5% | Yes |

Every single synthetic non-speech probe is now correctly flagged as
low-confidence - a real improvement over the previous production model
(Gradient Boosting, under the leaky split), which was confidently wrong
on all five of these (78-96% confidence, none flagged). The underlying
limitation is unchanged - the audio model still has no explicit concept
of "not speech at all", it just picks the closest-matching emotion class
regardless - but Random Forest's confidence estimates on out-of-
distribution input are honest about that uncertainty in a way Gradient
Boosting's weren't. Worth also trying by hand: real background noise/
music recordings, heavily accented or non-English speech, and
deliberately flat/monotone delivery.


# Dataset Requirements

## Text Dataset

 Labelled emotional text CSV
 Must include text + emotion labels

## Audio Dataset

 WAV files sorted into emotion folders:

text
data/audio/
 happy/
 sad/
 angry/
 fear/
 neutral/

 Two source datasets combined into that same folder layout:
  [RAVDESS](https://zenodo.org/record/1188976) (sorted by
  `src/sort_ravdess.py`) and [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
  (sorted by `src/sort_cremad.py`) - see "Model Comparison" for why both
  are in the mix, and what that traded off against RAVDESS alone

## Music Dataset

CSV containing:

 song
 artist
 emotion

`love` and `neutral` are still the original 5 hand-picked songs each; the
other five emotions were expanded to 30 each by `src/build_music_dataset.py`
using real songs from the
[ernestchu/lyrics-emotion-classification](https://huggingface.co/datasets/ernestchu/lyrics-emotion-classification)
dataset on Hugging Face, labelled independently via the
[NRC Emotion Lexicon](https://saifmohammad.com/WebPages/AccessResource.htm)
(Mohammad & Turney) rather than that dataset's own undocumented label
column. **Citation** (required by the NRC lexicon's terms wherever it's
used): Saif M. Mohammad and Peter D. Turney, "Crowdsourcing a Word-Emotion
Association Lexicon," Computational Intelligence, 29(3), 436-465, 2013.

## Sentiment Lexicon

`data/afinn_sentiment_lexicon.txt` - word-level sentiment scores (-5 to
+5) from [AFINN-en-165](https://github.com/fnielsen/afinn) (Nielsen,
2011; Apache-2.0, Technical University of Denmark), bundled directly in
this repo since (unlike NRC above) its license permits redistribution.
Powers the negation-aware confidence warning - see "Negation-Aware
Warning" under "Robustness Testing".


# Evaluation Outputs

Generated in `/results/`:

 Classification reports
 Accuracy metrics
 Confusion matrices
 Saved trained models
 Metadata files
 Prediction history (from EmotiSense)


# Performance Summary

## Text Model

 Linear SVM, calibrated via `CalibratedClassifierCV` for real confidence
  scores (chosen over the original Logistic Regression baseline via
  `src/compare_text_models.py` - see "Model Comparison" above)
 TF-IDF Vectorisation
 89.2% accuracy / 83.9% F1 (macro)
 Original Logistic Regression baseline (~87% accuracy) backed up as
  `results/emotion_model.pkl.bak`; the pre-calibration Linear SVM is
  backed up separately as `results/emotion_model.pkl.uncalibrated.bak`

## Audio Model

 Random Forest Classifier (chosen over Gradient Boosting and the
  original SVM baseline via `src/compare_audio_models.py` - see "Model
  Comparison" above), trained on RAVDESS + CREMA-D combined, split with
  `StratifiedGroupKFold` grouped by actor (see "Fairness &
  Generalisation Audit" - the earlier plain stratified split leaked
  every test-set actor into training)
 MFCC + ZCR + Chroma + RMS
 52.8% accuracy / 50.8% F1 (macro) - down from 64.2%/63.5% on RAVDESS
  alone; see "Model Comparison" for why that's a meaningful result, not a
  bug, and how to roll back if you'd rather keep the higher-scoring
  RAVDESS-only model
 Three backups kept: the original SVM baseline as a plain `.bak` file;
  the RAVDESS-only Gradient Boosting model as
  `results/audio_emotion_model.pkl.ravdess_only_gb.bak`; and the
  CREMA-D-combined Gradient Boosting model trained under the since-fixed
  leaky split as `results/audio_emotion_model.pkl.leaky_split_gb.bak`


# Ethical Considerations

This project acknowledges:

 Dataset bias risks
 Speech diversity limitations
 Privacy considerations
 Accessibility concerns
 Responsible AI deployment principles

EmotiSense displays a standing reminder in its footer that predictions are
statistical patterns associated with emotion, not a definitive read of how
someone feels.

The optional Spotify login (see "Spotify Login") only ever reads/writes
your own account's data (via Spotify's standard OAuth consent screen,
where you can see and revoke exactly what's being granted) and stores
your login token locally in a gitignored file - nothing about your
listening habits or predictions is sent anywhere except directly to
Spotify's own API.


# Future Improvements

Potential future upgrades:

 ~~Deep learning models (CNN/LSTM/BERT)~~ text half attempted - real
  results (89.0% accuracy, matching production, but didn't fix the
  negation blind spot it targeted) in `ROADMAP.md` item 1
 Larger real-world datasets (audio half partly done - see "Audio Dataset")
 ~~Improved fairness and generalisation~~ leakage audit done - see
  "Fairness & Generalisation Audit" above
 A jointly-trained multimodal model, if a paired text+audio dataset
  becomes available (current multimodal mode uses decision-level fusion
  of two independently-trained models instead - see "Multimodal Mode")
 Real-time streaming emotion detection

See `ROADMAP.md` for each of these scoped out in more detail - what it'd
actually take, effort/complexity, real results for the items attempted so
far, and suggested next steps for what's left.


# Author

**Jayden Steadman-Jeffrey**
Final Year Computer Science Synoptic Project
Manchester Metropolitan University

Continued as an independent portfolio project post-submission.


# License

All Rights Reserved - see [LICENSE](LICENSE). This repository is public
for portfolio/demonstration purposes; it is not open-source and reuse
requires permission from the author.
