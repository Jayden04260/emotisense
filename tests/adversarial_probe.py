"""
Adversarial input exploration (Phase 6, "Robustness Testing" - see
README). This is NOT part of the pass/fail pytest suite in
test_emotion_logic.py - it's an evaluation script that deliberately feeds
both production models tricky, real-world-ish inputs and reports what
they do, so the project's actual failure modes are documented rather than
assumed. Re-run any time after retraining to see whether a limitation
noted here has changed.

Run from the project root with:

    python tests/adversarial_probe.py

Needs the real production artefacts (results/emotion_model.pkl,
vectorizer.pkl, audio_emotion_model.pkl) - trains nothing itself.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np

import emotion_logic as el


def probe_text():
    print("=" * 70)
    print("TEXT MODEL - adversarial probes")
    print("=" * 70)

    model_path = RESULTS_DIR / "emotion_model.pkl"
    vectorizer_path = RESULTS_DIR / "vectorizer.pkl"
    if not (model_path.exists() and vectorizer_path.exists()):
        print("Skipped - results/emotion_model.pkl or vectorizer.pkl not found.")
        return

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    probes = [
        ("Sarcasm", "Oh great, my flight got cancelled again. Just perfect."),
        ("Mixed sentiment", "I'm so happy for you but honestly it makes me a little sad too"),
        ("Non-English (Spanish)", "Estoy muy feliz hoy"),
        ("Gibberish", "asdkfj qwoeiru zzzxxx blorp"),
        ("Emoji-only", "\U0001F60A\U0001F60A\U0001F60A"),
        ("Very short", "ok"),
        ("Negation", "I am not happy at all"),
        ("Double negative", "I wouldn't say I'm not pleased"),
        ("ALL CAPS shouting", "I AM SO ANGRY RIGHT NOW"),
        ("Neutral/factual", "The meeting is scheduled for 3pm tomorrow."),
        ("Single punctuation", "..."),
    ]

    for name, text in probes:
        vec = vectorizer.transform([text])
        label, probs = el.predict_with_confidence(model, vec)
        warning = el.confidence_warning(probs)
        top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        top3_str = ", ".join(f"{k}={v * 100:.0f}%" for k, v in top3)
        flag = "  [FLAGGED by confidence_warning]" if warning else ""
        print(f"{name:22} -> {label:10} ({probs[label] * 100:5.1f}%){flag}")
        print(f"{'':22}    top 3: {top3_str}")


def _write_wav(path, samples, sample_rate=22050):
    from scipy.io.wavfile import write as write_wav

    samples_int16 = np.int16(np.clip(samples, -1.0, 1.0) * 32767)
    write_wav(str(path), sample_rate, samples_int16)


def probe_audio(tmp_dir):
    print()
    print("=" * 70)
    print("AUDIO MODEL - adversarial probes (synthetic, non-speech signals)")
    print("=" * 70)

    model_path = RESULTS_DIR / "audio_emotion_model.pkl"
    if not model_path.exists():
        print("Skipped - results/audio_emotion_model.pkl not found.")
        return
    try:
        import librosa  # noqa: F401 (just checking availability)
    except ImportError:
        print("Skipped - librosa isn't installed in this environment (pip install -r requirements.txt).")
        return

    model = joblib.load(model_path)
    sample_rate = 22050

    def tone(freq, seconds):
        t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
        return 0.5 * np.sin(2 * np.pi * freq * t)

    rng = np.random.default_rng(0)
    probes = {
        "Pure 220Hz tone (2s, non-speech)": tone(220, 2),
        "Pure 880Hz tone (2s, non-speech)": tone(880, 2),
        "White noise (2s)": rng.normal(0, 0.3, sample_rate * 2),
        "Very short clip (0.3s tone)": tone(440, 0.3),
        "Quiet tone near silence threshold": 0.02 * np.sin(2 * np.pi * 220 * np.linspace(0, 2, sample_rate * 2)),
    }

    for name, samples in probes.items():
        wav_path = tmp_dir / "probe.wav"
        _write_wav(wav_path, samples, sample_rate)
        try:
            features = el.extract_audio_features(str(wav_path)).reshape(1, -1)
            label, probs = el.predict_with_confidence(model, features)
            warning = el.confidence_warning(probs)
            flag = "  [FLAGGED by confidence_warning]" if warning else ""
            print(f"{name:36} -> {label:10} ({probs[label] * 100:5.1f}%){flag}")
        except ValueError as e:
            print(f"{name:36} -> rejected: {e}")

    print()
    print("Also worth trying by hand (can't be scripted): real background")
    print("noise/music recordings, heavily accented or non-English speech,")
    print("and deliberately flat/monotone speech - see README 'Robustness")
    print("Testing' for what to look for.")


if __name__ == "__main__":
    import tempfile

    probe_text()
    with tempfile.TemporaryDirectory() as tmp:
        probe_audio(Path(tmp))
