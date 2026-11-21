"""
Automated test suite for api/main.py - the deployable FastAPI wrapper
around the fine-tuned DistilBERT model (see ROADMAP.md item 1).

Marked requires_production_models since every test here loads the real
results/distilbert_emotion_model/ artefacts (~268MB) - there's no
meaningful way to test "does the API correctly wire up the actual model"
without the actual model.

Run from the project root with:

    pytest tests/test_api.py
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import api.main as main_module  # noqa: E402 (must follow sys.path insert)
from api.main import app  # noqa: E402

SIX_LABELS = {"anger", "fear", "joy", "love", "sadness", "surprise"}

pytestmark = pytest.mark.requires_production_models


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_schema(client):
    response = client.post("/predict", json={"text": "I am so happy today"})
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in set(body["probabilities"])
    assert 0.0 <= body["confidence"] <= 1.0
    # v1-v3 models have six classes; v4 (src/train_text_transformer.py) adds neutral.
    assert set(body["probabilities"]) in (SIX_LABELS, SIX_LABELS | {"neutral"})
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert body["probabilities"][body["label"]] == pytest.approx(body["confidence"])


def test_predict_confident_positive_text_has_no_warning(client):
    response = client.post("/predict", json={"text": "I am so happy today, everything is wonderful"})
    body = response.json()
    assert body["label"] == "joy"
    assert body["warning"] is None


def test_negation_warning_wording_for_v1_to_v3_models():
    """Regression test for the wording bug caught while building this API:
    the reused warning text originally said "the text model is bag-of-
    words," which is true of the OTHER (TF-IDF) model but false for
    DistilBERT. v1-v3 (six classes, no neutral) still get a negation caveat
    - checked here directly against _build_warning so it doesn't depend on
    which model happens to be loaded."""
    six_class_probs = {"joy": 0.95, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "love": 0.01, "surprise": 0.01}
    warning = main_module._build_warning(six_class_probs, "I am not happy at all")

    assert warning is not None
    assert "bag-of-words" not in warning or "not a bag-of-words" in warning
    assert "not" in warning.lower()


def test_no_negation_caveat_for_the_v4_model(client):
    """v4 (neutral class present) handles "not happy" itself - see ROADMAP.md
    "v4" - so a confident v4 read gets no negation caveat."""
    body = client.post("/predict", json={"text": "I am not happy at all"}).json()
    assert "neutral" in body["probabilities"]
    assert body["warning"] is None or "negates" not in body["warning"]

def test_predict_matches_documented_v4_probe_results(client):
    """Pins the API to the behaviour documented in
    results/distilbert_negation_probe.txt for the v4 model, so a silent model
    swap (e.g. deploying an older version by mistake) breaks this test
    rather than going unnoticed. Includes a probe v4 still gets wrong
    ("I am not sad" -> sadness) - this records behaviour, not wishes."""
    happy_response = client.post("/predict", json={"text": "I am not happy at all"})
    assert happy_response.json()["label"] == "sadness"  # v1-v3 said joy

    sad_response = client.post("/predict", json={"text": "I am not sad"})
    assert sad_response.json()["label"] == "sadness"

    neutral_response = client.post("/predict", json={"text": "The meeting is at 3pm tomorrow"})
    assert neutral_response.json()["label"] == "neutral"  # v1-v3 had no neutral class
