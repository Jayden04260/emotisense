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

from api.main import app  # noqa: E402 (must follow sys.path insert)

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

    assert body["label"] in {"anger", "fear", "joy", "love", "sadness", "surprise"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == {"anger", "fear", "joy", "love", "sadness", "surprise"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert body["probabilities"][body["label"]] == pytest.approx(body["confidence"])


def test_predict_confident_positive_text_has_no_warning(client):
    response = client.post("/predict", json={"text": "I am so happy today, everything is wonderful"})
    body = response.json()
    assert body["label"] == "joy"
    assert body["warning"] is None


def test_predict_flags_negation_with_accurate_wording(client):
    """Regression test for the wording bug caught while building this API:
    the reused warning text originally said "the text model is bag-of-
    words," which is true of the OTHER (TF-IDF) model but false for
    DistilBERT. This checks the corrected message doesn't make that claim,
    and that the warning still fires - a real, documented (ROADMAP.md)
    limitation of this specific fine-tuned model, not the architecture."""
    response = client.post("/predict", json={"text": "I am not happy at all"})
    body = response.json()

    assert body["warning"] is not None
    assert "bag-of-words" not in body["warning"] or "not a bag-of-words" in body["warning"]
    assert "not" in body["warning"].lower()


def test_predict_matches_documented_v1_negation_probe_results(client):
    """Same two sentences ROADMAP.md's v1 write-up documents as failing -
    this pins the API to that exact documented behavior, so a silent
    model swap (e.g. accidentally deploying v2/v3 instead of v1) would
    break this test rather than go unnoticed."""
    happy_response = client.post("/predict", json={"text": "I am not happy at all"})
    assert happy_response.json()["label"] == "joy"

    sad_response = client.post("/predict", json={"text": "I am not sad"})
    assert sad_response.json()["label"] == "sadness"
