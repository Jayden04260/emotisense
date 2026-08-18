"""
api/main.py

FastAPI service wrapping the fine-tuned DistilBERT text-emotion model
(see src/train_text_distilbert.py, ROADMAP.md item 1) as a deployable
HTTP endpoint - the piece none of this author's other local-only
projects have: something actually served, not just runnable locally.

This is deliberately a wrapper around the DistilBERT candidate, not the
production Linear SVM app/app.py serves - the point of deploying this
specific model is to demonstrate the PyTorch/transformers fine-tuning
work directly (see README "Model Comparison"), not to duplicate the
Streamlit app as a second interface.

Reuses src/emotion_logic.py's detect_negated_sentiment (the lexicon/
negation-detection logic itself) but NOT confidence_warning's negation
message text verbatim - that message says "the text model is bag-of-
words," which is true of the *other* model (Linear SVM over TF-IDF) but
factually wrong for DistilBERT, a contextual transformer. ROADMAP.md's
v1/v2/v3 write-ups proved DistilBERT has the same practical blind spot
despite not being bag-of-words ("I am not happy at all" still scores
~94-96% Joy either way) - so the warning is still shown here, on real
evidence it's needed for this model too, just worded honestly about
*why* (a training-data/objective gap, not an architectural one).

Run locally with:

    uvicorn api.main:app --reload

Environment variables:
    MODEL_SOURCE   "local" (default, loads results/distilbert_emotion_model/)
                   or a Hugging Face Hub repo id (e.g. "username/emotisense-distilbert")
                   to load from the Hub instead - see README "Deployment" for why
                   a deployed instance can't just read the local results/ folder.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
from emotion_logic import confidence_warning, detect_negated_sentiment  # noqa: E402

LOCAL_MODEL_DIR = ROOT_DIR / "results" / "distilbert_emotion_model"
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local")
MAX_LENGTH = 64  # matches train_text_distilbert.py

_model = None
_tokenizer = None
_label_names = None


def _load_model():
    global _model, _tokenizer, _label_names
    if _model is not None:
        return

    source = str(LOCAL_MODEL_DIR) if MODEL_SOURCE == "local" else MODEL_SOURCE
    _tokenizer = DistilBertTokenizerFast.from_pretrained(source)
    # low_cpu_mem_usage avoids transformers creating a full duplicate copy of
    # the model's weights in memory while loading (its default behaviour) -
    # first deploy attempt on Render's free tier (512MB RAM) was killed with
    # exit 137 (SIGKILL/OOM) during startup, before this was set. That alone
    # still wasn't enough (second attempt also hit exit 137, this time during
    # the Hub download/load) - loading straight into float16 halves the
    # ~268MB fp32 weight footprint on top of that. CPU inference in fp16 is
    # slower than fp32 per-op, but this model is tiny (67M params) and
    # requests aren't latency-sensitive here, so the memory saving wins.
    _model = DistilBertForSequenceClassification.from_pretrained(
        source, low_cpu_mem_usage=True, torch_dtype=torch.float16
    )
    _model.eval()
    _label_names = [_model.config.id2label[i] for i in range(_model.config.num_labels)]


def _build_warning(prob_dict: dict[str, float], text: str) -> str | None:
    """Same underlying signals as emotion_logic.confidence_warning, but
    with the negation message reworded for this model - see module
    docstring for why the original wording (written for the TF-IDF
    model) doesn't apply verbatim here."""
    negated = detect_negated_sentiment(text)
    if negated is not None:
        neg_word, sent_word, score = negated
        top_label = max(prob_dict, key=prob_dict.get)
        polarity = "positive" if score > 0 else "negative"
        return (
            f'"{neg_word} ... {sent_word}" negates a {polarity}-scored word (AFINN score {score:+d}). '
            "Testing has shown this fine-tuned DistilBERT model still doesn't reliably represent that "
            f"flip (see ROADMAP.md item 1) despite being a contextual transformer, not a bag-of-words "
            f"model - so this confident-looking read ({prob_dict[top_label] * 100:.0f}% "
            f"{top_label.capitalize()}) may well have the sentiment backwards. Take it with real caution."
        )
    # Fall back to the generic (model-agnostic) low-confidence/margin checks -
    # those don't reference an architecture, so the original wording is fine.
    return confidence_warning(prob_dict)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    warning: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="EmotiSense DistilBERT API",
    description="Fine-tuned DistilBERT text-emotion classifier - see the emotisense repo README for full context.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_source": MODEL_SOURCE}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    encoding = _tokenizer(
        request.text, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _model(**encoding).logits
    probs = torch.softmax(logits, dim=-1)[0]
    prob_dict = {label: float(p) for label, p in zip(_label_names, probs)}
    top_label = max(prob_dict, key=prob_dict.get)

    return PredictResponse(
        label=top_label,
        confidence=prob_dict[top_label],
        probabilities=prob_dict,
        warning=_build_warning(prob_dict, request.text),
    )
