# Container image for api/main.py (the deployed DistilBERT API) - not
# app/app.py (the Streamlit app), which isn't containerized here. Mirrors
# render.yaml's build/start commands so behaviour matches the existing
# Render deployment; MODEL_SOURCE is passed in at `docker run` time
# (see render.yaml) rather than baked into the image, since the ~268MB
# results/distilbert_emotion_model/ directory is gitignored and never
# part of the build context anyway - the container always loads weights
# from the Hugging Face Hub at startup.

FROM python:3.11-slim

WORKDIR /app

# System deps for building/running torch's CPU wheel and friends.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Only what api/main.py actually imports at runtime: itself, and
# src/emotion_logic.py (see main.py's sys.path.insert + import).
COPY api/ api/
COPY src/emotion_logic.py src/emotion_logic.py

ENV MODEL_SOURCE=local
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
