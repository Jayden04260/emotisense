"""
scripts/push_distilbert_to_hub.py

Uploads the trained results/distilbert_emotion_model/ (model + tokenizer)
to a Hugging Face Hub model repo, so a deployed instance of api/main.py
can load it via MODEL_SOURCE=<your-username>/<repo-name> instead of
needing the ~268MB directory checked into git (it's gitignored - see
.gitignore's "Trained model artefacts" comment).

Requires being logged in first - this script deliberately does NOT
accept a token as an argument or read one from a file, so there's no
credential to accidentally commit. Either:

    huggingface-cli login

or set the HF_TOKEN environment variable for this one command:

    HF_TOKEN=hf_xxx python scripts/push_distilbert_to_hub.py your-username/emotisense-distilbert

Usage:

    python scripts/push_distilbert_to_hub.py <your-hf-username>/<repo-name>

Then set MODEL_SOURCE to that same repo id in render.yaml (or wherever
the API is deployed) and in the README "Deployed DistilBERT API"
section's `curl` example / live URL.
"""

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "results" / "distilbert_emotion_model"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/push_distilbert_to_hub.py <your-hf-username>/<repo-name>")
        sys.exit(1)
    repo_id = sys.argv[1]

    if not MODEL_DIR.exists():
        print(f"No model found at {MODEL_DIR} - run src/train_text_distilbert.py first.")
        sys.exit(1)

    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    print(f"Pushing to https://huggingface.co/{repo_id} ...")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print(f"\nDone. Set MODEL_SOURCE={repo_id} in render.yaml (or wherever this is deployed).")


if __name__ == "__main__":
    main()
