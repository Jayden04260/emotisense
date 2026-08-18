"""
scripts/push_distilbert_fp16_to_hub.py

Re-exports results/distilbert_emotion_model/ (the canonical fp32 model -
left untouched on disk) as float16 weights and pushes THAT to the Hugging
Face Hub repo api/main.py deploys from.

Why this exists on top of push_distilbert_to_hub.py: casting to float16
*after* from_pretrained() downloads the checkpoint (api/main.py's
torch_dtype=torch.float16) doesn't shrink the download itself - Render's
free-tier deploy (512MB RAM) was still hitting exit 137 mid-download of
the ~268MB fp32 safetensors file, before a single weight was even loaded.
Pushing an actually-fp16 (~134MB) safetensors file halves the bytes that
have to be downloaded and held in memory in the first place.

Requires being logged in first (see push_distilbert_to_hub.py's docstring
for the same reasoning: no token accepted as an argument here either).

Usage:

    python scripts/push_distilbert_fp16_to_hub.py <your-hf-username>/<repo-name>
"""

import sys
import tempfile
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "results" / "distilbert_emotion_model"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/push_distilbert_fp16_to_hub.py <your-hf-username>/<repo-name>")
        sys.exit(1)
    repo_id = sys.argv[1]

    if not MODEL_DIR.exists():
        print(f"No model found at {MODEL_DIR} - run src/train_text_distilbert.py first.")
        sys.exit(1)

    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model = model.half()

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        weights_file = Path(tmp_dir) / "model.safetensors"
        size_mb = weights_file.stat().st_size / (1024 * 1024)
        print(f"fp16 weights file: {size_mb:.1f}MB (fp32 original was ~255MB)")

        print(f"Pushing fp16 weights to https://huggingface.co/{repo_id} ...")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    print(f"\nDone. {repo_id} on the Hub is now fp16 - results/distilbert_emotion_model/ on disk is untouched (still fp32).")


if __name__ == "__main__":
    main()
