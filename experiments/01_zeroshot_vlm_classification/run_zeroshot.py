"""Zero-shot binary classification of BreastMNIST with a local vision LLM.

Asks the model one question per image — "is this malignant, yes or no" — and
records both the parsed answer and P(malignant), recovered from the first
token's logprobs so ROC/AUC can be computed from hard yes/no answers.

BreastMNIST is imbalanced (27% malignant), so raw accuracy is misleading: a
model that always answers "no" scores 0.731. Metrics are computed by
evaluate.py, which reports balanced accuracy as the headline figure.

Label convention (MedMNIST v2): 0 -> MALIGNANT, 1 -> BENIGN.
"yes" (is malignant) therefore maps to the positive class, MALIGNANT.

Writes results/{model}_{split}.jsonl incrementally and skips samples already
present, so a long run can be resumed after an interruption.

Usage:
  python run_zeroshot.py --model qwen2.5vl:7b --split test
  python run_zeroshot.py --model medgemma:latest --split test
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
RESULTS_DIR = Path(__file__).parent / "results"

# MedMNIST v2 BreastMNIST: 0 -> MALIGNANT, 1 -> BENIGN
_LABEL_MAP = {0: "MALIGNANT", 1: "BENIGN"}

PROMPT = (
    "This is a breast ultrasound image. Based on the visible features "
    "(shape, margin, orientation, echo pattern, posterior acoustic features), "
    "is the finding malignant?\n"
    "Answer with exactly one word: yes or no."
)


def _image_to_b64(img_array, size: int) -> str:
    """Encode a numpy image array as a base64 RGB JPEG at the given size."""
    from PIL import Image

    img = Image.fromarray(img_array.astype("uint8"))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _p_malignant(logprobs: list[dict] | None) -> float | None:
    """Recover P(malignant) from the first token's top_logprobs.

    Sums probability mass over yes-like and no-like tokens and normalises.
    Returns None when neither appears among the alternatives.
    """
    if not logprobs:
        return None
    top = logprobs[0].get("top_logprobs") or []
    p_yes = p_no = 0.0
    for cand in top:
        tok = cand.get("token", "").strip().lower().lstrip("*_ ")
        p = math.exp(cand.get("logprob", -100.0))
        if tok.startswith("yes"):
            p_yes += p
        elif tok.startswith("no"):
            p_no += p
    total = p_yes + p_no
    return p_yes / total if total > 0 else None


def _parse_answer(text: str) -> str | None:
    """Map free text to MALIGNANT / BENIGN, or None when unparseable."""
    t = text.strip().lower().lstrip("*_# ")
    if t.startswith("yes"):
        return "MALIGNANT"
    if t.startswith("no"):
        return "BENIGN"
    # Fall back to whichever word appears first anywhere in the response.
    iy, ino = t.find("yes"), t.find("no")
    if iy == -1 and ino == -1:
        return None
    if ino == -1 or (iy != -1 and iy < ino):
        return "MALIGNANT"
    return "BENIGN"


def query(model: str, image_b64: str, timeout: int, retries: int = 3) -> dict:
    """One chat call to Ollama, requesting logprobs. Retries on transient errors."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 5, "num_ctx": 2048},
            "logprobs": True,
            "top_logprobs": 10,
        }
    ).encode()

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, body, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_err = exc
            wait = 10 * attempt
            logger.warning("  request failed (%d/%d): %s — retrying in %ds",
                           attempt, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Ollama call failed after {retries} attempts: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot VLM classification of BreastMNIST")
    parser.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen2.5vl:7b")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=224,
                        help="MedMNIST+ source resolution (28/64/128/224)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N samples")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    from medmnist import BreastMNIST

    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    images, labels = ds.imgs, ds.labels
    n_total = len(images) if args.limit is None else min(args.limit, len(images))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace(":", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"{safe_model}_{args.split}.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["index"])
        logger.info("Resuming: %d samples already recorded in %s", len(done), out_path.name)

    logger.info("Model=%s split=%s size=%d | %d samples (%d remaining)",
                args.model, args.split, args.image_size, n_total, n_total - len(done))

    t_start = time.time()
    with out_path.open("a") as fh:
        for idx in range(n_total):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(labels[idx][0])]
            b64 = _image_to_b64(images[idx], args.image_size)

            t0 = time.time()
            resp = query(args.model, b64, args.timeout)
            elapsed = time.time() - t0

            raw = resp["message"]["content"]
            pred = _parse_answer(raw)
            lp = resp["message"].get("logprobs") or resp.get("logprobs")
            p_mal = _p_malignant(lp)

            fh.write(json.dumps({
                "index": idx,
                "gold": gold,
                "pred": pred,
                "p_malignant": p_mal,
                "raw": raw.strip(),
                "time_s": round(elapsed, 2),
            }) + "\n")
            fh.flush()

            done.add(idx)
            rate = (time.time() - t_start) / max(1, len(done))
            eta_min = rate * (n_total - len(done)) / 60
            logger.info("  [%3d/%3d] gold=%-9s pred=%-9s p_mal=%s %5.1fs | ETA %.0f min",
                        len(done), n_total, gold, pred,
                        f"{p_mal:.3f}" if p_mal is not None else "  n/a",
                        elapsed, eta_min)

    logger.info("Done in %.1f min -> %s", (time.time() - t_start) / 60, out_path)


if __name__ == "__main__":
    main()
