"""KG-grounded single-agent classification of BreastMNIST.

Three steps per sample, no debate:

  1. OBSERVE   the VLM describes the image against the KG-derived descriptor
               schema. Stance-free: no diagnostic vocabulary, no stance-bearing
               triples in context.
  2. RETRIEVE  those observations condition KG retrieval (anchor match, 1-hop
               graph expansion, dense similarity), yielding triples specific to
               this image rather than a fixed set.
  3. CLASSIFY  the VLM sees the image again, now with the retrieved triples, and
               answers malignant yes/no. P(malignant) comes from first-token
               logprobs so AUC is computable.

Compare against experiments/01 (same model, same images, no KG) to isolate what
the knowledge graph contributes.

Label convention (MedMNIST v2): 0 -> MALIGNANT, 1 -> BENIGN. MALIGNANT positive.

Usage:
  python run_kg_grounded.py --model medgemma:latest --split test --limit 2
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BREAST_SRC = Path(__file__).resolve().parents[2] / "breastMnist" / "src"
sys.path.insert(0, str(_BREAST_SRC))

OLLAMA_URL = "http://localhost:11434/api/chat"
RESULTS_DIR = Path(__file__).parent / "results"
_LABEL_MAP = {0: "MALIGNANT", 1: "BENIGN"}

CLASSIFY_PROMPT = (
    "This is a breast ultrasound image.\n\n"
    "Your own structured observation of it:\n{observation}\n\n"
    "Relevant ACR BI-RADS knowledge:\n{kg_block}\n\n"
    "Weigh the knowledge above against what you observe in the image. "
    "Is the finding malignant?\n"
    "Answer with exactly one word: yes or no."
)


class _OllamaShim:
    """Minimal client exposing .complete(), matching what observation.observe expects."""

    def __init__(self, model: str, timeout: int = 1800):
        self.model = model
        self.timeout = timeout
        self.calls = 0
        self.total_s = 0.0

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0,
                 image: str | None = None) -> str:
        msg: dict = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [image]
        messages = ([{"role": "system", "content": system}] if system else []) + [msg]
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": 400},
        }).encode()
        t0 = time.time()
        req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        self.calls += 1
        self.total_s += time.time() - t0
        return out["message"]["content"]


def _image_to_b64(arr, size: int) -> str:
    from PIL import Image

    img = Image.fromarray(arr.astype("uint8"))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _p_malignant(logprobs) -> float | None:
    """P(malignant) from the first token's yes/no probability mass."""
    if not logprobs:
        return None
    top = logprobs[0].get("top_logprobs") or []
    p_yes = p_no = 0.0
    for c in top:
        tok = c.get("token", "").strip().lower().lstrip("*_ ")
        p = math.exp(c.get("logprob", -100.0))
        if tok.startswith("yes"):
            p_yes += p
        elif tok.startswith("no"):
            p_no += p
    return p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else None


def _parse_yes_no(text: str) -> str | None:
    t = text.strip().lower().lstrip("*_# ")
    if t.startswith("yes"):
        return "MALIGNANT"
    if t.startswith("no"):
        return "BENIGN"
    iy, ino = t.find("yes"), t.find("no")
    if iy == -1 and ino == -1:
        return None
    return "MALIGNANT" if (ino == -1 or (iy != -1 and iy < ino)) else "BENIGN"


def classify(model: str, image_b64: str, observation_text: str, kg_block: str,
             timeout: int) -> tuple[str, float | None, str, float]:
    """Final yes/no call with image + retrieved triples. Returns (pred, p, raw, seconds)."""
    prompt = CLASSIFY_PROMPT.format(observation=observation_text, kg_block=kg_block)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 5, "num_ctx": 4096},
        "logprobs": True, "top_logprobs": 10,
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    elapsed = time.time() - t0
    raw = out["message"]["content"]
    lp = out["message"].get("logprobs") or out.get("logprobs")
    return _parse_yes_no(raw), _p_malignant(lp), raw.strip(), elapsed


def main() -> None:
    p = argparse.ArgumentParser(description="KG-grounded VLM classification")
    p.add_argument("--model", default="medgemma:latest")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--kg-root", default=str(Path(__file__).resolve().parents[2] / "breastMnist"))
    args = p.parse_args()

    from debate_kg.debate.observation import load_schema, observe
    from debate_kg.retriever.kg_retrieval import DenseRanker, retrieve, verbalise
    from medmnist import BreastMNIST

    root = Path(args.kg_root)
    schema = load_schema(root / "data/breast/schema.json")
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    stance = set(schema.get("stance_triple_ids", []))
    ranker = DenseRanker(triples, "all-MiniLM-L6-v2")

    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    n = len(ds.imgs) if args.limit is None else min(args.limit, len(ds.imgs))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace(":", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"{safe}_{args.split}_kg.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d already done", len(done))

    client = _OllamaShim(args.model, args.timeout)
    logger.info("KG-grounded run | model=%s | %d samples", args.model, n)

    t_start = time.time()
    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64 = _image_to_b64(ds.imgs[idx], args.image_size)
            sid = f"{idx:03d}"

            t0 = time.time()
            obs = observe(sid, b64, schema, client, max_attempts=2)
            t_obs = time.time() - t0

            res = retrieve(obs, triples, schema, stance_ids=stance,
                           config={"kg_token_budget": 200}, dense_ranker=ranker)
            kg_block = "\n".join(f"- {verbalise(t)}" for t in res.triples) or "(none retrieved)"
            obs_text = "; ".join(
                f"{c.replace('_', ' ')}: {v}" for c, v in obs.values.items() if v != "uncertain"
            ) or "(no features assessable)"

            pred, p_mal, raw, t_cls = classify(args.model, b64, obs_text, kg_block, args.timeout)

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_mal,
                "raw": raw, "n_triples": len(res.triples), "kg_tokens": res.token_estimate,
                "observation": obs.to_dict(),
                "time_observe_s": round(t_obs, 1), "time_classify_s": round(t_cls, 1),
                "time_s": round(t_obs + t_cls, 1),
            }) + "\n")
            fh.flush()
            done.add(idx)

            rate = (time.time() - t_start) / max(1, len(done))
            logger.info(
                "  [%3d/%3d] gold=%-9s pred=%-9s p=%s | %d triples | obs %.0fs + cls %.0fs = %.0fs"
                " | ETA %.1f h",
                len(done), n, gold, pred,
                f"{p_mal:.3f}" if p_mal is not None else " n/a",
                len(res.triples), t_obs, t_cls, t_obs + t_cls,
                rate * (n - len(done)) / 3600,
            )

    logger.info("Done in %.1f min -> %s", (time.time() - t_start) / 60, out_path)


if __name__ == "__main__":
    main()
