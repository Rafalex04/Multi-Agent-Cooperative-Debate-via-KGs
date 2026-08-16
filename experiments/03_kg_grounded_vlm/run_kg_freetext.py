"""Free-text KG-grounded classification of BreastMNIST.

Three steps, no debate:

  1. OBSERVE   MedGemma describes the image in free text (shape, margins,
               echo pattern, etc.) — no structured schema.
  2. RETRIEVE  The free-text description is embedded and scored against all
               KG triples via dense cosine similarity. Stance triples are
               sorted first within the token budget; is_a taxonomy triples
               are filtered.
  3. CLASSIFY  MedGemma receives its own prior description + retrieved KG
               triples (NO image) and predicts malignant / benign.

Label convention (MedMNIST v2): 0 -> MALIGNANT, 1 -> BENIGN.
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

_KG_TOKEN_BUDGET = 350
_TAXONOMY_RELS = {"is_a"}

OBSERVE_PROMPT = (
    "You are examining a breast ultrasound image.\n"
    "Describe the characteristics of the mass you see:\n"
    "- Shape (e.g. oval, round, irregular)\n"
    "- Margins (e.g. circumscribed, microlobulated, spiculated, angular, indistinct)\n"
    "- Echo pattern (e.g. anechoic, hypoechoic, isoechoic, hyperechoic, heterogeneous)\n"
    "- Internal content (e.g. solid, cystic, mixed solid and cystic)\n"
    "- Any posterior acoustic features\n"
    "Be concise and specific. Use standard radiology terminology."
)

CLASSIFY_PROMPT = (
    "Your prior observation of this breast ultrasound image:\n"
    "{observation}\n\n"
    "Relevant ACR BI-RADS diagnostic knowledge:\n"
    "{kg_block}\n\n"
    "Based on your observation and the radiological knowledge above, "
    "is this breast mass malignant or benign?\n"
    "Answer with exactly one word: malignant or benign."
)


class _OllamaShim:
    def __init__(self, model: str, timeout: int = 1800):
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str, image: str | None = None,
                 num_predict: int = 400) -> str:
        msg: dict = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [image]
        body = json.dumps({
            "model": self.model,
            "messages": [msg],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]


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
    if not logprobs:
        return None
    top = logprobs[0].get("top_logprobs") or []
    p_m = p_b = 0.0
    for c in top:
        tok = c.get("token", "").strip().lower().lstrip("*_ ")
        p = math.exp(c.get("logprob", -100.0))
        if tok.startswith("malign") or tok.startswith("mal"):
            p_m += p
        elif tok.startswith("benign") or tok.startswith("ben"):
            p_b += p
    return p_m / (p_m + p_b) if (p_m + p_b) > 0 else None


def _parse_mal_ben(text: str) -> str | None:
    t = text.strip().lower().lstrip("*_# ")
    if t.startswith("malign"):
        return "MALIGNANT"
    if t.startswith("benign"):
        return "BENIGN"
    im, ib = t.find("malign"), t.find("benign")
    if im == -1 and ib == -1:
        return None
    return "MALIGNANT" if (ib == -1 or (im != -1 and im < ib)) else "BENIGN"


def _retrieve(free_text: str, triples: list, stance_ids: set,
              ranker) -> tuple[list, int]:
    """Dense retrieval from free-text query, stance-first, taxonomy filtered."""
    from debate_kg.retriever.kg_retrieval import verbalise, _tid, _fields, _token_len

    scores = ranker.scores(free_text)  # {triple_id: float}

    # Sort all triples: stance first, then by descending cosine score
    def sort_key(t):
        is_stance = 0 if _tid(t) in stance_ids else 1
        return (is_stance, -scores.get(_tid(t), 0.0))

    ranked = sorted(triples, key=sort_key)

    # Fill token budget, skip taxonomy
    selected = []
    tokens = 0
    for t in ranked:
        _, rel, _ = _fields(t)
        if rel in _TAXONOMY_RELS:
            continue
        cost = _token_len(verbalise(t)) + 6
        if tokens + cost > _KG_TOKEN_BUDGET:
            break
        selected.append(t)
        tokens += cost

    return selected, tokens


def classify_text_only(model: str, observation: str, kg_block: str,
                        timeout: int) -> tuple[str | None, float | None, str, float]:
    prompt = CLASSIFY_PROMPT.format(observation=observation, kg_block=kg_block)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    lp = out.get("logprobs") or out["message"].get("logprobs")
    return _parse_mal_ben(raw), _p_malignant(lp), raw.strip(), elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medgemma:4b")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--kg-root", default=str(Path(__file__).resolve().parents[2] / "breastMnist"))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    from debate_kg.retriever.kg_retrieval import DenseRanker, verbalise
    from medmnist import BreastMNIST

    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    stance  = set(schema.get("stance_triple_ids", []))
    ranker  = DenseRanker(triples, "all-MiniLM-L6-v2")

    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    n  = len(ds.imgs) if args.limit is None else min(args.limit, len(ds.imgs))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        safe = args.model.replace(":", "_").replace("/", "_")
        out_path = RESULTS_DIR / f"{safe}_{args.split}_kg_freetext.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d already done", len(done))

    client = _OllamaShim(args.model, args.timeout)
    logger.info("KG freetext run | model=%s | %d samples | out=%s", args.model, n, out_path.name)

    t_start = time.time()
    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64  = _image_to_b64(ds.imgs[idx], args.image_size)

            # Step 1: free-text observation WITH image
            t0 = time.time()
            obs_text = client.complete(OBSERVE_PROMPT, image=b64, num_predict=200)
            t_obs = time.time() - t0

            # Step 2: dense retrieval, stance-first, taxonomy filtered
            selected, tok_est = _retrieve(obs_text, triples, stance, ranker)
            kg_block = "\n".join(f"- {verbalise(t)}" for t in selected) or "(none retrieved)"
            n_stance = sum(1 for t in selected if t["id"] in stance)

            # Step 3: text-only classification (no image)
            pred, p_mal, raw, t_cls = classify_text_only(
                args.model, obs_text, kg_block, args.timeout)

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_mal,
                "raw": raw, "n_triples": len(selected), "n_stance": n_stance,
                "kg_tokens": tok_est, "observation": obs_text,
                "time_observe_s": round(t_obs, 1), "time_classify_s": round(t_cls, 1),
                "time_s": round(t_obs + t_cls, 1),
            }) + "\n")
            fh.flush()
            done.add(idx)

            rate = (time.time() - t_start) / max(1, len(done))
            logger.info(
                "  [%3d/%3d] gold=%-9s pred=%-9s p=%s | %d triples (%d stance) | "
                "obs %.0fs + cls %.0fs | ETA %.1fh",
                len(done), n, gold, pred,
                f"{p_mal:.3f}" if p_mal is not None else " n/a",
                len(selected), n_stance, t_obs, t_cls,
                rate * (n - len(done)) / 3600,
            )

    logger.info("Done in %.1f min -> %s", (time.time() - t_start) / 60, out_path)


if __name__ == "__main__":
    main()
