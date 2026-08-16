"""Option 2: Binary feature questions → map to stance triples.

Step 1: Ask MedGemma YES/NO for each malignancy indicator (with image).
Step 2: Map YES answers to stance triples.
Step 3: Classify with image + only the matched stance triples.

Forces the model to commit to specific features rather than hedging.
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, re, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BREAST_SRC = Path(__file__).resolve().parents[2] / "breastMnist" / "src"
sys.path.insert(0, str(_BREAST_SRC))

OLLAMA_URL  = "http://localhost:11434/api/chat"
RESULTS_DIR = Path(__file__).parent / "results"
_LABEL_MAP  = {0: "MALIGNANT", 1: "BENIGN"}

# Feature questions mapped to their KG stance triple IDs
# Format: (question_text, triple_id, malignant=True/False)
_FEATURES = [
    ("Is the shape irregular (not oval or round)?",          "t_355", True),
    ("Are the margins spiculated?",                          "t_352", True),
    ("Are the margins irregular or angular (not circumscribed)?", None, True),
    ("Is the orientation non-parallel (taller than wide)?",  "t_361", True),
    ("Is there posterior acoustic shadowing?",               "t_358", True),
    ("Is there architectural distortion?",                   "t_356", True),
    ("Is the shape oval?",                                   "t_368", False),
    ("Are the margins circumscribed (smooth, well-defined)?","t_367", False),
    ("Is the orientation parallel (wider than tall)?",       "t_369", False),
]

FEATURE_PROMPT = (
    "Examine this breast ultrasound image carefully.\n"
    "Answer each question with YES or NO only.\n\n"
    "{questions}\n\n"
    "Reply with one answer per line in the same order."
)

CLASSIFY_PROMPT = (
    "You are examining a breast ultrasound image.\n\n"
    "Features you identified in this image:\n"
    "{features_found}\n\n"
    "ACR BI-RADS rules for these features:\n"
    "{kg_block}\n\n"
    "Based on the features you identified and the rules above, "
    "is this mass malignant or benign?\n"
    "Answer with exactly one word: malignant or benign."
)

CLASSIFY_NO_MATCH_PROMPT = (
    "You are examining a breast ultrasound image.\n\n"
    "Examine the image carefully. "
    "Is this breast mass malignant or benign?\n"
    "Answer with exactly one word: malignant or benign."
)


class _Shim:
    def __init__(self, model, timeout=1800):
        self.model = model
        self.timeout = timeout

    def call(self, prompt, image=None, num_predict=200, logprobs=False):
        msg = {"role": "user", "content": prompt}
        if image: msg["images"] = [image]
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096,
                            "num_predict": num_predict}}
        if logprobs:
            body["logprobs"] = True; body["top_logprobs"] = 10
        req = urllib.request.Request(
            OLLAMA_URL, json.dumps(body).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


def _b64(arr, size):
    from PIL import Image
    img = Image.fromarray(arr.astype("uint8"))
    if img.mode != "RGB": img = img.convert("RGB")
    if img.size != (size, size): img = img.resize((size, size))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _p_mal(logprobs):
    if not logprobs: return None
    top = logprobs[0].get("top_logprobs") or []
    pm = pb = 0.0
    for c in top:
        tok = c.get("token","").strip().lower().lstrip("*_ ")
        p   = math.exp(c.get("logprob", -100.0))
        if tok.startswith("malign"): pm += p
        elif tok.startswith("benign") or tok.startswith("ben"): pb += p
    return pm / (pm + pb) if (pm + pb) > 0 else None


def _pred(text):
    t = text.strip().lower().lstrip("*_# ")
    if t.startswith("malign"): return "MALIGNANT"
    if t.startswith("benign"): return "BENIGN"
    im, ib = t.find("malign"), t.find("benign")
    if im == -1 and ib == -1: return None
    return "MALIGNANT" if (ib == -1 or (im != -1 and im < ib)) else "BENIGN"


def _parse_yes_no(text: str, n_questions: int) -> list[bool]:
    """Extract YES/NO answers from model response."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    answers = []
    for line in lines:
        clean = re.sub(r"^\d+[.)]\s*", "", line).strip().upper()
        if clean.startswith("YES"):
            answers.append(True)
        elif clean.startswith("NO"):
            answers.append(False)
    # Pad/trim to expected length
    while len(answers) < n_questions:
        answers.append(False)
    return answers[:n_questions]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="medgemma:4b")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--timeout",    type=int, default=1800)
    p.add_argument("--output",     default=None)
    p.add_argument("--kg-root",
        default=str(Path(__file__).resolve().parents[2] / "breastMnist"))
    args = p.parse_args()

    from debate_kg.retriever.kg_retrieval import verbalise
    from medmnist import BreastMNIST

    root    = Path(args.kg_root)
    triples = json.loads((root/"data/breast/knowledge_graph.json").read_text())["triples"]
    by_id   = {t["id"]: t for t in triples}

    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    n  = len(ds.imgs) if args.limit is None else min(args.limit, len(ds.imgs))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"{args.model.replace(':','_')}_{args.split}_kg_binary.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip(): done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d done", len(done))

    client  = _Shim(args.model, args.timeout)
    questions_text = "\n".join(
        f"{i+1}. {q}" for i, (q, _, _) in enumerate(_FEATURES))
    feature_prompt = FEATURE_PROMPT.format(questions=questions_text)

    t_start = time.time()
    logger.info("Option 2: binary features | model=%s | %d samples", args.model, n)

    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done: continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64  = _b64(ds.imgs[idx], args.image_size)

            t0 = time.time()

            # Step 1: binary feature questions (with image)
            feat_out  = client.call(feature_prompt, image=b64, num_predict=50)
            feat_text = feat_out["message"]["content"]
            answers   = _parse_yes_no(feat_text, len(_FEATURES))

            # Step 2: map YES answers to stance triples
            matched_triples = []
            features_found  = []
            for (q, tid, is_mal), yes in zip(_FEATURES, answers):
                if yes:
                    features_found.append(f"- {q.rstrip('?')} → YES")
                    if tid and tid in by_id:
                        matched_triples.append(by_id[tid])

            # Step 3: classify with image + matched rules
            if matched_triples:
                kg_block = "\n".join(f"- {verbalise(t)}" for t in matched_triples)
                clf_prompt = CLASSIFY_PROMPT.format(
                    features_found="\n".join(features_found),
                    kg_block=kg_block)
            else:
                clf_prompt = CLASSIFY_NO_MATCH_PROMPT

            clf_out = client.call(clf_prompt, image=b64, num_predict=5, logprobs=True)
            elapsed = time.time() - t0

            raw  = clf_out["message"]["content"]
            lp   = clf_out.get("logprobs") or clf_out["message"].get("logprobs")
            pred = _pred(raw)
            p_m  = _p_mal(lp)

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_m,
                "raw": raw, "features_yes": sum(answers),
                "matched_triples": len(matched_triples),
                "feature_answers": feat_text[:200],
                "time_s": round(elapsed, 1),
            }) + "\n")
            fh.flush(); done.add(idx)

            rate = (time.time() - t_start) / max(1, len(done))
            logger.info(
                "  [%3d/%3d] gold=%-9s pred=%-9s p=%s | %d yes / %d matched | %.0fs | ETA %.1fh",
                len(done), n, gold, pred,
                f"{p_m:.3f}" if p_m is not None else " n/a",
                sum(answers), len(matched_triples), elapsed,
                rate * (n - len(done)) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t_start)/60, out_path)


if __name__ == "__main__":
    main()
