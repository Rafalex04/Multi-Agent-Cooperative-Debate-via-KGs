"""Option 7: Principled retrieval pipeline — observe → dense retrieve → natural verbalize → classify.

Generalizable design:
  1. OBSERVE  Free-text description of what the model actually sees in the image
              (not constrained to schema options — avoids structured-observation corruption)
  2. RETRIEVE Dense similarity over ALL 370 KG triples, stance triples prioritized first,
              taxonomy (is_a/subtype_of) excluded, top-N by token budget
  3. VERBALIZE Natural-language sentence per triple (not raw subject/relation/object)
  4. CLASSIFY Image + verbalized triples → malignant/benign

Key improvement over all previous single-agent attempts:
  - No hand-picked triple categories: retrieval selects the relevant ones per image
  - Natural language verbalization: each triple becomes a clinical sentence
  - Image retained in classify step: model sees evidence alongside rules
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BREAST_SRC = Path(__file__).resolve().parents[2] / "breastMnist" / "src"
sys.path.insert(0, str(_BREAST_SRC))

OLLAMA_URL  = "http://localhost:11434/api/chat"
RESULTS_DIR = Path(__file__).parent / "results"
_LABEL_MAP  = {0: "MALIGNANT", 1: "BENIGN"}

# Relations to exclude from retrieval (taxonomy / structural, not diagnostic)
_EXCLUDE_RELS = {
    "is_a", "subtype_of", "birads_subcategory_of", "category_descriptor",
    "also_known_as", "often_located_in", "typical_size", "management",
    "definition", "likelihood_of_malignancy", "likelihood_of_malignancy_range",
    "likelihood_of_malignancy_minimum", "indicates",
}

# Stance relations — these are prioritized first in the token budget
_STANCE_RELS = {
    "suggests_malignancy", "suggests_benign", "present_in_benign",
    "combination_indicates", "typically_classified_as",
    "associated_with_malignancy_predictor", "supports_malignancy_diagnosis",
    "suggests_birads_5", "has_no_malignant_potential", "suggests_benign_simple_cyst",
    "mimics",
}

# Diagnostic roles for verbalization
_MAL_RELS  = {"suggests_malignancy", "supports_malignancy_diagnosis", "suggests_birads_5"}
_BEN_RELS  = {"suggests_benign", "present_in_benign", "has_no_malignant_potential",
              "suggests_benign_simple_cyst"}


def _h(s: str) -> str:
    """Snake_case → human readable."""
    return s.replace("_", " ").strip()


def verbalize(t: dict) -> str:
    """Convert a KG triple to a natural clinical sentence."""
    subj = _h(t.get("subject", ""))
    rel  = t.get("relation", "")
    obj  = _h(t.get("object", ""))

    # Remove trivial objects
    if obj in ("true", "false"):
        obj = ""

    if rel == "suggests_malignancy":
        return f"{subj} is a sign of malignancy{' (possible)' if t['object']=='possible' else ''}."
    if rel == "suggests_benign":
        return f"{subj} suggests the mass is benign."
    if rel == "present_in_benign":
        return f"{subj} is typically present in benign masses."
    if rel == "has_no_malignant_potential":
        return f"{subj} has no malignant potential."
    if rel == "suggests_benign_simple_cyst":
        return f"{subj} suggests a benign simple cyst."
    if rel == "supports_malignancy_diagnosis":
        return f"{subj} supports a diagnosis of malignancy."
    if rel == "suggests_birads_5":
        return f"{subj} is highly suggestive of malignancy (BI-RADS 5)."
    if rel == "ultrasound_appearance_includes":
        return f"{subj} appears on ultrasound as: {obj}."
    if rel == "combination_indicates":
        verdict = "benign" if any(k in obj for k in
                  ["fibroadenoma", "cyst", "lymph_node", "lipoma", "sebaceous"]) else obj
        return f"When you see {subj}, this indicates {verdict}."
    if rel == "typically_classified_as":
        birads_map = {"birads 2": "benign (BI-RADS 2)", "birads 3": "probably benign (BI-RADS 3)",
                      "birads 4": "suspicious (BI-RADS 4)", "birads 5": "highly suspicious (BI-RADS 5)"}
        birads = birads_map.get(obj, obj)
        return f"{subj} is typically classified as {birads}."
    if rel == "associated_with":
        return f"{subj} is associated with {obj}."
    if rel == "associated_with_malignancy_predictor":
        return f"In solid masses, {obj} is a predictor of malignancy."
    if rel == "mimics":
        return f"{subj} can mimic {obj} on imaging."
    if rel == "caused_by":
        return f"{subj} can be caused by {obj}."
    if rel == "differential_for":
        return f"{subj} is in the differential diagnosis for {obj}."
    if rel == "differs_from":
        return f"{subj} differs from {obj}."

    # Fallback: readable form
    return f"{subj} {rel.replace('_', ' ')} {obj}.".strip(" .")  + "."


OBSERVE_PROMPT = (
    "You are examining a breast ultrasound image.\n"
    "Describe what you see: the shape of the mass, its margins, echo pattern, "
    "orientation relative to the skin, and any posterior acoustic features.\n"
    "Be specific. Use radiology terminology. 2-4 sentences."
)

CLASSIFY_PROMPT = (
    "You are a radiologist examining a breast ultrasound image.\n\n"
    "Your prior description of this image:\n{observation}\n\n"
    "Relevant ACR BI-RADS knowledge for what you described:\n{kg_block}\n\n"
    "Look at the image again. Weigh what you see against the knowledge above.\n"
    "Is this breast mass malignant or benign?\n"
    "Answer with exactly one word: malignant or benign."
)


class _Shim:
    def __init__(self, model, timeout=1800):
        self.model = model
        self.timeout = timeout

    def call(self, prompt, image=None, num_predict=400, logprobs=False):
        msg = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [image]
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": num_predict}}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 10
        req = urllib.request.Request(
            OLLAMA_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


def _b64(arr, size):
    from PIL import Image
    img = Image.fromarray(arr.astype("uint8"))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _p_mal(logprobs):
    if not logprobs:
        return None
    top = logprobs[0].get("top_logprobs") or []
    pm = pb = 0.0
    for c in top:
        tok = c.get("token", "").strip().lower().lstrip("*_ ")
        p   = math.exp(c.get("logprob", -100.0))
        if tok.startswith("malign"):
            pm += p
        elif tok.startswith("benign") or tok.startswith("ben"):
            pb += p
    return pm / (pm + pb) if (pm + pb) > 0 else None


def _pred(text):
    t = text.strip().lower().lstrip("*_# ")
    if t.startswith("malign"):
        return "MALIGNANT"
    if t.startswith("benign"):
        return "BENIGN"
    im, ib = t.find("malign"), t.find("benign")
    if im == -1 and ib == -1:
        return None
    return "MALIGNANT" if (ib == -1 or (im != -1 and im < ib)) else "BENIGN"


def retrieve(obs_text: str, triples: list, stance_ids: set,
             ranker, n_triples: int = 18, token_budget: int = 400) -> list:
    """Dense retrieve top-N triples. Stance triples first, taxonomy excluded."""
    scores = ranker.scores(obs_text)  # {triple_id: float}

    def _tid(t):
        return t.get("id", "")

    def sort_key(t):
        if t["relation"] in _EXCLUDE_RELS:
            return (2, 0.0)  # excluded, sorted last
        is_stance = 0 if (_tid(t) in stance_ids or t["relation"] in _STANCE_RELS) else 1
        return (is_stance, -scores.get(_tid(t), 0.0))

    ranked = sorted(triples, key=sort_key)

    selected = []
    tokens = 0
    for t in ranked:
        if t["relation"] in _EXCLUDE_RELS:
            break  # all excluded triples are last
        v = verbalize(t)
        cost = len(v.split()) + 4  # rough token estimate
        if tokens + cost > token_budget:
            break
        selected.append((t, v))
        tokens += cost
        if len(selected) >= n_triples:
            break

    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="medgemma:4b")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--timeout",    type=int, default=1800)
    p.add_argument("--output",     default=None)
    p.add_argument("--n-triples",  type=int, default=18,
                   help="Max triples to include in context")
    p.add_argument("--token-budget", type=int, default=400,
                   help="Approximate token budget for KG context")
    p.add_argument("--kg-root",
        default=str(Path(__file__).resolve().parents[2] / "breastMnist"))
    args = p.parse_args()

    from debate_kg.retriever.kg_retrieval import DenseRanker
    from medmnist import BreastMNIST

    root    = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    stance  = set(schema.get("stance_triple_ids", []))
    ranker  = DenseRanker(triples, "all-MiniLM-L6-v2")

    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    n  = len(ds.imgs) if args.limit is None else min(args.limit, len(ds.imgs))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"{args.model.replace(':','_')}_{args.split}_kg_retrieval_v2.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d done", len(done))

    client  = _Shim(args.model, args.timeout)
    t_start = time.time()
    logger.info("Option 7: principled retrieval | model=%s | n_triples=%d | %d samples",
                args.model, args.n_triples, n)

    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64  = _b64(ds.imgs[idx], args.image_size)

            t0 = time.time()

            # Step 1: free-text observation WITH image
            obs_out  = client.call(OBSERVE_PROMPT, image=b64, num_predict=150)
            obs_text = obs_out["message"]["content"].strip()

            # Step 2: retrieve relevant triples by dense similarity
            retrieved = retrieve(obs_text, triples, stance, ranker,
                                 n_triples=args.n_triples,
                                 token_budget=args.token_budget)
            kg_block = "\n".join(f"- {v}" for _, v in retrieved) or "(no relevant triples found)"
            n_stance = sum(1 for t, _ in retrieved
                          if t["id"] in stance or t["relation"] in _STANCE_RELS)

            # Step 3: classify WITH image + retrieved triples
            clf_out = client.call(
                CLASSIFY_PROMPT.format(observation=obs_text, kg_block=kg_block),
                image=b64, num_predict=5, logprobs=True)
            elapsed = time.time() - t0

            raw  = clf_out["message"]["content"]
            lp   = clf_out.get("logprobs") or clf_out["message"].get("logprobs")
            pred = _pred(raw)
            p_m  = _p_mal(lp)

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_m,
                "raw": raw.strip(), "n_triples": len(retrieved), "n_stance": n_stance,
                "observation": obs_text, "time_s": round(elapsed, 1),
            }) + "\n")
            fh.flush()
            done.add(idx)

            rate = (time.time() - t_start) / max(1, len(done))
            logger.info(
                "  [%3d/%3d] gold=%-9s pred=%-9s p=%s | %d triples (%d stance) | %.0fs | ETA %.1fh",
                len(done), n, gold, pred,
                f"{p_m:.3f}" if p_m is not None else " n/a",
                len(retrieved), n_stance, elapsed,
                rate * (n - len(done)) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t_start) / 60, out_path)


if __name__ == "__main__":
    main()
