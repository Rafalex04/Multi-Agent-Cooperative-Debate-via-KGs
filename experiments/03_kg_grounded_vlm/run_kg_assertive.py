"""Option 3: Observation → retrieve → assertive classify (image retained).

Same observe + retrieve pipeline as v3, but the classify prompt explicitly
tells MedGemma that the KG rules override its default conservatism: if any
malignant feature is present, predict MALIGNANT.
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

OBSERVE_PROMPT = (
    "You are examining a breast ultrasound image.\n"
    "Describe the mass characteristics: shape, margins, echo pattern, "
    "internal content, orientation, and any posterior acoustic features.\n"
    "Be concise and specific. Use standard radiology terminology."
)

CLASSIFY_PROMPT = (
    "You are a radiologist examining a breast ultrasound image.\n\n"
    "Your earlier description of this image:\n{observation}\n\n"
    "ACR BI-RADS diagnostic rules (apply these strictly):\n{kg_block}\n\n"
    "IMPORTANT: These rules override conservative defaults. "
    "If the image shows ANY malignancy indicator above, predict MALIGNANT "
    "even if other features seem benign.\n\n"
    "Look at the image again. Apply the rules. "
    "Is this mass malignant or benign?\n"
    "Answer with exactly one word: malignant or benign."
)


class _Shim:
    def __init__(self, model, timeout=1800):
        self.model = model; self.timeout = timeout

    def call(self, prompt, image=None, num_predict=200, logprobs=False):
        msg = {"role": "user", "content": prompt}
        if image: msg["images"] = [image]
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096,
                            "num_predict": num_predict}}
        if logprobs: body["logprobs"] = True; body["top_logprobs"] = 10
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

    from debate_kg.debate.observation import load_schema, observe
    from debate_kg.retriever.kg_retrieval import DenseRanker, retrieve, verbalise
    from medmnist import BreastMNIST

    root   = Path(args.kg_root)
    schema = load_schema(root/"data/breast/schema.json")
    triples= json.loads((root/"data/breast/knowledge_graph.json").read_text())["triples"]
    stance = set(schema.get("stance_triple_ids", []))
    ranker = DenseRanker(triples, "all-MiniLM-L6-v2")

    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    n  = len(ds.imgs) if args.limit is None else min(args.limit, len(ds.imgs))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"{args.model.replace(':','_')}_{args.split}_kg_assertive.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip(): done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d done", len(done))

    class _ObsClient:
        """Adapter so observe() can use our shim."""
        model = args.model
        def complete(self, prompt, system="", temperature=0.0, image=None):
            msg = {"role": "user", "content": prompt}
            if image: msg["images"] = [image]
            messages = ([{"role":"system","content":system}] if system else []) + [msg]
            body = {"model": self.model, "messages": messages, "stream": False,
                    "options": {"temperature": temperature, "num_ctx": 4096,
                                "num_predict": 400}}
            req = urllib.request.Request(
                OLLAMA_URL, json.dumps(body).encode(),
                {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())["message"]["content"]

    obs_client = _ObsClient()
    shim       = _Shim(args.model, args.timeout)
    t_start    = time.time()
    logger.info("Option 3: assertive | model=%s | %d samples", args.model, n)

    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done: continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64  = _b64(ds.imgs[idx], args.image_size)
            sid  = f"{idx:03d}"

            t0 = time.time()

            # Step 1: structured observation (uses schema-driven observe())
            obs = observe(sid, b64, schema, obs_client, max_attempts=2)

            # Step 2: retrieve with all 3 fixes
            res = retrieve(obs, triples, schema, stance_ids=stance,
                           config={"kg_token_budget": 350, "max_stance_triples": 20,
                                   "taxonomy_relations": ["is_a"]},
                           dense_ranker=ranker)
            kg_block = "\n".join(f"- {verbalise(t)}" for t in res.triples) \
                       or "(none retrieved)"
            obs_text = "; ".join(
                f"{c.replace('_',' ')}: {v}"
                for c, v in obs.values.items() if v != "uncertain"
            ) or "(no features assessable)"

            # Step 3: assertive classify WITH image
            clf_out = shim.call(
                CLASSIFY_PROMPT.format(observation=obs_text, kg_block=kg_block),
                image=b64, num_predict=5, logprobs=True)
            elapsed = time.time() - t0

            raw  = clf_out["message"]["content"]
            lp   = clf_out.get("logprobs") or clf_out["message"].get("logprobs")
            pred = _pred(raw)
            p_m  = _p_mal(lp)
            n_stance = sum(1 for t in res.triples if t["id"] in stance)

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_m,
                "raw": raw, "n_triples": len(res.triples), "n_stance": n_stance,
                "observation": obs_text, "time_s": round(elapsed, 1),
            }) + "\n")
            fh.flush(); done.add(idx)

            rate = (time.time() - t_start) / max(1, len(done))
            logger.info(
                "  [%3d/%3d] gold=%-9s pred=%-9s p=%s | %d triples (%d stance) | %.0fs | ETA %.1fh",
                len(done), n, gold, pred,
                f"{p_m:.3f}" if p_m is not None else " n/a",
                len(res.triples), n_stance, elapsed,
                rate * (n - len(done)) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t_start)/60, out_path)


if __name__ == "__main__":
    main()
