"""Two-run contrastive scoring: benign context minus malignant context.

MOTIVATION
----------
Ten different ways of putting the KG in the prompt all lost to using no KG at
all (best 0.5825 against a 0.6013 baseline), and they failed in two directions:

  over-trigger    kg_binary answered MALIGNANT on all 156 samples
  over-suppress   kg_grounded/assertive/enriched collapsed to TP 2-5 of 42

Both are a constant offset applied to every image: whichever rule set is in
context drags the answer toward that rule set regardless of what is shown. A
single run cannot separate that offset from real evidence.

Running the image twice and subtracting can. The malignant-rule context and the
benign-rule context each carry their own bias, but the bias is roughly the same
for every image, whereas the *shift* between them depends on what is actually
visible. Subtracting cancels the shared component and keeps the image-driven
part -- the same reason contrastive decoding and paired prompting work.

Each prompt is also small: 14 malignant stance triples or 8 benign ones, roughly
100 tokens, rather than the ~400-token dumps that drowned out the 256 image
tokens in earlier attempts.

TWO SCORING VARIANTS
--------------------
  verdict  Ask "is this malignant?" under each context.
           score = p(malignant | malignant rules) - p(malignant | benign rules)

  match    Ask "do these features describe this image?" under each context.
           score = p(yes | malignant rules) - p(yes | benign rules)
           A likelihood-ratio reading: which rule set fits the image better.

Both are computed in the same pass, so one run reports both.

If this works, the same asymmetry can be applied to the debate: give the agent
arguing MALIGNANT only the benign triples and vice versa, forcing each side to
engage with the evidence against it instead of cherry-picking support.

Usage:
  python run_contrastive.py --split test --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

RESULTS_DIR = _HERE.parent / "results"
_LABEL_MAP  = {0: "MALIGNANT", 1: "BENIGN"}


def stance_sets(triples: list[dict], schema: dict) -> tuple[list[str], list[str]]:
    """Split KG stance features into malignant-side and benign-side wordings.

    Reuses the probe phrasings so the two contexts are described at the same
    level of visual detail; an asymmetry there would show up as a constant in
    the difference, which is exactly what this design is trying to remove.
    """
    from run_kg_featureprobe import build_probes
    mal, ben = [], []
    for p in build_probes(triples, schema):
        desc = p["question"].splitlines()[1]
        desc = desc.replace("Does the mass in this image show ", "").rstrip("?")
        (mal if p["weight"] > 0 else ben).append(desc)
    return mal, ben


_VERDICT_PROMPT = """\
This is a breast ultrasound image.

According to ACR BI-RADS, these findings indicate a {side} mass:
{rules}

Look at the image. Is the finding malignant?
Answer with exactly one word: yes or no."""

_MATCH_PROMPT = """\
This is a breast ultrasound image.

According to ACR BI-RADS, these findings indicate a {side} mass:
{rules}

Look at the image. Do the findings listed above describe what you see?
Answer with exactly one word: yes or no."""


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image=None, num_predict=3):
        msg = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [image]
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": {"temperature": 0, "num_ctx": 2048,
                            "num_predict": num_predict},
                "logprobs": True, "top_logprobs": 10}
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
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


def _p_yes(lp):
    if not lp:
        return None
    py = pn = 0.0
    for c in lp[0].get("top_logprobs") or []:
        tok = c.get("token", "").strip().lower().lstrip("*_# ")
        p = math.exp(c.get("logprob", -100.0))
        if tok.startswith("yes"):
            py += p
        elif tok.startswith("no"):
            pn += p
    return py / (py + pn) if (py + pn) > 0 else None


def ask(client, prompt, b64):
    out = client.call(prompt, image=b64)
    lp = out.get("logprobs") or out["message"].get("logprobs")
    return _p_yes(lp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="medgemma:4b")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--shard",      type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--url",        default="http://localhost:11434/api/chat")
    p.add_argument("--output",     default=None)
    p.add_argument("--kg-root",    default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    mal, ben = stance_sets(triples, schema)

    mal_rules = "\n".join(f"- {d}" for d in mal)
    ben_rules = "\n".join(f"- {d}" for d in ben)
    logger.info("malignant context: %d findings | benign context: %d findings",
                len(mal), len(ben))

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    n = len(imgs) if args.limit is None else min(args.limit, len(imgs))
    idxs = [i for i in range(n) if i % args.num_shards == args.shard]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"contrastive_{args.split}_{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])

    client = Client(args.url, args.model)
    t0 = time.time()
    logger.info("contrastive | split=%s shard=%d/%d | %d samples",
                args.split, args.shard, args.num_shards, len(idxs))

    with out_path.open("a") as fh:
        for k, idx in enumerate(idxs, 1):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(labels[idx][0])]
            b64 = _b64(imgs[idx], args.image_size)
            ts = time.time()

            v_mal = ask(client, _VERDICT_PROMPT.format(side="MALIGNANT", rules=mal_rules), b64)
            v_ben = ask(client, _VERDICT_PROMPT.format(side="BENIGN",    rules=ben_rules), b64)
            m_mal = ask(client, _MATCH_PROMPT.format(side="MALIGNANT",   rules=mal_rules), b64)
            m_ben = ask(client, _MATCH_PROMPT.format(side="BENIGN",      rules=ben_rules), b64)

            def diff(a, b):
                return None if a is None or b is None else a - b

            fh.write(json.dumps({
                "index": idx, "gold": gold,
                "verdict_mal_ctx": v_mal, "verdict_ben_ctx": v_ben,
                "match_mal_ctx": m_mal,   "match_ben_ctx": m_ben,
                "score_verdict": diff(v_mal, v_ben),
                "score_match":   diff(m_mal, m_ben),
                "time_s": round(time.time() - ts, 1),
            }) + "\n")
            fh.flush()
            rate = (time.time() - t0) / k
            logger.info("  [%3d/%3d] %03d gold=%-9s dV=%s dM=%s | %.0fs | ETA %.2fh",
                        k, len(idxs), idx, gold,
                        f"{diff(v_mal,v_ben):+.3f}" if v_mal and v_ben else "  n/a",
                        f"{diff(m_mal,m_ben):+.3f}" if m_mal and m_ben else "  n/a",
                        time.time() - ts, rate * (len(idxs) - k) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_path)


if __name__ == "__main__":
    main()
