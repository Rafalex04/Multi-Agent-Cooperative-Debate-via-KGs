"""Two-run contrast with a graded evidence score instead of a binary answer.

WHY THIS VARIANT
----------------
The binary version of this experiment failed for a specific, measurable reason:
listing 14 findings and asking one yes/no question pinned the logprobs to the
extremes, identically for every image.

  verdict | malignant rules   mean(mal) +0.999   mean(ben) +0.999
  verdict | benign rules      mean(mal) +0.000   mean(ben) +0.000

With no image-dependent variation left underneath, the subtraction had nothing
to preserve (best AUC 0.5587 against a 0.6013 baseline).

Two changes here:

1. GRADED OUTPUT. Ask "how much evidence do you see, 0-9" and take the expected
   value over the digit logprobs, E[d] = sum(d * p(d)). A ten-way distribution
   cannot collapse the way a two-way one does, and the expectation is continuous
   even when the argmax is not. This is the same fix that made the per-feature
   probes work, applied at list level.

2. THE KG IS DEMOTED TO A REFERENCE. Earlier prompts opened with "these findings
   indicate a MALIGNANT mass", which states the conclusion before the model has
   looked and is most likely what pinned the answer. Here the image and the
   model's own radiological knowledge are named as primary and the triples are
   explicitly secondary, present only as a checklist to keep in mind.

score = E[evidence | malignant findings] - E[evidence | benign findings]

Saturation is measured rather than assumed: the per-run digit entropy and the
spread of E[d] are both recorded, so if this collapses too it is visible in the
output instead of having to be inferred.

Usage:
  python run_contrastive_scored.py --split test --shard 0 --num-shards 4
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

_PROMPT = """\
You are an experienced radiologist examining a breast ultrasound image.

Judge this image using your own radiological expertise and what you can \
actually see. That is the primary basis for your answer.

As a secondary reference only, these ACR BI-RADS findings are associated with a \
{side} mass. Keep them in mind as a checklist, but do not let them override your \
own reading of the image, and do not assume any of them are present:
{rules}

Considering everything you see, how much evidence of {side} disease is present \
in this image?

Answer with a single digit from 0 to 9:
  0 = no evidence whatsoever
  3 = weak or equivocal evidence
  6 = moderate, fairly convincing evidence
  9 = overwhelming, unmistakable evidence

Answer with exactly one digit and nothing else."""


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
                "logprobs": True, "top_logprobs": 20}
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


def expected_digit(lp):
    """E[d] over the digit distribution, plus its entropy.

    The expectation is what makes this continuous: even when the argmax digit is
    the same for two images, the probability mass behind it usually is not.
    Entropy is returned so saturation can be detected rather than guessed at.
    """
    if not lp:
        return None, None, None
    probs = {}
    for c in lp[0].get("top_logprobs") or []:
        tok = c.get("token", "").strip()
        if tok.isdigit() and len(tok) == 1:
            probs[int(tok)] = probs.get(int(tok), 0.0) + math.exp(c.get("logprob", -100.0))
    tot = sum(probs.values())
    if tot <= 0:
        return None, None, None
    probs = {d: p / tot for d, p in probs.items()}
    exp = sum(d * p for d, p in probs.items())
    ent = -sum(p * math.log(p + 1e-12) for p in probs.values())
    return exp, ent, {str(d): round(p, 4) for d, p in sorted(probs.items())}


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
    from run_contrastive import stance_sets

    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    mal, ben = stance_sets(triples, schema)
    mal_rules = "\n".join(f"- {d}" for d in mal)
    ben_rules = "\n".join(f"- {d}" for d in ben)
    logger.info("malignant findings=%d  benign findings=%d", len(mal), len(ben))

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    n = len(imgs) if args.limit is None else min(args.limit, len(imgs))
    idxs = [i for i in range(n) if i % args.num_shards == args.shard]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"scored_{args.split}_{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])

    client = Client(args.url, args.model)
    t0 = time.time()
    logger.info("scored contrast | split=%s shard=%d/%d | %d samples",
                args.split, args.shard, args.num_shards, len(idxs))

    with out_path.open("a") as fh:
        for k, idx in enumerate(idxs, 1):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(labels[idx][0])]
            b64 = _b64(imgs[idx], args.image_size)
            ts = time.time()

            om = client.call(_PROMPT.format(side="MALIGNANT", rules=mal_rules), b64)
            ob = client.call(_PROMPT.format(side="BENIGN",    rules=ben_rules), b64)
            em, hm, dm = expected_digit(om.get("logprobs") or om["message"].get("logprobs"))
            eb, hb, db = expected_digit(ob.get("logprobs") or ob["message"].get("logprobs"))

            score = None if em is None or eb is None else em - eb
            fh.write(json.dumps({
                "index": idx, "gold": gold,
                "e_mal": em, "e_ben": eb, "score": score,
                "entropy_mal": hm, "entropy_ben": hb,
                "dist_mal": dm, "dist_ben": db,
                "raw_mal": om["message"]["content"].strip()[:8],
                "raw_ben": ob["message"]["content"].strip()[:8],
                "time_s": round(time.time() - ts, 1),
            }) + "\n")
            fh.flush()

            rate = (time.time() - t0) / k
            logger.info("  [%3d/%3d] %03d gold=%-9s Emal=%s Eben=%s d=%s H=%.2f/%.2f | ETA %.2fh",
                        k, len(idxs), idx, gold,
                        f"{em:.2f}" if em is not None else " n/a",
                        f"{eb:.2f}" if eb is not None else " n/a",
                        f"{score:+.2f}" if score is not None else "  n/a",
                        hm or 0, hb or 0, rate * (len(idxs) - k) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_path)


if __name__ == "__main__":
    main()
