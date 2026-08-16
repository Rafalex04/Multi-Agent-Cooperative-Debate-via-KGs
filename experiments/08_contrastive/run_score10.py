"""Two-run 0-10 scoring: how benign does this look, how malignant does it look.

One call rates the image against described benign findings, another against
described malignant findings, and the higher score wins.

Differences from the previous graded attempt, which scored AUC 0.4994:

  ANCHORED SCALE     Every point on 0-10 is given a meaning in the prompt
                     (0 = no sign at all ... 10 = absolutely certain), so the
                     number means the same thing in both runs. The earlier
                     version labelled only four points and left the rest to
                     interpretation.
  BALANCED CONTEXTS  Both sides get the same number of findings, deduplicated
                     and picked by KG weight. The earlier run sent 9 malignant
                     findings (139 tokens) against 8 benign (109 tokens); a
                     length difference like that survives the subtraction as a
                     constant, which is exactly what a contrast is meant to
                     remove.
  SYMMETRIC QUESTION Each run asks about its own side ("how malignant", "how
                     benign") rather than always asking "is it malignant",
                     so neither run has the answer implied by its own framing.

Findings keep their visual descriptions ("spiculated margins, with sharp
angular lines radiating outward from the mass edge" rather than
"spiculated_margin"), since the automated bare-name variant lost 0.08 AUC.

Two scores are recorded per run: the integer the model emits, and the
expectation over the first-token digit distribution, which stays continuous
even when the emitted integer does not move.

Usage:
  python run_score10.py --split test --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, re, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

RESULTS_DIR = _HERE.parent / "results"
_LABEL_MAP  = {0: "MALIGNANT", 1: "BENIGN"}

_PROMPT = """\
You are an experienced radiologist examining a breast ultrasound image.

The following ACR BI-RADS findings indicate a {side} mass. Each is described by \
how it looks on ultrasound:
{rules}

Examine the image yourself. Do not assume any finding above is present — check \
each one against what you can actually see, and rely on your own radiological \
judgement.

On a scale of 0 to 10, how strongly does THIS image show {side} disease?

  0  = no sign of it whatsoever
  2  = very unlikely, essentially nothing suggests it
  4  = possible, but the signs are weak or equivocal
  6  = probable, several convincing signs are present
  8  = strong, clear and unambiguous signs are present
  10 = absolutely certain, unmistakable

Answer with the number only."""


def balanced_sides(triples, schema, n_per_side):
    """Equal-sized, deduplicated finding lists for each side.

    Selection uses only KG weight and description uniqueness — never labels —
    so this stays a property of the graph rather than of the answer.
    """
    from run_kg_featureprobe import build_probes
    seen, mal, ben = set(), [], []
    for p in sorted(build_probes(triples, schema), key=lambda x: -abs(x["weight"])):
        d = p["question"].splitlines()[1]
        d = d.replace("Does the mass in this image show ", "").rstrip("?")
        if d in seen:                      # e.g. spiculated / spiculated_margin
            continue
        seen.add(d)
        (mal if p["weight"] > 0 else ben).append(d)
    k = min(n_per_side, len(mal), len(ben))
    return mal[:k], ben[:k]


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image=None, num_predict=4):
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


def parse(out):
    """Emitted integer plus the first-token digit expectation.

    '10' spans two tokens, so the integer is read from the generated text while
    the expectation is taken over single-digit first tokens. They answer
    different questions: the integer is what the model committed to, the
    expectation retains gradient when the integer saturates.
    """
    txt = out["message"]["content"].strip()
    m = re.search(r"\d+", txt)
    val = min(int(m.group()), 10) if m else None

    lp = out.get("logprobs") or out["message"].get("logprobs")
    exp = ent = None
    if lp:
        probs = {}
        for c in lp[0].get("top_logprobs") or []:
            t = c.get("token", "").strip()
            if t.isdigit() and len(t) == 1:
                probs[int(t)] = probs.get(int(t), 0.0) + math.exp(c.get("logprob", -100.0))
        tot = sum(probs.values())
        if tot > 0:
            probs = {d: p / tot for d, p in probs.items()}
            exp = sum(d * p for d, p in probs.items())
            ent = -sum(p * math.log(p + 1e-12) for p in probs.values())
    return val, exp, ent, txt[:8]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="medgemma:4b")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--shard",      type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--n-findings", type=int, default=6,
                   help="findings per side; both sides get the same number")
    p.add_argument("--url",        default="http://localhost:11434/api/chat")
    p.add_argument("--output",     default=None)
    p.add_argument("--kg-root",    default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    mal, ben = balanced_sides(triples, schema, args.n_findings)

    mal_rules = "\n".join(f"- {d}" for d in mal)
    ben_rules = "\n".join(f"- {d}" for d in ben)
    logger.info("findings per side: malignant=%d benign=%d", len(mal), len(ben))
    for d in mal:
        logger.info("   [M] %s", d[:88])
    for d in ben:
        logger.info("   [B] %s", d[:88])

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    n = len(imgs) if args.limit is None else min(args.limit, len(imgs))
    idxs = [i for i in range(n) if i % args.num_shards == args.shard]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"score10_{args.split}_{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])

    client = Client(args.url, args.model)
    t0 = time.time()
    logger.info("score10 | split=%s shard=%d/%d | %d samples",
                args.split, args.shard, args.num_shards, len(idxs))

    with out_path.open("a") as fh:
        for k, idx in enumerate(idxs, 1):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(labels[idx][0])]
            b64 = _b64(imgs[idx], args.image_size)
            ts = time.time()

            vm, em, hm, rm = parse(client.call(
                _PROMPT.format(side="MALIGNANT", rules=mal_rules), b64))
            vb, eb, hb, rb = parse(client.call(
                _PROMPT.format(side="BENIGN", rules=ben_rules), b64))

            diff_int = None if vm is None or vb is None else vm - vb
            diff_exp = None if em is None or eb is None else em - eb
            winner = None if diff_int is None else (
                "MALIGNANT" if diff_int > 0 else "BENIGN" if diff_int < 0 else "TIE")

            fh.write(json.dumps({
                "index": idx, "gold": gold,
                "score_mal": vm, "score_ben": vb, "diff_int": diff_int,
                "exp_mal": em, "exp_ben": eb, "diff_exp": diff_exp,
                "entropy_mal": hm, "entropy_ben": hb,
                "winner": winner, "raw_mal": rm, "raw_ben": rb,
                "time_s": round(time.time() - ts, 1),
            }) + "\n")
            fh.flush()

            rate = (time.time() - t0) / k
            logger.info("  [%3d/%3d] %03d gold=%-9s M=%s B=%s diff=%s win=%-9s | ETA %.2fh",
                        k, len(idxs), idx, gold, vm, vb,
                        f"{diff_int:+d}" if diff_int is not None else " n/a",
                        winner, rate * (len(idxs) - k) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_path)


if __name__ == "__main__":
    main()
