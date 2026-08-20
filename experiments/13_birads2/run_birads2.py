"""Two-turn BI-RADS assessment: one turn primed benign, one primed malignant.

The 0-10 predecessor (run_score10.py) asked each turn about its OWN side -- "how
malignant does this look" against malignant findings, "how benign" against
benign ones -- so the two numbers lived on opposite scales and had to be
SUBTRACTED. That made the result a contrast, and the contrast never separated
from the shared bias it was meant to cancel.

BI-RADS removes the need to subtract. The category always means one thing, the
probability of malignancy, no matter which findings preceded the question. So
both turns answer on the same scale and the two answers are AVERAGED. What
differs between them is only the priming: one turn sees the malignant findings
first, the other the benign ones. Averaging cancels the priming instead of
cancelling the scale.

The scale itself is not written here. birads_scale.py reads the categories,
their meanings and their probability bands out of the knowledge graph, so
converting a category to a number and an average back to a verdict are both
graph operations. That is the part a 0-10 scale cannot do: 7-out-of-10 has no
calibration, BI-RADS 4b is defined as 10-50%.

Findings keep their visual descriptions, and both sides get the same number of
them, for the reasons documented in run_score10.py.

Usage:
  python run_birads2.py --split test --shard 0 --num-shards 4
  python run_birads2.py --split train --shards 3,7 --num-shards 16
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "08_contrastive"))
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

from birads_scale import load_scale, menu, parse_category            # noqa: E402

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

Assign THIS image an ACR BI-RADS assessment category:

{menu}

Answer with the category only."""


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image=None, num_predict=6):
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


def parse(out, scale):
    """Category the model committed to, plus the first-token digit expectation.

    '4a' spans two tokens, so the category is read from the generated text. The
    expectation over single-digit first tokens is kept for the same reason as in
    run_score10: it stays continuous when the committed category quantises onto
    two or three rungs, which is the failure mode that made the emitted integer
    useless there.
    """
    txt = (out["message"]["content"] or "").strip()
    cat = parse_category(txt, scale)

    lp = out.get("logprobs") or out["message"].get("logprobs")
    exp = ent = None
    if lp:
        probs = {}
        for c in lp[0].get("top_logprobs") or []:
            t = c.get("token", "").strip()
            if t.isdigit() and len(t) == 1 and 1 <= int(t) <= 5:
                probs[int(t)] = probs.get(int(t), 0.0) + math.exp(c.get("logprob", -100.0))
        tot = sum(probs.values())
        if tot > 0:
            probs = {d: p / tot for d, p in probs.items()}
            exp = sum(d * p for d, p in probs.items())
            ent = -sum(p * math.log(p + 1e-12) for p in probs.values())
    return cat, exp, ent, txt[:12]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="qwen3-vl:8b-instruct")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--shard",      type=int, default=0)
    p.add_argument("--shards",     default=None,
                   help="comma-separated shard ids, for nodes taking more than one")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--n-findings", type=int, default=6,
                   help="findings per side; both sides get the same number")
    p.add_argument("--url",        default="http://localhost:11434/api/chat")
    p.add_argument("--output",     default=None)
    p.add_argument("--tag",        default="birads2")
    p.add_argument("--kg-root",    default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    from run_score10 import balanced_sides

    root = Path(args.kg_root)
    kg_path = root / "data/breast/knowledge_graph.json"
    triples = json.loads(kg_path.read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())

    scale = load_scale(kg_path)
    cat_menu = menu(scale)
    mal, ben = balanced_sides(triples, schema, args.n_findings)
    mal_rules = "\n".join(f"- {d}" for d in mal)
    ben_rules = "\n".join(f"- {d}" for d in ben)

    logger.info("ladder: %s", " ".join(f"{d['code']}={d['p_mal']:.1f}%" for d in scale))
    logger.info("findings per side: malignant=%d benign=%d", len(mal), len(ben))

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    n = len(imgs) if args.limit is None else min(args.limit, len(imgs))
    want = ([int(x) for x in args.shards.split(",")] if args.shards
            else [args.shard])
    idxs = [i for i in range(n) if i % args.num_shards in want]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"{args.tag}_{args.split}_{'-'.join(map(str, want))}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                try:
                    done.add(json.loads(ln)["index"])
                except Exception:
                    pass

    client = Client(args.url, args.model)
    t0, k = time.time(), 0
    logger.info("birads2 | split=%s shards=%s/%d | %d samples (%d already done)",
                args.split, want, args.num_shards, len(idxs), len(done))

    with out_path.open("a") as fh:
        for idx in idxs:
            if idx in done:
                continue
            k += 1
            gold = _LABEL_MAP[int(labels[idx][0])]
            b64 = _b64(imgs[idx], args.image_size)
            ts = time.time()

            cm, em, hm, rm = parse(client.call(
                _PROMPT.format(side="MALIGNANT", rules=mal_rules, menu=cat_menu), b64), scale)
            cb, eb, hb, rb = parse(client.call(
                _PROMPT.format(side="BENIGN", rules=ben_rules, menu=cat_menu), b64), scale)

            rec = {"index": idx, "gold": gold,
                   "cat_mal": cm["code"] if cm else None,
                   "cat_ben": cb["code"] if cb else None,
                   "p_mal_turn": cm["p_mal"] if cm else None,
                   "p_ben_turn": cb["p_mal"] if cb else None,
                   "ord_mal": cm["ordinal"] if cm else None,
                   "ord_ben": cb["ordinal"] if cb else None,
                   "exp_mal": em, "exp_ben": eb,
                   "entropy_mal": hm, "entropy_ben": hb,
                   "raw_mal": rm, "raw_ben": rb,
                   "time_s": round(time.time() - ts, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            rate = (time.time() - t0) / k
            avg = (None if cm is None or cb is None
                   else (cm["p_mal"] + cb["p_mal"]) / 2)
            logger.info("  [%3d/%3d] %03d gold=%-9s M=%-3s B=%-3s avg_p=%s | ETA %.2fh",
                        k, len(idxs), idx, gold,
                        rec["cat_mal"] or "?", rec["cat_ben"] or "?",
                        f"{avg:5.1f}%" if avg is not None else "  n/a",
                        rate * (len(idxs) - k) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_path)


if __name__ == "__main__":
    main()
