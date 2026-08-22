"""Per-finding visual probes with the good backbone, as KG node measurements.

The probe vectors sitting in dataset_v2 were produced by the weak backbone and
are noise: per-feature test AUCs run 0.35 to 0.56, i.e. nothing. But the BI-RADS
run showed that this backbone's first-token logprob distribution carries real
signal even when its committed answer does not (0.7598 AUC from two calls), and
a probe is exactly that shape of question.

So re-measure all F findings with qwen3-vl:8b-instruct, one yes/no call each,
reading p(yes) off the first-token distribution. The result is an F-dimensional
measured vector per image, which is what a KG finding node needs as a feature:
the debate says how the finding was ARGUED, the probe says whether it is THERE.

Descriptions come from the same all_findings() the debate kit uses, so the probe
and the debate refer to identical findings and can share a node.

Usage:
  python run_probes.py --split test --num-shards 16 --shards 0,1
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "10_debate_v5"))
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

RESULTS = _HERE.parent / "results"
_LABEL = {0: "MALIGNANT", 1: "BENIGN"}

# ---------------------------------------------------------------- phrasings --
# A 2x2 factorial, not four hand-written prompts. One axis is POLARITY (does the
# finding hold / does it fail to hold); the other is FRAMING (report what you see
# yourself / adjudicate a claim someone else made). Both axes are properties of
# how a question is asked, not of breast imaging, so the same four templates
# transfer to any ontology that supplies finding descriptions.
#
# Why more than one view helps at all: a yes/no probe carries acquiescence bias,
# a lean toward "yes" that inflates every finding equally and therefore survives
# any per-finding weighting. Flipping polarity points that lean the other way.
# Flipping framing moves the model from generation to verification, which is a
# different decision process on the same evidence. Measured on the first two:
# mean p(present) 0.346 against 0.608, correlation +0.100.
#
# `invert` says whether a "yes" means the finding is ABSENT, so every arm is
# stored as p(present) and the arms are directly poolable.
_HEAD = """\
You are an experienced radiologist examining a breast ultrasound image.
"""

_PHRASINGS = {
    # direct observation, positive polarity
    1: (_HEAD + """
Look at the image and answer one question about what you can actually see.

Does the mass in this image show {desc}?

Answer with one word, yes or no.""", False),

    # direct observation, negative polarity
    2: (_HEAD + """
Look at the image and answer one question about what you can actually see.

Is the mass in this image FREE of {desc}?

Answer with one word, yes or no.""", True),

    # verification framing, positive polarity
    3: (_HEAD + """
Another radiologist has reviewed this image and reports:

    "The mass shows {desc}."

Examine the image yourself. Do you agree with that report?

Answer with one word, yes or no.""", False),

    # verification framing, negative polarity
    4: (_HEAD + """
Another radiologist has reviewed this image and reports:

    "The mass does not show {desc}."

Examine the image yourself. Do you agree with that report?

Answer with one word, yes or no.""", True),
}


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image, num_predict=3):
        msg = {"role": "user", "content": prompt, "images": [image]}
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": {"temperature": 0, "num_ctx": 2048, "num_predict": num_predict},
                "logprobs": True, "top_logprobs": 20}
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


def p_yes(out):
    """P(yes) normalised over the yes/no mass in the first-token distribution."""
    lp = out.get("logprobs") or out["message"].get("logprobs")
    if not lp:
        return None
    y = n = 0.0
    for c in lp[0].get("top_logprobs") or []:
        t = c.get("token", "").strip().lower()
        p = math.exp(c.get("logprob", -100.0))
        if t.startswith("yes"):
            y += p
        elif t.startswith("no"):
            n += p
    return y / (y + n) if (y + n) > 0 else None


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--split", default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    p.add_argument("--tag", default="probe")
    p.add_argument("--phrasing", type=int, default=1, choices=(1, 2, 3, 4),
                   help="which of the four factorial templates to ask")
    p.add_argument("--root", default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    from run_debate_v5 import all_findings

    root = Path(args.root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema = json.loads((root / "data/breast/schema.json").read_text())
    findings = all_findings(triples, schema)          # (feature, description, side)
    logger.info("probing %d findings", len(findings))

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    want = [int(x) for x in args.shards.split(",")] if args.shards else [args.shard]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards in want]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{args.tag}_{args.split}_{'-'.join(map(str, want))}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                try:
                    done.add(json.loads(ln)["index"])
                except Exception:
                    pass

    cl = Client(args.url, args.model)
    t0, k = time.time(), 0
    logger.info("probes | %s shards=%s | %d samples (%d done)",
                args.split, want, len(idxs), len(done))
    with out_path.open("a") as fh:
        for idx in idxs:
            if idx in done:
                continue
            k += 1
            b64 = _b64(imgs[idx], args.image_size)
            ts = time.time()
            vals = {}
            tmpl, invert = _PHRASINGS[args.phrasing]
            for feat, desc, _side in findings:
                try:
                    v = p_yes(cl.call(tmpl.format(desc=desc), b64))
                    # every arm is stored as p(finding PRESENT) so they pool directly
                    vals[feat] = None if v is None else (1.0 - v if invert else v)
                except Exception as e:
                    logger.warning("  %s %s failed: %s", idx, feat, e)
                    vals[feat] = None
            fh.write(json.dumps({"index": idx, "phrasing": args.phrasing,
                                 "gold": _LABEL[int(labels[idx][0])],
                                 "p_yes": vals,
                                 "time_s": round(time.time() - ts, 1)}) + "\n")
            fh.flush()
            rate = (time.time() - t0) / k
            got = sum(1 for v in vals.values() if v is not None)
            logger.info("  [%3d/%3d] %03d  %d/%d probes  | ETA %.2fh",
                        k, len(idxs), idx, got, len(findings),
                        rate * (len(idxs) - k) / 3600)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_path)


if __name__ == "__main__":
    main()
