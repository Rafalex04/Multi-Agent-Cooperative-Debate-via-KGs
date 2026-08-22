"""Probe the KG's differential diagnosis: ask which LESION the image shows.

Same measurement mechanism as the finding probes -- one yes/no question, p(yes)
read off the first-token distribution -- pointed at the other half of the
ontology. Where a finding probe asks "does the mass show spiculated margins",
this asks "is this a simple breast cyst".

Two phrasings, mirroring the polarity axis that produced the biggest single gain
in the finding probes (the negated arm was worth +0.05 AUC by cancelling
acquiescence bias). Framing is held fixed at direct observation, because a
differential is a naming task and the verification framing does not apply to it
as cleanly.

Every arm is stored as p(lesion PRESENT) so the arms pool directly.
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from lesions import lesion_table                                      # noqa: E402

_LABEL = {0: "MALIGNANT", 1: "BENIGN"}
_HEAD = "You are an experienced radiologist examining a breast ultrasound image.\n"

_PHRASINGS = {
    1: (_HEAD + """
Look at the image and answer one question about the most likely diagnosis.

Is the lesion in this image {desc}?

Answer with a single word, yes or no.""", False),
    2: (_HEAD + """
Look at the image and answer one question about the most likely diagnosis.

Can you rule out that the lesion in this image is {desc}?

Answer with a single word, yes or no.""", True),
}


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image, num_predict=3):
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt, "images": [image]}],
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 2048, "num_predict": num_predict},
                "logprobs": True, "top_logprobs": 20}
        req = urllib.request.Request(self.url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


def p_yes(out):
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
    p.add_argument("--npz", default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--shards", default="0")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--phrasing", type=int, default=1, choices=(1, 2))
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    p.add_argument("--tag", default="lesion")
    p.add_argument("--out-dir", default=str(_HERE / "results"))
    p.add_argument("--root", default=str(_HERE.parents[1] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    lesions = lesion_table()
    logger.info("probing %d lesions", len(lesions))

    z = np.load(Path(args.npz) if args.npz
                else Path(args.root) / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    want = [int(x) for x in args.shards.split(",")]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards in want]

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{args.tag}_{args.split}_{'-'.join(map(str, want))}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text(errors="replace").splitlines():
            if ln.strip():
                try:
                    done.add(json.loads(ln)["index"])
                except Exception:
                    pass

    cl = Client(args.url, args.model)
    tmpl, invert = _PHRASINGS[args.phrasing]
    t0, k = time.time(), 0
    logger.info("lesion probes | %s shards=%s | %d samples (%d done)",
                args.split, want, len(idxs), len(done))
    with out_path.open("a") as fh:
        for idx in idxs:
            if idx in done:
                continue
            k += 1
            b64 = _b64(imgs[idx], args.image_size)
            ts, vals = time.time(), {}
            for ent, plain, _pr, _n, _src in lesions:
                try:
                    v = p_yes(cl.call(tmpl.format(desc=plain), b64))
                    vals[ent] = None if v is None else (1.0 - v if invert else v)
                except Exception as e:
                    logger.warning("  %s %s failed: %s", idx, ent, e)
                    vals[ent] = None
            fh.write(json.dumps({"index": idx, "phrasing": args.phrasing,
                                 "gold": _LABEL[int(labels[idx][0])], "p_yes": vals,
                                 "time_s": round(time.time() - ts, 1)}) + "\n")
            fh.flush()
            rate = (time.time() - t0) / k
            logger.info("  [%3d/%3d] %04d  %d/%d  | ETA %.2fh", k, len(idxs), idx,
                        sum(1 for v in vals.values() if v is not None), len(lesions),
                        rate * (len(idxs) - k) / 3600)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_path)


if __name__ == "__main__":
    main()
