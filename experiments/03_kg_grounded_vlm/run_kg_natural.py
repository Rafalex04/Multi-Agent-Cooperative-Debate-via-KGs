"""Option 4: Natural-language KG rules + visual descriptions + 2-feature threshold.

Fixes from Option 1 (stance_direct):
  - Rules rewritten as clinical sentences with visual descriptions (from KG appearance triples).
  - Only definite malignancy indicators kept (dropped "possible" and non-visual ones:
    hard_elasticity, axillary_adenopathy, skin_retraction, echogenic_rind, etc.).
  - Explicit "2+ high-suspicion features → malignant" threshold to stop single-trigger FPs.
  - Benign combination patterns added (from KG combination_indicates triples).
  - No observation step (avoids corrupted feature extraction).
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

# Natural language prompt derived from KG stance + appearance triples.
# Malignancy rules sourced from: suggests_malignancy=true + appearance triples for each feature.
# Benign rules sourced from: suggests_benign + combination_indicates triples.
# Dropped: hard_elasticity (requires elastography), axillary_adenopathy (out of frame),
#          skin_retraction/thickening (surface features, not always visible),
#          echogenic_rind, duct_changes (uncommon), posterior_shadowing alone (benign mimics).
CLASSIFY_PROMPT = """\
You are a radiologist examining a breast ultrasound image. Apply ACR BI-RADS criteria.

HIGH-SUSPICION features (any one strongly suggests malignancy):
1. Spiculated margins — angular projections or spicules radiating outward from the mass edge
2. Irregular shape — not oval and not round; angular, amorphous, or highly lobulated contour
3. Non-parallel orientation — mass is taller than wide (depth exceeds width)
4. Architectural distortion — radiating lines, blurring of normal tissue planes, or \
retraction/distortion of surrounding parenchyma around the mass

LOW-SUSPICION features (suggest benign):
A. Circumscribed margins — smooth, well-defined, abrupt boundary with surrounding tissue
B. Oval shape — mildly elliptical or gently lobulated (≤3 lobulations)
C. Parallel orientation — mass is wider than tall
D. Posterior acoustic enhancement — bright region directly posterior to mass (suggests fluid)

Benign mimics to avoid false alarms:
- Spiculated margins and architectural distortion can also appear in postsurgical scar and \
fat necrosis (late phase) — these are benign; consider clinical context.
- Irregular shape alone in a soft, compressible mass without other high-suspicion features \
may be benign.

Common benign patterns (predict benign if you see one of these):
- Oval + circumscribed + hypoechoic + parallel → likely fibroadenoma
- Round/oval + anechoic + circumscribed + posterior enhancement → likely simple cyst

DECISION RULE:
- Predict MALIGNANT if you clearly observe any HIGH-SUSPICION feature (1-4) that is NOT \
explained by a known benign mimic or a benign combination pattern.
- Predict BENIGN if features are predominantly low-suspicion or fit a benign pattern.

Examine the image carefully, then answer with exactly one word: malignant or benign.\
"""


class _Shim:
    def __init__(self, model, timeout=1800):
        self.model = model
        self.timeout = timeout

    def call(self, prompt, image=None, num_predict=5, logprobs=False):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="medgemma:4b")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--timeout",    type=int, default=1800)
    p.add_argument("--output",     default=None)
    args = p.parse_args()

    from medmnist import BreastMNIST
    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    n  = len(ds.imgs) if args.limit is None else min(args.limit, len(ds.imgs))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"{args.model.replace(':','_')}_{args.split}_kg_natural.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d done", len(done))

    client  = _Shim(args.model, args.timeout)
    t_start = time.time()
    logger.info("Option 4: natural KG | model=%s | %d samples", args.model, n)

    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64  = _b64(ds.imgs[idx], args.image_size)

            t0  = time.time()
            out = client.call(CLASSIFY_PROMPT, image=b64, num_predict=5, logprobs=True)
            elapsed = time.time() - t0

            raw  = out["message"]["content"]
            lp   = out.get("logprobs") or out["message"].get("logprobs")
            pred = _pred(raw)
            p_m  = _p_mal(lp)

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_m,
                "raw": raw, "time_s": round(elapsed, 1),
            }) + "\n")
            fh.flush()
            done.add(idx)

            rate = (time.time() - t_start) / max(1, len(done))
            logger.info("  [%3d/%3d] gold=%-9s pred=%-9s p=%s | %.0fs | ETA %.1fh",
                len(done), n, gold, pred,
                f"{p_m:.3f}" if p_m is not None else " n/a",
                elapsed, rate * (n - len(done)) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t_start) / 60, out_path)


if __name__ == "__main__":
    main()
