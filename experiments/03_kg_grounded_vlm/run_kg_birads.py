"""Option 5: BI-RADS scoring via KG rules — continuous probability via logprobs.

Instead of asking the model for a binary malignant/benign decision, we ask it to
assign a BI-RADS category (1–5) using KG-derived rules. We then extract the logprob
of each digit and compute:
  p_malignant = (p4 + p5) / (p1 + p2 + p3 + p4 + p5)

This gives a continuous probability rather than the bimodal 0/1 seen in all binary
approaches, which should improve AUC.

The KG provides:
- BI-RADS category definitions (from definition triples)
- BI-RADS 4/5 triggers (suggests_malignancy, suggests_birads_5)
- Benign pattern → BI-RADS 2/3 mappings (typically_classified_as triples)
- Likelihood_of_malignancy ranges per category
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

# KG-derived BI-RADS scoring rules
# Sources: definition, likelihood_of_malignancy, typically_classified_as, suggests_birads_5
CLASSIFY_PROMPT = """\
You are a radiologist assigning an ACR BI-RADS category to a breast ultrasound image.

ACR BI-RADS categories:
  1 = Negative — no mass or abnormality seen (malignancy ~0%)
  2 = Benign — definitely benign finding (malignancy ~0%)
      Examples: simple cyst (anechoic + circumscribed + posterior enhancement), lipoma,
                intramammary lymph node, sebaceous cyst, calcified fibroadenoma
  3 = Probably benign — short-term follow-up recommended (malignancy <2%)
      Examples: oval + circumscribed + hypoechoic + parallel solid mass (fibroadenoma-like),
                complicated cyst (thin wall + low-level echoes),
                architectural distortion from postsurgical scar
  4 = Suspicious — biopsy recommended (malignancy 2–95%)
      4A (2–10%): microlobulated or mildly irregular margins, not classic fibroadenoma
      4B (10–50%): irregular shape OR non-parallel orientation OR indistinct margins
      4C (50–95%): irregular + non-parallel + indistinct margins combined
  5 = Highly suggestive of malignancy (malignancy ≥95%)
      Requires: spiculated margins (angular projections from edge) AND/OR
                irregular shape + non-parallel orientation (taller than wide)

Features that indicate BI-RADS 5:
- Spiculated margins: sharp angular projections radiating from the mass surface
- Irregular shape (not oval, not round) combined with non-parallel orientation

Features that indicate BI-RADS 4:
- Architectural distortion (radiating lines or blurring of tissue planes around mass)
- Irregular shape without clear benign pattern
- Non-parallel orientation (taller than wide)
- Microlobulated margins (small undulations)

Features that favor BI-RADS 2–3:
- Oval shape, circumscribed margins, parallel orientation, posterior enhancement

Examine the image. Assign the most appropriate BI-RADS category.
Answer with exactly one digit: 1, 2, 3, 4, or 5.\
"""

# Map BI-RADS to malignancy probability midpoints
_BIRADS_P = {1: 0.0, 2: 0.01, 3: 0.02, 4: 0.25, 5: 0.95}


class _Shim:
    def __init__(self, model, timeout=1800):
        self.model = model
        self.timeout = timeout

    def call(self, prompt, image=None, num_predict=3, logprobs=False):
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


def _extract_birads(logprobs, raw_text):
    """Extract BI-RADS score and p_malignant from logprobs.

    p_malignant = (p4 + p5) / (p1+p2+p3+p4+p5)
    Also returns the most likely BI-RADS digit.
    """
    ps = {str(i): 0.0 for i in range(1, 6)}
    if logprobs:
        top = logprobs[0].get("top_logprobs") or []
        for c in top:
            tok = c.get("token", "").strip()
            if tok in ps:
                ps[tok] += math.exp(c.get("logprob", -100.0))

    total = sum(ps.values())
    if total > 0:
        p_mal = (ps["4"] + ps["5"]) / total
    else:
        p_mal = None

    # Parse the text output as fallback
    for ch in raw_text.strip():
        if ch in "12345":
            birads = int(ch)
            break
    else:
        birads = None

    # If logprob-based p_mal is degenerate (all mass on one token), fall back to birads map
    if p_mal is not None and total < 0.01:
        p_mal = _BIRADS_P.get(birads, None)

    return birads, p_mal


def _pred(birads):
    if birads is None:
        return None
    return "MALIGNANT" if birads >= 4 else "BENIGN"


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
        RESULTS_DIR / f"{args.model.replace(':','_')}_{args.split}_kg_birads.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d done", len(done))

    client  = _Shim(args.model, args.timeout)
    t_start = time.time()
    logger.info("Option 5: BI-RADS scoring | model=%s | %d samples", args.model, n)

    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64  = _b64(ds.imgs[idx], args.image_size)

            t0  = time.time()
            out = client.call(CLASSIFY_PROMPT, image=b64, num_predict=3, logprobs=True)
            elapsed = time.time() - t0

            raw = out["message"]["content"]
            lp  = out.get("logprobs") or out["message"].get("logprobs")
            birads, p_m = _extract_birads(lp, raw)
            pred = _pred(birads)

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_m,
                "birads": birads, "raw": raw.strip(), "time_s": round(elapsed, 1),
            }) + "\n")
            fh.flush()
            done.add(idx)

            rate = (time.time() - t_start) / max(1, len(done))
            logger.info("  [%3d/%3d] gold=%-9s pred=%-9s BI-RADS=%s p=%s | %.0fs | ETA %.1fh",
                len(done), n, gold, pred, birads,
                f"{p_m:.3f}" if p_m is not None else " n/a",
                elapsed, rate * (n - len(done)) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t_start) / 60, out_path)


if __name__ == "__main__":
    main()
