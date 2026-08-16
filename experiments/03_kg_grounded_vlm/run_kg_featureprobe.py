"""Option 8: KG as scoring scaffold — per-feature visual probes weighted by KG stance.

Every previous attempt injected KG text into the classification prompt, which made
the model read rules instead of looking at the image, and collapsed p_malignant into
a bimodal 0/1 distribution (AUC 0.39-0.58, all below the 0.601 no-KG baseline).

This inverts the design. The KG never enters the classification prompt. Instead it
determines *which* visual questions to ask and *how to weight the answers*:

  1. PROBE   For each visually-assessable stance feature in the KG, ask one yes/no
             question with the image. Extract p(yes) from first-token logprobs.
  2. WEIGHT  KG stance relation gives the sign: suggests_malignancy -> +1,
             suggests_benign -> -1. object=="possible" halves the magnitude.
  3. SCORE   score = sum(w_f * p_yes(f)) / sum(|w_f|), a genuinely continuous value.

Why this beats prompt injection:
  - Each call is a short yes/no with the image, the exact format the zero-shot
    baseline handles well, so the image dominates the context.
  - Summing many continuous probabilities destroys the bimodality that capped AUC.
  - Fully KG-driven and KG-generalizable: swap the graph, get a different probe set.
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL  = "http://localhost:11434/api/chat"
RESULTS_DIR = Path(__file__).parent / "results"
_LABEL_MAP  = {0: "MALIGNANT", 1: "BENIGN"}

# Stance relations mapped to their diagnostic polarity.
_POLARITY = {
    "suggests_malignancy":            +1.0,
    "supports_malignancy_diagnosis":  +1.0,
    "suggests_birads_5":              +1.0,
    "suggests_benign":                -1.0,
    "present_in_benign":              -1.0,
    "suggests_benign_simple_cyst":    -1.0,
    "has_no_malignant_potential":     -1.0,
}

# Features not assessable on a 2D grayscale ultrasound crop. This is a modality
# constraint, not a diagnostic one: elastography needs a different acquisition,
# nodal/skin findings sit outside the cropped field of view, and interval growth
# needs a prior study. Probing them yields noise, not signal.
_NOT_VISIBLE = {
    "soft_elasticity", "hard_elasticity", "compressibility",
    "axillary_adenopathy", "nipple_retraction", "skin_thickening",
    "skin_retraction", "evidence_of_interval_growth",
    "duct_changes_with_intraductal_mass",
    "intramammary_lymph_node_loss_of_circumscription",
    "fat_containing_lesion",
}

# Phrasing for features whose snake_case name is not a natural question subject.
# Falls back to the humanised name when absent.
_PHRASING = {
    "spiculated":            "spiculated margins, with sharp angular lines radiating outward from the mass edge",
    "spiculated_margin":     "spiculated margins, with sharp angular lines radiating outward from the mass edge",
    "irregular_shape":       "an irregular shape, meaning the mass is neither oval nor round",
    "non_parallel_orientation": "a non-parallel orientation, meaning the mass is taller than it is wide",
    "parallel_orientation":  "a parallel orientation, meaning the mass is wider than it is tall",
    "circumscribed_margin":  "circumscribed margins, meaning a smooth, sharply defined boundary",
    "oval_shape":            "an oval or round shape with a smooth contour",
    "anechoic_content":      "anechoic content, meaning the inside of the mass is uniformly black like fluid",
    "hyperechoic_mass":      "a hyperechoic mass, brighter than the surrounding fat",
    "thin_uniform_pseudocapsule": "a thin, uniform bright capsule around the mass",
    "echogenic_pseudocapsule":    "a thin bright echogenic rim surrounding the mass",
    "posterior_shadowing_with_solid_irregular_mass":
        "posterior acoustic shadowing, a dark band directly behind a solid irregular mass",
    "microcalcifications_in_hypoechoic_mass":
        "microcalcifications, tiny bright punctate specks inside a dark mass",
    "echogenic_rind":        "a thick irregular bright halo around the mass",
    "clustered_microcysts":  "a cluster of tiny anechoic cysts under 3mm with thin septations",
    "spiculated_or_irregular_mass":
        "a mass that is either spiculated or irregular in shape",
    "architectural_distortion":
        "architectural distortion, with radiating straight lines, retraction, or blurring "
        "of the normal tissue planes around the mass",
}


def _h(s: str) -> str:
    return s.replace("_", " ").strip()


def build_probes(triples: list[dict], schema: dict) -> list[dict]:
    """Derive the probe set from KG stance triples.

    Returns [{feature, question, weight}], deduplicated by feature, keeping the
    largest-magnitude weight when a feature carries several stance triples.
    """
    by_feature: dict[str, float] = {}
    for t in triples:
        rel = t.get("relation", "")
        if rel not in _POLARITY:
            continue
        subj = t.get("subject", "")
        if subj in _NOT_VISIBLE or subj.startswith("birads"):
            continue
        w = _POLARITY[rel]
        # "possible" markers are weaker evidence than "true"
        if t.get("object") == "possible":
            w *= 0.5
        if abs(w) > abs(by_feature.get(subj, 0.0)):
            by_feature[subj] = w

    probes = []
    for subj, w in by_feature.items():
        desc = _PHRASING.get(subj, _h(subj))
        probes.append({
            "feature": subj,
            "weight": w,
            "question": (
                "This is a breast ultrasound image.\n"
                f"Does the mass in this image show {desc}?\n"
                "Answer with exactly one word: yes or no."
            ),
        })
    # Stable order: malignant probes first, then benign, alphabetical within each.
    probes.sort(key=lambda p: (-p["weight"], p["feature"]))
    return probes


class _Shim:
    def __init__(self, model, timeout=1800):
        self.model = model
        self.timeout = timeout

    def call(self, prompt, image=None, num_predict=3, logprobs=True):
        msg = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [image]
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": {"temperature": 0, "num_ctx": 2048, "num_predict": num_predict}}
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


def _p_yes(logprobs):
    """P(yes) normalised over the yes/no token mass in the first position."""
    if not logprobs:
        return None
    top = logprobs[0].get("top_logprobs") or []
    py = pn = 0.0
    for c in top:
        tok = c.get("token", "").strip().lower().lstrip("*_# ")
        p   = math.exp(c.get("logprob", -100.0))
        if tok.startswith("yes"):
            py += p
        elif tok.startswith("no"):
            pn += p
    return py / (py + pn) if (py + pn) > 0 else None


def score_sample(probe_results: list[dict]) -> float | None:
    """Weighted mean of probe probabilities, mapped to [0, 1].

    score = sum(w * p_yes) / sum(|w|) lands in [-1, 1]; rescale to [0, 1] so it
    reads as a malignancy probability.
    """
    num = den = 0.0
    for r in probe_results:
        if r["p_yes"] is None:
            continue
        num += r["weight"] * r["p_yes"]
        den += abs(r["weight"])
    if den == 0:
        return None
    return (num / den + 1.0) / 2.0


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

    from medmnist import BreastMNIST

    root    = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    probes  = build_probes(triples, schema)

    logger.info("Derived %d probes from KG stance triples:", len(probes))
    for pr in probes:
        logger.info("   w=%+.1f  %s", pr["weight"], pr["feature"])

    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    n  = len(ds.imgs) if args.limit is None else min(args.limit, len(ds.imgs))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"{args.model.replace(':','_')}_{args.split}_kg_featureprobe.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d done", len(done))

    client  = _Shim(args.model, args.timeout)
    t_start = time.time()
    logger.info("Option 8: KG feature probes | model=%s | %d probes x %d samples",
                args.model, len(probes), n)

    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64  = _b64(ds.imgs[idx], args.image_size)

            t0 = time.time()
            results = []
            for pr in probes:
                out = client.call(pr["question"], image=b64, num_predict=3, logprobs=True)
                lp  = out.get("logprobs") or out["message"].get("logprobs")
                results.append({
                    "feature": pr["feature"],
                    "weight":  pr["weight"],
                    "p_yes":   _p_yes(lp),
                })
            elapsed = time.time() - t0

            p_m  = score_sample(results)
            pred = None if p_m is None else ("MALIGNANT" if p_m >= 0.5 else "BENIGN")

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_m,
                "probes": results, "time_s": round(elapsed, 1),
            }) + "\n")
            fh.flush()
            done.add(idx)

            rate = (time.time() - t_start) / max(1, len(done))
            logger.info("  [%3d/%3d] gold=%-9s pred=%-9s p=%s | %.0fs | ETA %.2fh",
                len(done), n, gold, pred,
                f"{p_m:.3f}" if p_m is not None else " n/a",
                elapsed, rate * (n - len(done)) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t_start) / 60, out_path)


if __name__ == "__main__":
    main()
