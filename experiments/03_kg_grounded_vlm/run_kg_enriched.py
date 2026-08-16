"""Option 6: Enriched KG context — stance + appearance + benign patterns + mimics.

Builds the prompt context dynamically from the full KG:
  - Stance triples (suggests_malignancy / suggests_benign) as the diagnostic rules
  - ultrasound_appearance_includes triples: what each feature/lesion LOOKS LIKE visually
  - combination_indicates triples: multi-feature patterns -> diagnosis
  - differential_for / mimics: which benign conditions look malignant (reduce FPs)
  - typically_classified_as: benign-appearing conditions that are BI-RADS 2/3

No observation step (avoids corrupted feature extraction).
Single call per sample with image.
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, sys, time, urllib.request
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BREAST_SRC = Path(__file__).resolve().parents[2] / "breastMnist" / "src"
sys.path.insert(0, str(_BREAST_SRC))

OLLAMA_URL  = "http://localhost:11434/api/chat"
RESULTS_DIR = Path(__file__).parent / "results"
_LABEL_MAP  = {0: "MALIGNANT", 1: "BENIGN"}

_STANCE_MAL = {"suggests_malignancy"}
_STANCE_BEN = {"suggests_benign", "present_in_benign"}
_TAXONOMY   = {"is_a", "subtype_of", "birads_subcategory_of"}

# Benign lesion types whose appearance is most relevant to distinguish from malignancy
_BENIGN_LESIONS_OF_INTEREST = {
    "fibroadenoma", "simple_breast_cyst", "simple_cyst",
    "fat_necrosis_late_phase", "postsurgical_scar_tissue",
    "intramammary_lymph_node", "complicated_breast_cyst",
}

# Features whose KG appearance triples are most visually informative
_MAL_FEATURES_OF_INTEREST = {
    "architectural_distortion",
    "spiculated_margin",
    "irregular_shape",
    "non_parallel_orientation",
}


def _humanise(snake: str) -> str:
    """Convert snake_case KG labels to readable text."""
    return snake.replace("_", " ").replace("  ", " ").strip()


def build_kg_context(triples: list[dict], schema: dict) -> str:
    """Build enriched context block from the full KG."""
    stance_ids = set(schema.get("stance_triple_ids", []))
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        by_subject[t.get("subject", "")].append(t)

    # --- 1. Malignancy stance triples + their appearance ---
    mal_lines = []
    seen_subjects = set()
    for t in triples:
        if t["id"] not in stance_ids or t["relation"] not in _STANCE_MAL:
            continue
        subj = t.get("subject", "")
        # Skip non-visual / out-of-frame features
        if subj in {"hard_elasticity", "axillary_adenopathy", "skin_retraction",
                    "skin_thickening", "duct_changes_with_intraductal_mass",
                    "intramammary_lymph_node_loss_of_circumscription"}:
            continue
        if subj in seen_subjects:
            continue
        seen_subjects.add(subj)

        label = _humanise(subj)
        certainty = t.get("object", "true")
        qual = "" if certainty == "true" else f" ({certainty})"
        line = f"- {label}{qual}"

        # Attach appearance triples if this feature has them
        appearances = [tt["object"] for tt in by_subject.get(subj, [])
                       if tt["relation"] == "ultrasound_appearance_includes"]
        if appearances:
            app_text = "; ".join(_humanise(a) for a in appearances[:5])
            line += f"\n    Looks like: {app_text}"

        mal_lines.append(line)

    # --- 2. Benign stance triples + their appearance ---
    ben_lines = []
    seen_subjects = set()
    for t in triples:
        if t["id"] not in stance_ids or t["relation"] not in _STANCE_BEN:
            continue
        subj = t.get("subject", "")
        # Skip non-visual features
        if subj in {"soft_elasticity", "compressibility", "fat_containing_lesion"}:
            continue
        if subj in seen_subjects:
            continue
        seen_subjects.add(subj)

        label = _humanise(subj)
        line = f"- {label}"
        appearances = [tt["object"] for tt in by_subject.get(subj, [])
                       if tt["relation"] == "ultrasound_appearance_includes"]
        if appearances:
            app_text = "; ".join(_humanise(a) for a in appearances[:4])
            line += f"\n    Looks like: {app_text}"
        ben_lines.append(line)

    # --- 3. Typical benign lesion appearances ---
    benign_appearance_lines = []
    for lesion in ["fibroadenoma", "simple_breast_cyst", "complicated_breast_cyst"]:
        feats = [tt["object"] for tt in by_subject.get(lesion, [])
                 if tt["relation"] == "ultrasound_appearance_includes"]
        if feats:
            feat_text = ", ".join(_humanise(f) for f in feats[:5])
            benign_appearance_lines.append(f"- {_humanise(lesion)}: {feat_text}")

    # --- 4. Benign mimics of malignancy ---
    mimic_lines = []
    for lesion in ["fat_necrosis_late_phase", "postsurgical_scar_tissue"]:
        feats = [tt["object"] for tt in by_subject.get(lesion, [])
                 if tt["relation"] == "ultrasound_appearance_includes"]
        # also get typically_classified_as
        birads = next((tt["object"] for tt in by_subject.get(lesion, [])
                       if tt["relation"] == "typically_classified_as"), None)
        if feats:
            feat_text = ", ".join(_humanise(f) for f in feats[:5])
            birads_note = f" [{_humanise(birads)} = BENIGN]" if birads else " [BENIGN]"
            mimic_lines.append(f"- {_humanise(lesion)}{birads_note}: {feat_text}")

    # --- 5. Combination patterns ---
    combo_lines = []
    for t in triples:
        if t["relation"] == "combination_indicates":
            subj, obj = t.get("subject", ""), t.get("object", "")
            # Only include diagnostically useful ones
            if any(k in obj for k in ["fibroadenoma", "malignancy", "birads_5",
                                       "cyst", "lymph_node"]):
                verdict = "→ BENIGN" if any(k in obj for k in
                          ["fibroadenoma", "cyst", "lymph_node"]) else "→ HIGH SUSPICION"
                combo_lines.append(f"- {_humanise(subj)} {verdict}")

    for t in triples:
        if t["relation"] == "suggests_birads_5":
            combo_lines.append(
                f"- {_humanise(t['subject'])} → HIGHLY SUGGESTIVE OF MALIGNANCY (BI-RADS 5)")

    # Assemble
    sections = []
    if mal_lines:
        sections.append(
            "MALIGNANCY INDICATORS (any one present = suspicious):\n" +
            "\n".join(mal_lines))
    if ben_lines:
        sections.append(
            "BENIGN INDICATORS:\n" + "\n".join(ben_lines))
    if benign_appearance_lines:
        sections.append(
            "TYPICAL BENIGN LESION APPEARANCES (if image matches, predict benign):\n" +
            "\n".join(benign_appearance_lines))
    if mimic_lines:
        sections.append(
            "BENIGN MIMICS (can look like malignancy but are NOT):\n" +
            "\n".join(mimic_lines))
    if combo_lines:
        sections.append(
            "COMBINATION PATTERNS:\n" + "\n".join(combo_lines))

    return "\n\n".join(sections)


CLASSIFY_PROMPT = """\
You are a radiologist examining a breast ultrasound image.

Use the ACR BI-RADS knowledge below to classify this mass as malignant or benign.

{kg_context}

Examine the image carefully. Apply the knowledge above to what you see.
Is this breast mass malignant or benign?
Answer with exactly one word: malignant or benign.\
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
    p.add_argument("--kg-root",
        default=str(Path(__file__).resolve().parents[2] / "breastMnist"))
    args = p.parse_args()

    root    = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())

    kg_context = build_kg_context(triples, schema)
    prompt = CLASSIFY_PROMPT.format(kg_context=kg_context)

    logger.info("KG context built (%d chars, ~%d tokens)", len(kg_context), len(kg_context)//4)
    logger.info("--- KG CONTEXT ---\n%s\n---", kg_context)

    from medmnist import BreastMNIST
    ds = BreastMNIST(split=args.split, download=True, size=args.image_size)
    n  = len(ds.imgs) if args.limit is None else min(args.limit, len(ds.imgs))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"{args.model.replace(':','_')}_{args.split}_kg_enriched.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["index"])
        logger.info("Resuming: %d done", len(done))

    client  = _Shim(args.model, args.timeout)
    t_start = time.time()
    logger.info("Option 6: enriched KG | model=%s | %d samples", args.model, n)

    with out_path.open("a") as fh:
        for idx in range(n):
            if idx in done:
                continue
            gold = _LABEL_MAP[int(ds.labels[idx][0])]
            b64  = _b64(ds.imgs[idx], args.image_size)

            t0  = time.time()
            out = client.call(prompt, image=b64, num_predict=5, logprobs=True)
            elapsed = time.time() - t0

            raw  = out["message"]["content"]
            lp   = out.get("logprobs") or out["message"].get("logprobs")
            pred = _pred(raw)
            p_m  = _p_mal(lp)

            fh.write(json.dumps({
                "index": idx, "gold": gold, "pred": pred, "p_malignant": p_m,
                "raw": raw.strip(), "time_s": round(elapsed, 1),
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
