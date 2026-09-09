"""Judge arm: a separate model reads the transcript and issues the FINAL verdict.

Every debate corpus in this project is scored by AGGREGATING claims -- weighted
vote on breast, mean claim grade on retina, modal class on derma. That is a
readout WE chose. The judge arm asks a different question: if a fresh model reads
the whole discussion and decides, does the debate carry more than our aggregator
extracts?

This matters because C2 ("debate adds nothing") could in principle be an artefact
of the aggregator rather than of the debate. `skip_judge=true` was set for every
dataset run on breast (SPEC Key Design Decision 2: MedGemma cost ~5 min/sample),
so the question has never actually been tested at corpus scale.

The judge sees the TRANSCRIPT ONLY, never the image -- so it cannot re-do the
perception, only adjudicate what was argued. That is the point.

  python run_judge.py --tree retina --split test --num-shards 8 --shard 0
"""
from __future__ import annotations

import argparse, json, logging, re, sys, time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments/10_debate_v5"))
sys.path.insert(0, str(REPO / "experiments/26_baselines"))
from run_debate_v5 import Client                                     # noqa: E402
from corpus_io import append, done_indices                           # noqa: E402

TREES = {
    "retina": dict(pack=REPO / "retinaMnist/data/retina",
                   ontology=REPO / "retinaMnist/conf/ontology_retina.yaml",
                   debates="debates_r1", gold="gold_grade", kind="ordinal",
                   expert="ophthalmologist", modality="colour fundus photograph"),
    "derma": dict(pack=REPO / "dermaMnist/data/derma",
                  ontology=REPO / "dermaMnist/conf/ontology_derma.yaml",
                  debates="debates_d1", gold="gold_class", kind="nominal",
                  expert="dermatologist", modality="dermoscopic image"),
}


def label_space(cfg, kind):
    if kind == "ordinal":
        lv = cfg["levels"]
        menu = "\n".join(f"  {i}  {n.replace('_',' ')}" for i, n in enumerate(lv))
        return lv, menu, "a single ICDR grade 0-%d" % (len(lv) - 1)
    cl = cfg["classes"]; nm = cfg["class_names"]
    menu = "\n".join(f"  {c:6s} {nm[c]}" for c in cl)
    return cl, menu, "a single class code"


def judge_prompt(claims, menu, ask, expert, modality, kind):
    lines = "\n".join(
        f"  {c['node_id']} | {c.get('grade', c.get('label')) if (c.get('grade') is not None or c.get('label')) else '-'}"
        f" | {c['text']}" for c in claims) or "  (no claims)"
    field = "grade" if kind == "ordinal" else "class code"
    return f"""\
You are a senior {expert} adjudicating a case discussion about a {modality}. \
You have NOT seen the image. Below is every observation the two {expert}s \
recorded, each with the {field} they thought it implied on its own.

{lines}

The possible answers are:
{menu}

Weigh the observations against each other and decide the single most likely \
answer for this case. Observations may conflict; that is normal. Some may be \
mistaken.

Reply with {ask} and nothing else. No explanation."""


def parse_verdict(text, labels, kind):
    t = (text or "").strip().lower()
    if kind == "ordinal":
        m = re.search(r"\b([0-9])\b", t)
        if m and int(m.group(1)) < len(labels):
            return int(m.group(1))
        return None
    for i, c in enumerate(labels):
        if re.search(rf"\b{re.escape(c)}\b", t):
            return i
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, choices=sorted(TREES))
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    p.add_argument("--split", default="test")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    T = TREES[a.tree]
    cfg = yaml.safe_load(Path(T["ontology"]).read_text())
    labels, menu, ask = label_space(cfg, T["kind"])
    src = Path(T["pack"]) / T["debates"]
    out_dir = Path(a.out_dir) if a.out_dir else Path(T["pack"]) / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = {}
    for f in sorted(src.glob(f"{a.split}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if ln.strip():
                try:
                    d = json.loads(ln); recs[int(d["sample_id"])] = d
                except Exception:
                    pass
    idxs = [i for i in sorted(recs) if i % a.num_shards == a.shard]
    done = done_indices(out_dir, a.split)
    todo = [i for i in idxs if i not in done]

    cl = Client(a.url, a.model)
    logger.info("judge %s | %s shard %d/%d | %d samples (%d done)",
                a.tree, a.split, a.shard, a.num_shards, len(todo), len(done))
    t0, ok = time.time(), 0
    for k, idx in enumerate(todo, 1):
        d = recs[idx]; ts = time.time()
        try:
            txt = cl.call(judge_prompt(d.get("claims") or [], menu, ask,
                                       T["expert"], T["modality"], T["kind"]),
                          image=None, num_predict=8, temperature=0.0
                          )["message"]["content"]
        except Exception as e:
            logger.warning("  %s failed: %s", idx, e); continue
        v = parse_verdict(txt, labels, T["kind"])
        ok += v is not None
        append(out_dir, a.split, a.shard,
               {"sample_id": idx, "gold": d[T["gold"]], "verdict": v,
                "raw": (txt or "")[:60], "n_claims": len(d.get("claims") or [])})
        if k % 25 == 0 or k == len(todo):
            r = (time.time() - t0) / k
            logger.info("  [%4d/%4d] parsed %d/%d | %.1fs/img | ETA %.1fh",
                        k, len(todo), ok, k, r, r * (len(todo) - k) / 3600)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
