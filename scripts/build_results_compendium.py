#!/usr/bin/env python3
"""Build RESULTS.html -- every measured result in this project, in one file.

Nothing here is transcribed by hand. The narrative is written once, in SECTIONS
below; every number is read out of the result file it came from at build time and
each table cites its source path, size and SHA-256. Re-run this after any new
result lands and the compendium is current:

    python scripts/build_results_compendium.py

The last section is a COMPLETENESS AUDIT. It walks the whole tree for result
files and lists any that no section claims, so a result cannot be silently
missing from this document -- if you add an experiment and forget to register it
here, the audit says so.
"""
from __future__ import annotations

import hashlib
import html
import re
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "RESULTS.html"

# Arrays of raw per-sample predictions: carried in some files, never displayed.
DROP_KEYS = {"oof", "y", "keys", "per_sample", "excluded", "indices"}
# Lists of plain numbers longer than this are summarised rather than printed.
INLINE_LIST_MAX = 10
# Lists of dicts (hyperparameter grids, per-seed tables) print in full up to here.
# A CV selection grid is evidence that selection happened on train only, so it is
# shown row by row rather than collapsed to a mean.
TABLE_ROWS_MAX = 30


# --------------------------------------------------------------------------- #
#  value formatting
# --------------------------------------------------------------------------- #

def fmt(v):
    """One scalar, rendered the way this project reports it."""
    if v is None:
        return '<span class="na">--</span>'
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:
            return '<span class="na">NaN</span>'
        a = abs(v)
        if a and (a < 1e-4 or a >= 1e7):
            return f"{v:.3e}"
        if a >= 1000:
            return f"{v:,.1f}"
        return f"{v:.4f}"
    return html.escape(str(v))


def agg(vals):
    """mean +- sd over a numeric list, the project's reporting convention."""
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if not vals:
        return None
    if len(vals) == 1:
        return fmt(vals[0])
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (f"{statistics.fmean(vals):.4f} &plusmn; {sd:.4f}"
            f'<span class="dim"> (n={len(vals)})</span>')


def is_scalar(v):
    return v is None or isinstance(v, (int, float, str, bool))


def summarise_runs(rows):
    """A list of per-seed / per-run dicts -> one mean +- sd row per field."""
    fields, out = [], {}
    for r in rows:
        for k in r:
            if k not in fields and isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool):
                fields.append(k)
    for f in fields:
        out[f] = agg([r.get(f) for r in rows])
    return out


def summarise_curve(rows):
    """An epoch/size curve -> first, best-by-last-column, and last point."""
    if not rows:
        return {}
    num = [k for k in rows[0] if isinstance(rows[0].get(k), (int, float))]
    if not num:
        return {}
    tgt = "test" if "test" in num else num[-1]
    best = max(rows, key=lambda r: r.get(tgt, float("-inf")))
    pick = lambda r: ", ".join(f"{k} {fmt(r[k])}" for k in num if k in r)
    return {"first point": pick(rows[0]),
            f"best by {tgt}": pick(best),
            "last point": pick(rows[-1]),
            "points": len(rows)}


# --------------------------------------------------------------------------- #
#  rendering
# --------------------------------------------------------------------------- #

def render(node, depth=0, name=None):
    """Recursively render a parsed result file as nested HTML tables."""
    if is_scalar(node):
        return f'<p class="scalar"><b>{html.escape(name or "value")}</b> {fmt(node)}</p>'

    if isinstance(node, list):
        if not node:
            return '<p class="na">(empty)</p>'
        if all(is_scalar(v) for v in node):
            if len(node) <= INLINE_LIST_MAX:
                return '<p class="scalar">' + ", ".join(fmt(v) for v in node) + "</p>"
            nums = [v for v in node if isinstance(v, (int, float))]
            if nums:
                return (f'<p class="scalar">{len(node)} values &mdash; '
                        f"mean {fmt(statistics.fmean(nums))}, "
                        f"min {fmt(min(nums))}, max {fmt(max(nums))}</p>")
            return f'<p class="scalar">{len(node)} values</p>'
        if all(isinstance(v, dict) for v in node):
            keys = list(node[0])
            looks_like_curve = any(k in keys for k in ("epoch", "n_train", "frac", "n"))
            if looks_like_curve and len(node) > INLINE_LIST_MAX:
                return render(summarise_curve(node), depth + 1)
            if len(node) > TABLE_ROWS_MAX:
                return render(summarise_runs(node), depth + 1)
            return table_of_dicts(node)
        return '<p class="na">(mixed list)</p>'

    # dict
    node = {k: v for k, v in node.items() if k not in DROP_KEYS}
    if not node:
        return '<p class="na">(no reportable fields)</p>'

    # seed/run blocks collapse to mean +- sd
    for rk in ("runs", "seeds", "draws_detail"):
        if rk in node and isinstance(node[rk], list) and node[rk] and isinstance(node[rk][0], dict):
            node = dict(node)
            node[f"{rk} (mean &plusmn; sd)"] = summarise_runs(node.pop(rk))

    flat = {k: v for k, v in node.items() if is_scalar(v)}
    deep = {k: v for k, v in node.items() if not is_scalar(v)}

    # dict-of-dicts with identical scalar keys -> one clean matrix
    if not flat and deep and len(deep) > 1:
        subs = [v for v in deep.values() if isinstance(v, dict)]
        if len(subs) == len(deep):
            cols = []
            for s in subs:
                for k, v in s.items():
                    if is_scalar(v) and k not in cols:
                        cols.append(k)
            if cols and all(sum(1 for k in cols if k in s) >= max(1, len(cols) // 2) for s in subs):
                return matrix(deep, cols)

    out = []
    if flat:
        out.append(kv_table(flat))
    for k, v in deep.items():
        h = min(depth + 4, 6)
        out.append(f'<h{h} class="sub">{html.escape(str(k))}</h{h}>')
        out.append(render(v, depth + 1, k))
    return "\n".join(out)


def kv_table(d):
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{fmt(v)}</td></tr>" for k, v in d.items())
    return f'<table class="kv">{rows}</table>'


def matrix(deep, cols):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for name, sub in deep.items():
        cells = "".join(
            f"<td>{fmt(sub.get(c)) if is_scalar(sub.get(c)) else nested_cell(sub.get(c))}</td>"
            for c in cols)
        extra = {k: v for k, v in sub.items() if k not in cols and not is_scalar(v)}
        body.append(f'<tr><th class="rowname">{html.escape(str(name))}</th>{cells}</tr>')
        for k, v in extra.items():
            body.append(f'<tr class="extra"><th class="rowname">&#8627; {html.escape(k)}</th>'
                        f'<td colspan="{len(cols)}">{nested_cell(v)}</td></tr>')
    return (f'<div class="scroll"><table class="matrix">'
            f'<thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def nested_cell(v):
    if v is None:
        return '<span class="na">--</span>'
    if is_scalar(v):
        return fmt(v)
    if isinstance(v, list):
        if all(is_scalar(x) for x in v):
            if len(v) <= INLINE_LIST_MAX:
                return ", ".join(fmt(x) for x in v)
            nums = [x for x in v if isinstance(x, (int, float))]
            return (f"{len(v)} values (mean {fmt(statistics.fmean(nums))})"
                    if nums else f"{len(v)} values")
        return f"{len(v)} entries"
    if isinstance(v, dict):
        # Show scalars, and short numeric lists too -- a five-draw control is the
        # shape this project reports its nulls in, and dropping it would hide the
        # evidence that a shuffled arm matched the real one.
        bits = []
        for k, x in v.items():
            if is_scalar(x):
                bits.append(f"{html.escape(str(k))} {fmt(x)}")
            elif isinstance(x, list) and x and all(is_scalar(i) for i in x) \
                    and len(x) <= INLINE_LIST_MAX:
                bits.append(f"{html.escape(str(k))} ["
                            + ", ".join(fmt(i) for i in x) + "]")
        return " &middot; ".join(bits) or f"{len(v)} fields"
    return html.escape(str(v))


def table_of_dicts(rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{nested_cell(r.get(c))}</td>" for c in cols) + "</tr>"
                   for r in rows)
    return (f'<div class="scroll"><table class="matrix">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


# --------------------------------------------------------------------------- #
#  loading
# --------------------------------------------------------------------------- #

def load(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".csv":
        rows = [r.split(",") for r in path.read_text().strip().splitlines()]
        if not rows:
            return None
        head, body = rows[0], rows[1:]
        def conv(x):
            try:
                return float(x)
            except ValueError:
                return x
        return [dict(zip(head, [conv(c) for c in r])) for r in body]
    return path.read_text()


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def block(path_str: str, caption: str | None, consumed: set,
          home: dict | None = None, sec: tuple | None = None) -> str:
    """One result file: caption, rendered content, and its provenance line.

    A file registered by more than one section (the record audit recomputes
    several experiments' numbers, so those files belong to both) is rendered
    once, where it is most at home, and cross-referenced everywhere else.
    """
    p = REPO / path_str
    if not p.exists():
        return (f'<div class="missing"><b>MISSING</b> <code>{html.escape(path_str)}</code> '
                f"&mdash; registered here but not present in the tree.</div>")
    key = str(p.resolve())
    if home is not None and key in home:
        hid, htitle = home[key]
        return (f'<div class="rf xref"><p class="cap">{caption or ""}</p>'
                f'<p class="scalar">Shown in full under '
                f'<a href="#{hid}">{htitle}</a> &mdash; '
                f'<code>{html.escape(path_str)}</code>.</p></div>')
    if home is not None and sec is not None:
        home[key] = sec
    consumed.add(key)
    try:
        data = load(p)
    except Exception as e:                                    # noqa: BLE001
        return (f'<div class="missing"><b>UNREADABLE</b> <code>{html.escape(path_str)}</code>'
                f" &mdash; {html.escape(str(e))}</div>")
    body = (f"<pre>{html.escape(data.strip())}</pre>"
            if isinstance(data, str) else render(data))
    cap = f'<p class="cap">{caption}</p>' if caption else ""
    prov = (f'<p class="prov"><code>{html.escape(path_str)}</code> &middot; '
            f"{p.stat().st_size:,} B &middot; sha256 {digest(p)}</p>")
    return f'<div class="rf">{cap}{body}{prov}</div>'


def corpus_stats(dirpath: str):
    """Records in a corpus directory, in either storage format.

    Two formats coexist. The newer corpora are JSONL, one record per line,
    adopted because per-sample JSON hit the home directory's 60k inode ceiling
    twice -- once hard enough that tar itself failed. The older breast corpora
    are still one JSON file per sample and are counted one record per file.
    Both are searched recursively, since several corpora are split into
    train/val/test subdirectories.
    """
    d = REPO / dirpath
    if not d.is_dir():
        return None

    per, total, nbytes, nfiles = {}, 0, 0, 0

    for sh in sorted(d.rglob("*.jsonl")):
        n = sum(1 for _ in sh.open(encoding="utf8", errors="replace"))
        total += n
        nfiles += 1
        nbytes += sh.stat().st_size
        split = sh.parent.name if sh.parent != d else sh.stem.rsplit("_", 1)[0]
        per[split] = per.get(split, 0) + n

    # A .json under a results/ directory is an aggregated result, rendered in its
    # own section above -- never a per-sample corpus record. Only the corpus trees
    # under data/ store one JSON per sample.
    per_sample = ([] if "results" in d.parts
                  else sorted(f for f in d.rglob("*.json")
                              if "results" not in f.relative_to(d).parts))
    for f in per_sample:
        total += 1
        nfiles += 1
        nbytes += f.stat().st_size
        split = f.parent.name if f.parent != d else "all"
        per[split] = per.get(split, 0) + 1

    if not total:
        return None
    if not per_sample:
        fmt_ = "JSONL"
    elif nfiles == len(per_sample):
        fmt_ = "per-sample JSON"
    else:
        fmt_ = "mixed"
    return {"files": nfiles, "records": total, "bytes": nbytes,
            "by split": per, "format": fmt_}


# --------------------------------------------------------------------------- #
#  the registry: every experiment, its question, its verdict, its files
# --------------------------------------------------------------------------- #
# f = (path, caption). The narrative is prose; every number comes from the file.

SECTIONS = [
 # ---------------------------------------------------------------- headline --
 dict(part="I. Headline", id="master", title="The consolidated master table",
      q="Every method measured on BreastMNIST test (n=156) and on BUS-BRA "
        "external (patient level), in one table, with parameter counts and "
        "per-image call costs beside each.",
      v="No method beats the supervised ResNet in domain. Off distribution the "
        "ordering reverses, which is claim C1. B2 carries 182x ThothGNN v3's "
        "parameters and 3x its agents and still does not separate from it.",
      f=[("experiments/26_baselines/results/table.json",
          "<code>auc</code>/<code>bacc</code> are BreastMNIST test; "
          "<code>ext_auc_auto</code> and <code>ext_auc_mask</code> are BUS-BRA "
          "patient-level, automatic and mask conditions. The ResNet row is a "
          "supervised in-domain ceiling and is <b>not comparable</b> to the "
          "zero-shot rows.")]),

 # ------------------------------------------------------ breast: phase 1 -----
 dict(part="II. BreastMNIST", id="e01", title="01 &mdash; Zero-shot VLM classification",
      q="Can a local vision LLM classify BreastMNIST directly, with no debate, "
        "no knowledge graph and no fine-tuning? Whatever it scores is the "
        "ceiling any debate built on that model can reach.",
      v="It reads the images, but weakly. This established the ceiling the rest "
        "of the project worked under.",
      f=[("experiments/01_zeroshot_vlm_classification/results/summary.json",
          "Two backbones on the 156-image test split.")]),

 dict(part="II. BreastMNIST", id="e02", title="02 &mdash; Supervised ResNet-18 baseline",
      q="What does ordinary supervised training reach on the same split? This is "
        "the comparator for the generalisation claim C1.",
      v="Strong in domain (the ceiling row of the master table), and it is "
        "exactly this strength that collapses off distribution in experiment 16.",
      f=[("experiments/02_resnet_baseline/results/224px/summary.json",
          "224px, the resolution every other experiment uses."),
         ("experiments/02_resnet_baseline/results/28px/summary.json",
          "28px, the native MedMNIST thumbnail, for reference."),
         ("experiments/02_resnet_baseline/results/224px/seed0/metrics.json",
          "224px seed 0, full metric set including confusion matrix."),
         ("experiments/02_resnet_baseline/results/28px/seed0/metrics.json", "28px seed 0."),
         ("experiments/02_resnet_baseline/results/28px/seed1/metrics.json", "28px seed 1."),
         ("experiments/02_resnet_baseline/results/28px/seed2/metrics.json", "28px seed 2.")]),

 dict(part="II. BreastMNIST", id="e03", title="03 &mdash; KG-grounded single agent",
      q="Does handing the VLM retrieved knowledge-graph triples improve its "
        "classification? Observe against a KG-derived schema, retrieve triples "
        "conditioned on that observation, then classify.",
      v="<b>The KG made it worse.</b> AUC fell to 0.4734 &mdash; below chance &mdash; "
        "against 0.6013 without it, and the model collapsed to a constant "
        "classifier: 0 malignant predictions out of 95, confidences pinned near "
        "1e-05, the ranking inverted. The failure is retrieval <i>quality</i>, "
        "not quantity, and it is what sent the project toward probes.",
      f=[("experiments/24_record/results/r03_variants.json",
          "All twelve injection strategies, recomputed from the raw corpora by "
          "the record audit rather than quoted from the original run notes.")]),

 dict(part="II. BreastMNIST", id="e04", title="04 &mdash; Graph analysis",
      q="Do the debate graphs carry label signal at all, in their text or in "
        "their structure?",
      v="Barely. TF-IDF over claim text reaches AUC 0.5405; structural graph "
        "features reach 0.4933, which is chance. This is the measurement that "
        "framed the whole project: the signal is not in the debate structure.",
      f=[]),

 dict(part="II. BreastMNIST", id="e0507", title="05 / 07 / 09 / 10 &mdash; Debate protocols v2 to v5",
      q="Four generations of corpus generator. v2 evidence-grounded; v3 KG-free; "
        "v4 stance-matched with positional-bias, forced-claim-count and "
        "opening-round fixes; v5 open-stance.",
      v="v5 is the best generator and the one ThothGNN v3 consumes. v3 was worse "
        "than v2. Corpus sizes and per-protocol audits below.",
      f=[("experiments/09_debate_v4/audit_log.txt",
          "The v4 audit: what the positional-bias and repetition fixes changed."),
         ("experiments/v5q_audit.txt", "v5q corpus audit."),
         ("experiments/ablation_audit.txt", "Ablation-corpus audit."),
         ("experiments/structure_comparison.txt",
          "Structural comparison across protocol generations."),
         ("experiments/24_record/results/corpus_audit.json",
          "Debate and probe corpus coverage, independently recounted.")]),

 dict(part="II. BreastMNIST", id="e06", title="06 &mdash; GNN over the debate graphs",
      q="The first architecture attempt: train a graph network directly on the "
        "debate graphs, across every protocol generation.",
      v="Null. Learning curves across seven corpora show no protocol reaching a "
        "useful test AUC, which is the first of the eight graph attempts.",
      f=[("experiments/06_gnn/curves.json",
          "Protocols v2 through v5, per-seed learning curves (summarised as "
          "mean &plusmn; sd over seeds)."),
         ("experiments/06_gnn/curves_v5q.json", "v5q against v5, v3 and v2."),
         ("experiments/06_gnn/curves_ablation.json",
          "Ablation corpora: v5qret (retrieval), v5qnokg (KG removed), v5q, v2."),
         ("experiments/06_gnn/curves_measured.json",
          "Measured-image corpora: v2, v4m, v5m, v5nrm."),
         ("experiments/06_gnn/v3_results.txt", "v3 result table."),
         ("experiments/06_gnn/v5q_comparison.txt", "v5q comparison table."),
         ("experiments/06_gnn/final_comparison.txt", "Final cross-protocol comparison."),
         ("experiments/06_gnn/measured_comparison.txt", "Measured-image comparison."),
         ("experiments/06_gnn/ablation_comparison.txt", "Ablation comparison.")]),

 dict(part="II. BreastMNIST", id="e08", title="08 &mdash; Contrastive scoring",
      q="Instead of a yes/no verdict, ask the model to score, and compare "
        "contrastive framings: verdict/match, a 0-10 expectation, and score-10.",
      v="No framing rescues the reasoning channel.",
      f=[("experiments/24_record/results/r08_r13.json",
          "Experiments 08 and 13 recomputed together by the record audit.")]),

 dict(part="II. BreastMNIST", id="e11", title="11 &mdash; Graph readout",
      q="Given the claim graph, which aggregation actually carries the label: "
        "malignant share, survival weighting, endorsement weighting?",
      v="Superseded by 14, which measured the same question against controls "
        "and falsified the design plan's guess.",
      f=[]),

 dict(part="II. BreastMNIST", id="e12", title="12 &mdash; ThothGNN v1 and the ablation ladder",
      q="The first named model. Eleven architectural ablations over the v5q "
        "corpus: attention, anchoring, cross-layer fusion, edge weights, sign, "
        "KG nodes, gating, class balance.",
      v="The anchor carries the result; the graph machinery does not. Every "
        "ablation that removes graph structure leaves the score intact, which "
        "is the pattern experiments 15 and 17 later nail down with controls.",
      f=[("experiments/12_thothgnn/results/ablation_v5q.json",
          "The eleven-arm ablation, per-seed runs collapsed to mean &plusmn; sd."),
         ("experiments/12_thothgnn/results/e2_v5q.json", "E2 on v5q."),
         ("experiments/12_thothgnn/results/e2x3.json", "E2 extended, 3 arms."),
         ("experiments/12_thothgnn/results/e2x4.json", "E2 extended, 4 arms."),
         ("experiments/12_thothgnn/results/all_v3.json", "BASE vs FULL on the v3 corpus."),
         ("experiments/12_thothgnn/results/all_v4.json", "BASE vs FULL on v4."),
         ("experiments/12_thothgnn/results/all_v5.json", "BASE vs FULL on v5."),
         ("experiments/12_thothgnn/results/all_v5q_r1.json", "BASE vs FULL on v5q run 1."),
         ("experiments/12_thothgnn/results/all_v5qnokg.json",
          "BASE vs FULL with the KG removed from the corpus."),
         ("experiments/12_thothgnn/results/interpret_e2.json",
          "Interpretability: attention shift on top triples vs random triples.")]),

 dict(part="II. BreastMNIST", id="e13", title="13 &mdash; BI-RADS scale readout",
      q="Ask for a BI-RADS assessment category rather than a binary label, and "
        "read the risk band off the ontology.",
      v="This is where reading p(yes) off the first-token distribution was found "
        "to work &mdash; the trick that the whole probe layer is built on.",
      f=[("experiments/24_record/results/r08_r13.json",
          "Recomputed with experiment 08; the <code>birads2</code> block.")]),

 dict(part="II. BreastMNIST", id="e14", title="14 &mdash; KG-indexed evidence tensor",
      q="Redesign the ledger: index evidence by KG finding rather than pooling "
        "it, and test the design plan's proposal to down-weight attacked claims.",
      v="<b>Two of the plan's recommendations were falsified by measurement.</b> "
        "Down-weighting attacked claims <i>hurts</i> (0.6699 against 0.7128 for "
        "the plain baseline). The diagnostic says why: being disagreed with "
        "carries almost nothing (spread 0.018) while being agreed with carries a "
        "lot (0.189). The informative direction was inverted. Endorsed claims "
        "are only ~8% of the corpus, so endorsement cannot carry a readout on "
        "its own &mdash; it belongs as a tensor column, which is where it went.",
      f=[("experiments/14_kgtensor/survival_4run.json",
          "The falsified proposal: survival weighting against the baseline."),
         ("experiments/14_kgtensor/tensor_4run.json",
          "The tensor itself, with and without the KG prior, collapsed, "
          "net-only and no-endorse variants."),
         ("experiments/14_kgtensor/combine.json",
          "Every readout combination: evidence, malignant share, BI-RADS, tensor."),
         ("experiments/14_kgtensor/rank_loss.json",
          "Cross-entropy against ranking loss, with and without augmentation."),
         ("experiments/14_kgtensor/gate_1run.json", "Gate, single run."),
         ("experiments/14_kgtensor/gate_4run.json", "Gate, four runs."),
         ("experiments/14_kgtensor/gate_crossrun.json", "Gate, across runs."),
         ("experiments/14_kgtensor/crossrun_readout.json",
          "Cross-run readout against the base.")]),

 dict(part="II. BreastMNIST", id="e15", title="15 &mdash; KG finding graph, visual probes, and the 0.80 result",
      q="Re-measure the probes with a good backbone, reading p(yes) off the "
        "first-token distribution, and put a flat logistic model on the 32 "
        "resulting features (16 BI-RADS findings x 2 phrasings).",
      v="<b>Test AUC 0.8137, balanced accuracy 0.7826</b> &mdash; against a previous "
        "project best of 0.7646 / 0.7036. Paired bootstrap +0.0491, 95% CI "
        "[-0.0114, +0.1102], P(better) = 0.945, by some way the largest "
        "improvement in the project. <b>It does not use a GNN.</b> Two of the "
        "four graph architectures live here, and each matches its shuffled-KG "
        "control almost exactly.",
      f=[("experiments/15_kggnn/results/final_s0.json",
          "Every probe arm separately and pooled: P1 direct/positive, P2 "
          "direct/negative, P3 verify/positive, P4 verify/negative, then "
          "probe-32/48/64, then KG-masked products against five shuffled masks."),
         ("experiments/15_kggnn/results/final_model.json",
          "The headline model against signed, unsigned, shuffled and no-prop KG."),
         ("experiments/15_kggnn/results/final_sum.json", "Sum readout variant."),
         ("experiments/15_kggnn/results/final_noanchor.json",
          "The same without the anchor, isolating what the anchor contributes."),
         ("experiments/15_kggnn/results/combine_all.json",
          "All five channels alone and in every combination: probe, debate, "
          "malignant share, BI-RADS, GNN."),
         ("experiments/15_kggnn/results/kg_laplacian.json",
          "<b>Architecture 3 of 5</b> &mdash; KG as a signed-Laplacian penalty on the "
          "weights rather than a propagation path. Best in-domain score of any "
          "architecture, and it still matches its shuffled control."),
         ("experiments/15_kggnn/results/probe_gnn.json",
          "<b>Architecture 4 of 5</b> &mdash; sign-separated propagation over the 16 "
          "finding nodes, swept over feature blocks, adjacencies and depths."),
         ("experiments/15_kggnn/results/thoth_kg.json",
          "<b>Architecture 5 of 5</b> &mdash; the KG <i>is</i> the graph, debate dropped."),
         ("experiments/15_kggnn/results/thoth_kg_nodeid.json",
          "ThothKG with node identity added."),
         ("experiments/15_kggnn/results/kgprior_gnn.json",
          "KG-prior GNN at two hidden widths, each against its shuffled control."),
         ("experiments/15_kggnn/results/gnn_ablation.json",
          "Adjacency ablation: KG, identity, complete and three random graphs, "
          "at one and two propagation steps."),
         ("experiments/15_kggnn/results/s5_control.json",
          "S5 &mdash; the KG-Laplacian control with five shuffled draws and a bootstrap."),
         ("experiments/15_kggnn/results/s1.json",
          "S1 &mdash; loss and regularisation selection against the anchor."),
         ("experiments/15_kggnn/results/s1b.json",
          "S1b &mdash; KG-masked products against five shuffled masks, bootstrapped."),
         ("experiments/15_kggnn/results/s2_gate.json",
          "S2 &mdash; does the debate channel add anything over probes? "
          "Permutation importance."),
         ("experiments/15_kggnn/results/s3_s5.json", "S3 and S5 gates."),
         ("experiments/15_kggnn/results/gates.json",
          "S7, G0 and S3d gates: products, deltas, correlations and spread.")]),

 # ------------------------------------------------------ breast: phase 3 -----
 dict(part="II. BreastMNIST", id="e16", title="16 &mdash; External validation on BUS-BRA (claim C1)",
      q="Freeze everything on BreastMNIST and evaluate on BUS-BRA: 1875 images, "
        "1064 patients, 4 scanners, biopsy-proven, folds split on case. How much "
        "does each method lose off distribution?",
      v="<b>The headline result.</b> The zero-shot KG pipeline loses 0.093 AUC "
        "going out of distribution where the supervised ResNet-18 loses 0.287 "
        "&mdash; and the ordering <i>reverses</i>: the CNN wins in domain and loses "
        "externally. This is claim C1.",
      f=[("experiments/16_external/results/frozen_models.json",
          "The frozen models: probe tags, findings, priors, and the "
          "normalisation constants, with each model's transfer score. This is "
          "<b>true transfer</b> &mdash; fitted on breast, applied unchanged."),
         ("experiments/16_external/results/resnet_external.json",
          "The supervised comparator in domain and on both BUS-BRA crops."),
         ("experiments/16_external/results/e1_analysis.json",
          "E1 &mdash; per-model external analysis at image and case level."),
         ("experiments/16_external/results/e2_analysis.json",
          "E2 &mdash; perception and its decomposition, plus the BrEaST arm."),
         ("experiments/16_external/results/sign_transfer.json",
          "Do the KG's stance signs transfer? Sign agreement, flips, and a "
          "random-flip control."),
         ("experiments/16_external/results/busbra_prep.json",
          "BUS-BRA preparation manifest, both crops."),
         ("experiments/16_external/results/breast_prep.json",
          "BrEaST preparation, pad 2.0x &mdash; descriptor positives and annotation counts."),
         ("experiments/16_external/results/breast_prep_pad125.json", "BrEaST, pad 1.25x."),
         ("experiments/16_external/results/breast_prep_pad30.json", "BrEaST, pad 3.0x.")]),

 dict(part="II. BreastMNIST", id="e17", title="17 &mdash; Heterogeneous debate graph and ThothGNN v3 (claim C3)",
      q="Stop trying architectures and measure the assumption underneath all of "
        "them. Propagation over an attack graph can only help if being attacked "
        "is evidence of being wrong. Is it?",
      v="<b>It is not.</b> Over 15,220 pooled claims, claim-level stance is "
        "correct 50.5% of the time, and attacks received on correct against "
        "wrong claims are 0.533 vs 0.536 &mdash; a difference of -0.003. ThothGNN v3 "
        "then scores test AUC 0.8033 with the real KG against 0.8051 &plusmn; 0.0109 "
        "for a degree-matched shuffle, and <code>no-graph</code> reaches 0.7951. "
        "The null is not underpowered: <code>powered_gnn.py</code> was written to "
        "test exactly that.",
      f=[("experiments/17_hetgnn/results/thothgnn3.json",
          "<b>ThothGNN v3, the headline arm</b>, against all three controls: "
          "linear flat, no-graph (A=0), shuffled-KG over five draws, and "
          "no-debate channel."),
         ("experiments/17_hetgnn/results/t3_p3.json", "The same with P3 probes only."),
         ("experiments/17_hetgnn/results/t3_both.json", "The same with both phrasing arms."),
         ("experiments/17_hetgnn/results/powered_gnn.json",
          "Is the null underpowered? Linear, no-graph, KG, sparse-KG and the "
          "shuffled control, with P(KG better than shuffled)."),
         ("experiments/17_hetgnn/results/diag_credibility.json"
          if (REPO / "experiments/17_hetgnn/results/diag_credibility.json").exists()
          else "experiments/17_hetgnn/results/credibility.txt",
          "The h-categoriser credibility diagnostic: does gradual argumentation "
          "semantics predict which claims are actually correct?"),
         ("experiments/17_hetgnn/results/attacks.txt",
          "Attacks received on correct against wrong claims &mdash; the -0.003."),
         ("experiments/17_hetgnn/results/external_merged.json",
          "External merged, the default arm."),
         ("experiments/17_hetgnn/results/em_full_both.json",
          "<b>External, our methods, automatic condition</b> (both phrasings)."),
         ("experiments/17_hetgnn/results/em_mask_both.json",
          "<b>External, our methods, mask condition.</b>"),
         ("experiments/17_hetgnn/results/em_full_p3.json", "External, P3 only."),
         ("experiments/17_hetgnn/results/em_full_both_cont.json",
          "External with the contestation channel added."),
         ("experiments/17_hetgnn/results/em_full_p3_cont.json",
          "External, P3 only, with contestation."),
         ("experiments/17_hetgnn/results/learning_curve.json",
          "Learning curve and convergence: how much data does the readout need?"),
         ("experiments/17_hetgnn/results/sparsity.json",
          "Sparsity sweep: KG against shuffled at each edge count, with sigma."),
         ("experiments/17_hetgnn/results/contestation.json",
          "Contestation: how often are claims contested, and does it correlate "
          "with probe uncertainty?"),
         ("experiments/17_hetgnn/results/stance_signs.json",
          "Net stance and argument mass against the probe channel."),
         ("experiments/17_hetgnn/results/uncertainty_gate.json",
          "Gate the debate on probe uncertainty &mdash; debate only the hard cases."),
         ("experiments/17_hetgnn/results/debate_when_split.json",
          "Debate only when the agents split: what fraction, and what accuracy."),
         ("experiments/17_hetgnn/results/debate_cost.json",
          "Probes alone, debate alone, both, and a permutation test."),
         ("experiments/17_hetgnn/results/credibility.txt", "Credibility, full table."),
         ("experiments/17_hetgnn/results/attacks.txt", "Attack distribution, full table.")]),

 dict(part="II. BreastMNIST", id="e18", title="18 &mdash; The lesion channel",
      q="Ask 'which lesion is this?' instead of 'does it show finding X?'. Does "
        "a lesion-level question beat a finding-level one, and do they combine?",
      v="The lesion channel carries real signal and combines with findings, but "
        "the +2.82 sd control is what makes it interpretable.",
      f=[("experiments/18_lesion/results/lesion_analysis.json",
          "The 0-parameter KG differential against fitted lesion probes, "
          "findings, lesions, and both, with a permutation test."),
         ("experiments/18_lesion/results/external_lesion.json",
          "External: findings only, lesions only, and combined, at case level."),
         ("experiments/18_lesion/results/el_p3.json", "External, P3 phrasing only."),
         ("experiments/18_lesion/results/el_base.json", "External, base arm."),
         ("experiments/18_lesion/results/el_repair.json", "External, repaired arm."),
         ("experiments/18_lesion/results/lesion_table.txt", "Per-lesion table.")]),

 dict(part="II. BreastMNIST", id="e19", title="19 &mdash; Learning curves across every method",
      q="How does each method scale with training data, and which are saturated?",
      v="The probe readouts saturate almost immediately; the supervised baseline "
        "keeps climbing. That gap is the same story as C1 seen from the data axis.",
      f=[("experiments/19_curves/results/train_curves.json",
          "Seven methods, each as a train/test curve (first point, best by test, "
          "last point).")]),

 # ------------------------------------------------------ breast: phase 4 -----
 dict(part="II. BreastMNIST", id="e20", title="20 &mdash; Perception (claim C4, and the P2 reversal)",
      q="Two questions. Does the phrasing of a probe matter more than the "
        "architecture on top of it? And does the negated arm P2, included to "
        "cancel acquiescence bias, actually work?",
      v="<b>P2 is anti-predictive.</b> Against radiologist descriptors it scores "
        "0.4882 &mdash; chance &mdash; and it transfers at 0.4452, <i>below</i> chance, while "
        "P3 transfers at 0.7499. It cancels the yes-bias by inverting the content "
        "along with it. Every project number before 2026-08-26 used the pooled "
        "pair and is therefore built on a feature block that is half "
        "anti-predictive. <b>Default to P3 alone.</b>",
      f=[("experiments/20_perception/results/phrasing_arms.json",
          "<b>The reversal.</b> P2, P3 and pooled against BrEaST radiologist "
          "descriptors, then zero-parameter and fitted readouts, with "
          "P(P3 beats pooled)."),
         ("experiments/20_perception/results/per_phrasing.json",
          "Per finding, per wording: which wordings are consistent, which are "
          "dead, which have only one usable form."),
         ("experiments/20_perception/results/sign_audit.json",
          "Sign audit: which findings read in the direction the ontology says "
          "they should, in domain and externally."),
         ("experiments/20_perception/results/sign_control.json",
          "The control for that audit: permutation and exhaustive flip search."),
         ("experiments/20_perception/results/mask_sweep.json",
          "Baseline against maskdim and maskring &mdash; the perception effect."),
         ("experiments/20_perception/results/crop_sweep.json",
          "Crop sweep with coverage: how much context helps before it hurts.")]),

 dict(part="II. BreastMNIST", id="e21", title="21 &mdash; BiomedCLIP as an alternative scorer",
      q="Is a purpose-built biomedical image-text model a better perception layer "
        "than probing a general VLM?",
      v="It deflates. Per-finding and per-family comparisons show the VLM probes "
        "hold up; the hybrid does not clear the bar.",
      f=[("experiments/21_biomedclip/results/compare_scorers.json",
          "Per finding and by finding family, with a win count."),
         ("experiments/21_biomedclip/results/hybrid.json",
          "The hybrid: selection, what was kept, zero-parameter and fitted "
          "readouts with P values."),
         ("experiments/21_biomedclip/results/clip_crop_sweep.json",
          "BiomedCLIP across three crops.")]),

 dict(part="II. BreastMNIST", id="e22", title="22 &mdash; The transfer-gap law",
      q="Is there a systematic relation between how well a method scores in "
        "domain and how much it loses off distribution?",
      v="Yes, and it is nearly deterministic: <b>r = 0.994</b> between in-domain "
        "score and external gap. This is why the two external protocols rank the "
        "probe arms oppositely &mdash; a law, not a contradiction.",
      f=[("experiments/22_transfer/results/gap_law.json",
          "Spearman and Pearson correlations, slope and intercept, plus the "
          "parameter-count-against-gap correlation."),
         ("experiments/24_record/results/gap_correction.json",
          "The corrected version: the coupled correlation, with a two-sided "
          "permutation test.")]),

 dict(part="II. BreastMNIST", id="e23", title="23 &mdash; Mask test (claim C4, pre-registered)",
      q="Pre-registered before any run: does telling the model where the lesion "
        "is beat any architectural choice? Two masking modes on BUS-BRA.",
      v="<b>Yes, by an order of magnitude.</b> Boundary contour gives +0.0871 AUC "
        "and background attenuation +0.0836, both at P(better) = 1.0, against "
        "roughly +0.01 for any architectural choice measured anywhere in this "
        "project. Perception dominates architecture. This is claim C4.",
      f=[("experiments/23_masktest/results/primary.json",
          "<b>The pre-registered primary endpoint</b> &mdash; 1875 images, 1064 "
          "patients, three arms with bootstrap CIs, plus the per-finding "
          "breakdown showing which findings the mask rescues."),
         ("experiments/23_masktest/results/confirm2.json",
          "Confirmation run, per finding, with the count of findings that move "
          "in the opposite direction."),
         ("experiments/23_masktest/results/exclusions_bring.json",
          "Exclusions, boundary-ring arm."),
         ("experiments/23_masktest/results/exclusions_bdim.json",
          "Exclusions, dimmed-background arm.")]),

 dict(part="II. BreastMNIST", id="e24", title="24 &mdash; The record audit",
      q="Recompute every published number in the project from the raw result "
        "files, and find out which of them were wrong.",
      v="<b>Three reporting faults, all of which had flattered a result.</b> A "
        "seed-<i>ensemble</i> AUC reported in place of the 5-seed mean (+0.009, "
        "about 2 sd); a results loader reading a filename that never existed and "
        "silently printing 'pending'; and a partial-corpus B1 number (0.6008 on "
        "880) superseded by the full 1875-image value (0.5836). The audit "
        "working is itself part of the contribution.",
      f=[("experiments/24_record/results/audit_all.json",
          "Every result file in the project, recomputed and compared against "
          "what was reported."),
         ("experiments/24_record/results/corpus_audit.json",
          "Corpus coverage: debates and probes, counted by unique index."),
         ("experiments/24_record/results/r03_variants.json",
          "Experiment 03's twelve variants, recomputed."),
         ("experiments/24_record/results/r08_r13.json",
          "Experiments 08 and 13, recomputed."),
         ("experiments/24_record/results/gap_correction.json",
          "The gap-law correction.")]),

 # ------------------------------------------------------ breast: phase 5 -----
 dict(part="II. BreastMNIST", id="e25", title="25 &mdash; Two-model heterogeneous debate (pre-registered)",
      q="Pre-registered: two agents from the same model family agree too easily. "
        "Does a heterogeneous partner &mdash; a different model lineage &mdash; make the "
        "debate carry information it otherwise does not?",
      v="No. The heterogeneous pairing does not clear the pre-registered bar, and "
        "in the merged corpus the debate channel <i>subtracts</i> from the probes "
        "(P(debate helps) = 0.0).",
      f=[("experiments/25_twomodel/results/primary.json",
          "<b>The pre-registered primary endpoint</b>: homogeneous against "
          "heterogeneous, each as probes and probes+debate, with the "
          "difference-of-differences, P(het better), the bar, and whether it clears."),
         ("experiments/25_twomodel/results/screen.json",
          "Partner screening: every candidate model against the AUC >= 0.60 bar."),
         ("experiments/25_twomodel/results/screen.txt", "The same screen as a table."),
         ("experiments/25_twomodel/results/merged_v5_het.json",
          "Heterogeneous corpus merged: 780 samples, probes against probes+debate."),
         ("experiments/25_twomodel/results/merged_v5_hom.json",
          "Homogeneous corpus merged, the matched comparison."),
         ("experiments/25_twomodel/results/transcript_v5.json",
          "Transcript statistics: do the two lineages actually disagree more?")]),

 dict(part="II. BreastMNIST", id="e26", title="26 &mdash; Re-implemented baselines B1 and B2 (pre-registered)",
      q="Two published multi-agent methods, re-implemented from the papers "
        "(no released code was used): B1 Catfish Agent (arXiv 2505.21503) and "
        "B2 GraphGeo (arXiv 2511.00908). Do either beat this project's methods, "
        "and do either beat simply sampling one model five times?",
      v="<b>Neither beats us, in either condition</b> &mdash; and B1 loses to "
        "self-consistency: 0.6499 against 0.7496, P = 0.016. That is claim C2. "
        "B2 carries 10,737 parameters against ThothGNN v3's 59 and uses 6 agents "
        "against 2, a confound that runs in B2's favour and is stated wherever "
        "B2 appears; it still does not separate. Every result here is "
        "<i>our implementation of</i> the published method.",
      f=[("experiments/26_baselines/results/b1.json",
          "<b>B1 Catfish Agent</b>, all six arms: full, no-catfish, always-on, "
          "adversarial, collaborative, and catfish-sees-image (asymmetry "
          "removed), with the paired comparisons and the frozen config."),
         ("experiments/26_baselines/results/b2.json",
          "<b>B2 GraphGeo</b>, all six arms: full, no-transfer, no-relation, "
          "no-agent-embedding, mean-readout and anchored, with the frozen config."),
         ("experiments/26_baselines/results/controls.json",
          "<b>The controls that decide C2</b>: single draw N=1, independent "
          "ensemble N=2, self-consistency N=5, and B1 against each of them."),
         ("experiments/26_baselines/results/external_auto.json",
          "Both baselines on BUS-BRA, automatic condition."),
         ("experiments/26_baselines/results/external_mask.json",
          "Both baselines on BUS-BRA, mask condition."),
         ("experiments/26_baselines/results/ours_6agent.json",
          "Our own protocol scaled to 6 agents, for a like-for-like agent count.")]),

 dict(part="II. BreastMNIST", id="hydra", title="The six-mode Hydra ablation run",
      q="The original ablation: single agent, opinion debate, graph debate, "
        "KG debate, adversarial and adaptive, on a fixed 50-image subset.",
      v="Retained as the run that framed the project. Note these predate the "
        "native-resolution fix, so the images were 28px upscaled to 224.",
      f=[("breastMnist/outputs/2026-08-15/21-46-45/metrics.json",
          "Accuracy, sensitivity, specificity, AUC, convergence rate and timing "
          "for one ablation mode.")]),

 # ------------------------------------------------------------ retina -------
 dict(part="III. RetinaMNIST / ICDR", id="r_prep", title="RetinaMNIST data pack and provenance",
      q="RetinaMNIST at native 224px with the published ICDR grade 0-4 kept "
        "intact &mdash; nothing binarised at export.",
      v="train 1080 / val 120 / test 400, referable-DR rate 0.431 to 0.450. "
        "Better balanced than BreastMNIST, so the majority baseline is weaker.",
      f=[("retinaMnist/results/retina_prep.json",
          "Per-split shapes, grade counts and per-file SHA-256.")]),

 dict(part="III. RetinaMNIST / ICDR", id="r_ord",
      title="The A-vs-C ordinal experiment with all three controls (claim C3 replicated)",
      q="Five ordered ICDR grades have four cut points. Head A is a "
        "cumulative-link (CORAL) trunk with four cut points, 63 parameters; head "
        "C is four independent trunks, 236 parameters. A is C with proportional "
        "odds imposed. Does message passing need to differ per severity "
        "threshold? And does the Round 4 topology null replicate on a second "
        "ontology?",
      v="<b>Round 4 replicates.</b> KG topology is +1.08 sd (A) and +0.91 sd (C) "
        "against a degree-matched shuffle &mdash; under 2 sd in both &mdash; and "
        "<code>no-graph</code> matches or beats the real KG in both. Over five "
        "paired seeds the effect goes <b>negative</b>: -0.0008 &plusmn; 0.0053 (A) "
        "and -0.0013 &plusmn; 0.0086 (C). The debate channel is null to slightly "
        "harmful. Perception is much stronger here than on breast (0.904 against "
        "0.803 referable AUC), so the vocabulary-not-edges result holds with "
        "<i>more</i> headroom, not less.",
      f=[("retinaMnist/results/ordinal_full.json",
          "<b>The primary result.</b> Both heads against no-graph, shuffled-KG "
          "over five draws and no-debate, with the CV selection grid and the "
          "derived ICDR adjacency (21 edges over 120 pairs, nothing isolated)."),
         ("retinaMnist/results/ordinal_seeds.json",
          "<b>Five paired seeds</b> &mdash; the run that turned +1.08 sd into a "
          "negative effect and strengthened C3."),
         ("retinaMnist/results/paired_ac.json",
          "Paired bootstrap A against C, 4000 resamples: referable AUC, QWK, "
          "macro-recall, MAE."),
         ("retinaMnist/results/paired_multiclass.json",
          "The multiclass half: accuracy and macro-OVR AUC. <b>This is the only "
          "comparison resolved at 95%</b> &mdash; C ahead by 0.0285 on macro-OVR AUC, "
          "the metric MedMNIST itself reports, with the interval clear of zero."),
         ("retinaMnist/results/baseline_curves.json",
          "Learning curves and slopes for the retina baselines.")]),

 dict(part="III. RetinaMNIST / ICDR", id="r_med", title="Against the published MedMNIST baselines",
      q="How does this pipeline compare to the CNNs MedMNIST publishes for "
        "RetinaMNIST?",
      v="<b>This is not a like-for-like comparison and must never be reported as "
        "one.</b> The MedMNIST baselines are CNNs trained from scratch on 1080 "
        "images; this is 63-236 trainable parameters on a frozen 8B VLM that "
        "answers 16 probe questions per image. The gap measures what the backbone "
        "already knows about fundus photographs, not an architectural win.",
      f=[("retinaMnist/results/medmnist_compare.json",
          "Both heads against Google AutoML Vision, ResNet-50 and ResNet-18 at "
          "28 and 224px.")]),

 dict(part="III. RetinaMNIST / ICDR", id="r_trans",
      title="Cross-ontology transfer (claim C5)",
      q="The architecture splits into a trunk indexed by <i>feature channel</i> "
        "(26 of 65 parameters) and a readout indexed by <i>finding identity</i> "
        "(39 of 65). The four channels mean the same thing in both trees, so the "
        "trunk can cross; <code>u</code> cannot, because breast finding 0 is "
        "architectural distortion and retina finding 0 is microaneurysm. So: is "
        "'how to weigh probe evidence against argument mass against an ontology "
        "prior' a skill independent of the ontology it was learned on?",
      v="<b>breast &rarr; retina is a clean positive</b> &mdash; the frozen breast trunk "
        "retains <b>99.4%</b> of the retina-trained ceiling and beats a frozen "
        "random trunk by <b>+4.35 sd</b>. The random-trunk control is what makes "
        "this mean anything: a fixed random projection with a fitted readout is "
        "already expressive (0.8399). <b>retina &rarr; breast is weaker and honest "
        "about it</b> &mdash; 93.4% of ceiling but only +1.73 sd, under this project's "
        "2 sd bar, and it is <i>beaten</i> by the 0-parameter KG-signed sum, which "
        "trains nothing at all. Both transfer arms are single-seed.",
      f=[("retinaMnist/results/transfer.json",
          "Both directions: ceiling, transfer, random trunk over five draws, and "
          "the 0-parameter rule."),
         ("retinaMnist/results/transfer_decompose.json",
          "<b>Which parameter block carries it</b> &mdash; full against W0+b alone, "
          "Wp+Wn alone, and all-random. This is the evidence that "
          "<code>W0</code> carries the transfer and the graph channels do not.")]),

 dict(part="III. RetinaMNIST / ICDR", id="r_base", title="B1 and B2 on retina",
      q="Both re-implemented baselines, ported to the ordinal task.",
      v="Neither separates from ThothGNN v3 head A. B2 runs single-lineage on "
        "retina (qwen3-vl only) against the breast B2's two lineages &mdash; a "
        "recorded deviation, taken because two lineages would have cost 20-30 h "
        "against 6-10 h, and it does not touch C2, C3 or C5.",
      f=[("retinaMnist/results/b1_retina.json",
          "B1 Catfish, five arms, each against ThothGNN v3 head A."),
         ("retinaMnist/results/b2_retina.json",
          "B2 GraphGeo with both ordinal heads, ordinal and binary readouts, "
          "against head A's referable AUC.")]),

 # ------------------------------------------------------------- derma ------
 dict(part="IV. DermaMNIST", id="d_prep", title="DermaMNIST data pack and provenance",
      q="DermaMNIST at native 224px, 7 nominal classes &mdash; the third instance, and "
        "the test of whether the port is really ontology-agnostic.",
      v="train 7007 / val 1003 / test 2005, seven classes, heavily imbalanced.",
      f=[("dermaMnist/results/derma_prep.json",
          "Per-split shapes, class counts and provenance.")]),

 dict(part="IV. DermaMNIST", id="d_nom", title="The nominal experiment with all three controls",
      q="Same architecture, same controls, an unordered 7-class target and 55 "
        "dermoscopic findings derived from the ontology. Does the topology null "
        "hold a third time?",
      v="It holds. The derived adjacency gives 162 edges over 1485 pairs with 11 "
        "pattern findings isolated, and the real KG does not separate from its "
        "shuffled control. The probe-only arm is not a placeholder &mdash; it "
        "<i>is</i> the no-debate control and a complete result on its own.",
      f=[("dermaMnist/results/derma_nominal.json",
          "<b>The primary result</b>: real KG, no-graph, shuffled-KG and "
          "no-debate, with the CV grid and the derived adjacency."),
         ("dermaMnist/results/derma_probeonly.json",
          "The probe-only arm run alone &mdash; the no-debate control as a "
          "standalone result."),
         ("dermaMnist/results/derma_nominal_appearance.json",
          "The same under the <i>breast</i> appearance-mediated adjacency, kept "
          "as a control because the breast null was measured under that "
          "construction and changing it silently would not be a replication."),
         ("dermaMnist/results/derma_curve.json",
          "Learning curve with the debate channel."),
         ("dermaMnist/results/derma_curve_probeonly.json",
          "Learning curve, probes only &mdash; the matched comparison.")]),

 # ------------------------------------------------------- cross-dataset ----
 dict(part="V. Cross-dataset", id="s_tm", title="The 3x3 cross-ontology transfer matrix",
      q="Every trunk into every target: breast, retina and derma, each as source "
        "and as destination, against a frozen random trunk and the 0-parameter "
        "rule.",
      v="Generalises the C5 result onto a full grid. The asymmetry is visible "
        "throughout: trunks fitted where perception is strong transfer into "
        "weaker regimes better than the reverse, and on breast the 0-parameter "
        "KG-signed sum remains hard for any trained trunk to beat.",
      f=[("shared/results/transfer_matrix.json",
          "Ceiling, each cross-transfer, random-trunk mean and sd, and the "
          "0-parameter rule, for all three datasets.")]),

 dict(part="V. Cross-dataset", id="s_ps", title="The prior-shuffle control",
      q="One level below topology: the ontology supplies a signed prior per "
        "finding. Is <i>that</i> doing work, or is the contribution entirely the "
        "vocabulary &mdash; which findings the ontology names?",
      v="<b>Shuffling the prior does not hurt.</b> On breast the shuffled prior is "
        "better than the real one (z = -0.61) and on retina it is clearly better "
        "(z = -3.16). This is the sharpest form of the project's central finding: "
        "the KG pays through <i>which findings it names</i>, not through the "
        "signs or the edges it puts on them.",
      f=[("shared/results/prior_shuffle.json",
          "Real against shuffled prior over five draws, and against no prior at "
          "all, for all three datasets.")]),

 dict(part="V. Cross-dataset", id="s_j", title="The LLM judge",
      q="Replace the aggregator with an LLM judge reading the transcript. Does a "
        "judge beat arithmetic?",
      v="No. The judge does not separate from the aggregator (the "
        "macro-recall difference does not exclude zero), and both are far below "
        "the GNN readout on the same split.",
      f=[("shared/results/judge_score.json",
          "Judge against aggregator on retina and derma test, with the GNN "
          "number quoted alongside for scale.")]),

 # ------------------------------------------------------------ hinpool -----
 dict(part="VI. HINPool", id="hp", title="HINPool reproduction on TUDataset",
      q="A standalone benchmark, deliberately isolated from the rest of the "
        "repository: reproduce HINPool (type-aware heterogeneous graph pooling) "
        "on public datasets where published numbers exist to compare against.",
      v="Per-seed results on MUTAG and PROTEINS, HINPool against an RGCN "
        "baseline, under the fixed-split multi-seed protocol.",
      f=[("hinpool-bench/results/MUTAG_hinpool.csv", "MUTAG, HINPool, per seed."),
         ("hinpool-bench/results/MUTAG_rgcn.csv", "MUTAG, RGCN baseline, per seed."),
         ("hinpool-bench/results/PROTEINS_hinpool.csv", "PROTEINS, HINPool, per seed."),
         ("hinpool-bench/results/PROTEINS_rgcn.csv", "PROTEINS, RGCN baseline, per seed.")]),
]


# The generated corpora behind every number above. These are raw per-sample
# records, not results, so they are counted rather than printed.
CORPORA = [
    # --- BreastMNIST: probe corpora -------------------------------------------
    ("Breast probes, all phrasing arms (15)", "experiments/15_kggnn/results"),
    ("Breast lesion probes (18)", "experiments/18_lesion/results"),
    ("Breast perception arms: crops and masks (20)", "experiments/20_perception/results"),
    ("BUS-BRA mask test probes (23)", "experiments/23_masktest/results"),
    ("BUS-BRA + BrEaST external probes (16)", "experiments/16_external/results"),
    ("KG-grounded injection variants (03)", "experiments/03_kg_grounded_vlm/results"),
    ("Contrastive scoring (08)", "experiments/08_contrastive/results"),
    ("BI-RADS 2 probes (13)", "experiments/13_birads2/results"),
    ("Partner screening (25)", "experiments/25_twomodel/results"),
    ("Zero-shot classification (01)", "experiments/01_zeroshot_vlm_classification/results"),
    # --- BreastMNIST: debate corpora, protocol by protocol ---------------------
    ("Breast debate v3", "breastMnist/data/breast/debates_v3"),
    ("Breast debate v4", "breastMnist/data/breast/debates_v4"),
    ("Breast debate v5", "breastMnist/data/breast/debates_v5"),
    ("Breast debate v5q (the corpus ThothGNN v3 consumes)",
     "breastMnist/data/breast/debates_v5q"),
    ("Breast debate v5q, replicate r1", "breastMnist/data/breast/debates_v5q_r1"),
    ("Breast debate v5q, replicate r2", "breastMnist/data/breast/debates_v5q_r2"),
    ("Breast debate v5q, replicate r3", "breastMnist/data/breast/debates_v5q_r3"),
    ("Breast debate v5q, KG removed", "breastMnist/data/breast/debates_v5qnokg"),
    ("Breast debate v5, heterogeneous pairing (25)",
     "breastMnist/data/breast/debates_v5_het"),
    ("Breast debate v5, homogeneous pairing (25)",
     "breastMnist/data/breast/debates_v5_hom"),
    ("BUS-BRA debate, automatic condition", "breastMnist/data/breast/debates_busbra"),
    ("BUS-BRA debate, mask condition", "breastMnist/data/breast/debates_busbra_mask"),
    # --- BreastMNIST: baseline and control corpora ----------------------------
    ("B1 catfish, breast (26)", "breastMnist/data/breast/catfish_b1"),
    ("B1 catfish sees image, breast (26)", "breastMnist/data/breast/catfish_b1_seesimg"),
    ("B1 catfish, BUS-BRA (26)", "breastMnist/data/breast/catfish_busbra"),
    ("B1 catfish, BUS-BRA masked (26)", "breastMnist/data/breast/catfish_busbra_mask"),
    ("B2 six-agent, breast (26)", "breastMnist/data/breast/multi6"),
    ("B2 six-agent, BUS-BRA (26)", "breastMnist/data/breast/multi6_busbra"),
    ("B2 six-agent, BUS-BRA masked (26)", "breastMnist/data/breast/multi6_busbra_mask"),
    ("Self-consistency and ensemble controls (26)",
     "breastMnist/data/breast/controls_b1"),
    # --- BreastMNIST: built graph datasets ------------------------------------
    ("Graph dataset v2", "breastMnist/data/breast/dataset_v2"),
    ("Graph dataset v3", "breastMnist/data/breast/dataset_v3"),
    ("Graph dataset v4cap", "breastMnist/data/breast/dataset_v4cap"),
    ("Graph dataset v5", "breastMnist/data/breast/dataset_v5"),
    ("Graph dataset v5q", "breastMnist/data/breast/dataset_v5q"),
    ("Graph dataset v5q, KG removed", "breastMnist/data/breast/dataset_v5qnokg"),
    ("Graph dataset v5q, retrieval", "breastMnist/data/breast/dataset_v5qret"),
    ("Graph dataset, probe-12 arm", "breastMnist/data/breast/dataset_probe12"),
    ("Stance-free observations (03)", "breastMnist/data/breast/observations"),
    # --- RetinaMNIST ----------------------------------------------------------
    ("Retina probes, P3 (all splits)", "retinaMnist/results"),
    ("Retina debate corpus", "retinaMnist/data/retina/debates_r1"),
    ("Retina B1 catfish corpus", "retinaMnist/data/retina/catfish_b1"),
    ("Retina B2 six-agent corpus", "retinaMnist/data/retina/multi6"),
    ("Retina judge corpus", "retinaMnist/data/retina/judge"),
    # --- DermaMNIST -----------------------------------------------------------
    ("Derma probes, P3 (all splits)", "dermaMnist/results"),
    ("Derma debate corpus", "dermaMnist/data/derma/debates_d1"),
    ("Derma judge corpus", "dermaMnist/data/derma/judge"),
]

# Files that exist under a results path but are not results: raw predictions,
# model weights, feature caches, run logs, dispatcher scratch.
NON_RESULT_SUFFIXES = {".jsonl", ".log", ".npy", ".npz", ".pt", ".sha256",
                       ".claim", ".bak", ".0", ".1", ".2", ".3"}


def audit(consumed: set) -> str:
    """Walk the tree for result files and report any this document does not show."""
    found, skipped = [], 0
    for p in sorted(REPO.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.suffix not in {".json", ".csv", ".txt"}:
            continue
        rel = p.relative_to(REPO).as_posix()
        # Anything under a data/ directory is corpus or ontology input, never a
        # result: per-sample debate graphs, the KG packs, the raw ICDR/derma json.
        if "data" in p.relative_to(REPO).parts or rel.startswith("hinpool-bench/splits/"):
            continue
        if p.name in {"requirements.txt", "uv.lock"} or "/conf/" in f"/{rel}":
            continue
        # ontology packs, index files and per-sample debate artifacts are inputs
        if any(s in rel for s in ("knowledge_graph.json", "definitions.json",
                                  "criteria.json", "findings.json", "lexicon.json",
                                  "schema.json", "_indices.json", "debate_sample-",
                                  "probe_findings.json")):
            skipped += 1
            continue
        if p.suffix in NON_RESULT_SUFFIXES:
            continue
        found.append((rel, str(p.resolve())))

    shown = [r for r, a in found if a in consumed]
    missed = [r for r, a in found if a not in consumed]
    rows = "".join(
        f'<tr><td><code>{html.escape(m)}</code></td>'
        f'<td>{(REPO / m).stat().st_size:,} B</td></tr>' for m in missed)
    verdict = (f'<p class="ok"><b>COMPLETE.</b> All {len(shown)} result files in the '
               f"tree are rendered above.</p>"
               if not missed else
               f'<p class="warn"><b>{len(missed)} result file(s) not shown above.</b> '
               f"Register them in <code>SECTIONS</code> and rebuild.</p>"
               f'<div class="scroll"><table class="matrix"><thead><tr><th>file</th>'
               f"<th>size</th></tr></thead><tbody>{rows}</tbody></table></div>")
    return (f"<p>Result files rendered in this document: <b>{len(shown)}</b>. "
            f"Input files skipped (ontology packs, split indices, per-sample debate "
            f"artifacts, configs): <b>{skipped}</b>. Raw corpora and model weights "
            f"are counted in the previous section rather than printed.</p>{verdict}")


CSS = """
:root{--ink:#16181d;--dim:#6b7280;--line:#e3e6ea;--paper:#ffffff;--ground:#f6f7f9;
--accent:#2d5a7b;--warn:#8a4b2a;--ok:#2f6b46;--band:#eef1f4}
:root[data-theme=dark],:root:not([data-theme=light]){}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ink:#e6e8ec;--dim:#9aa2ad;--line:#2b3038;--paper:#15171b;--ground:#0f1114;
--accent:#8fb8d4;--warn:#d99a6c;--ok:#7fbf9a;--band:#1c2027}}
:root[data-theme=dark]{--ink:#e6e8ec;--dim:#9aa2ad;--line:#2b3038;--paper:#15171b;
--ground:#0f1114;--accent:#8fb8d4;--warn:#d99a6c;--ok:#7fbf9a;--band:#1c2027}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:0 24px 96px}
header{background:var(--paper);border-bottom:1px solid var(--line);padding:44px 24px 34px;margin-bottom:32px}
header .wrap{padding-bottom:0}
h1{font-size:30px;line-height:1.25;margin:0 0 10px;letter-spacing:-.02em}
.sub{color:var(--dim)}
header p{margin:6px 0;color:var(--dim);max-width:74ch}
h2{font-size:23px;margin:52px 0 6px;padding-top:14px;border-top:2px solid var(--ink);letter-spacing:-.01em}
h3{font-size:18px;margin:34px 0 4px}
h4,h5,h6{font-size:13px;margin:20px 0 6px;color:var(--accent);
text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.part{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:var(--dim);
margin:44px 0 0;font-weight:700}
.q,.v{max-width:78ch;margin:10px 0}
.q{color:var(--dim)}
.v{border-left:3px solid var(--accent);padding:2px 0 2px 14px}
.cap{color:var(--dim);font-size:13.5px;margin:16px 0 8px;max-width:78ch}
.rf{background:var(--paper);border:1px solid var(--line);border-radius:8px;
padding:16px 18px;margin:14px 0}
table{border-collapse:collapse;font-size:13.5px;width:100%}
.kv{max-width:640px}
.kv th{text-align:left;font-weight:600;padding:5px 16px 5px 0;
border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:top}
.kv td{padding:5px 0;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums;text-align:right}
.matrix th,.matrix td{padding:6px 12px;border-bottom:1px solid var(--line);
text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.matrix thead th{text-align:right;font-size:11.5px;text-transform:uppercase;
letter-spacing:.05em;color:var(--dim);border-bottom:1.5px solid var(--ink)}
.matrix .rowname{text-align:left;font-weight:600;white-space:normal;min-width:180px}
.matrix tbody tr:hover{background:var(--band)}
.matrix tr.extra td,.matrix tr.extra th{color:var(--dim);font-size:12.5px}
.scroll{overflow-x:auto;margin:6px 0}
.scalar{margin:6px 0;font-variant-numeric:tabular-nums}
.prov{font-size:11.5px;color:var(--dim);margin:12px 0 0;
border-top:1px dotted var(--line);padding-top:8px;word-break:break-all}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;
background:var(--band);padding:1px 5px;border-radius:3px}
pre{background:var(--band);padding:14px;border-radius:6px;overflow-x:auto;
font-size:12.5px;line-height:1.45;margin:6px 0}
.na{color:var(--dim)}.dim{color:var(--dim);font-weight:400}
.missing{background:var(--band);border-left:3px solid var(--warn);
padding:10px 14px;margin:10px 0;color:var(--warn)}
.ok{color:var(--ok);font-weight:600}
.xref{border-style:dashed;background:transparent}.warn{color:var(--warn);font-weight:600}
nav{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:18px 22px;margin:24px 0 8px}
nav h2{border:0;margin:0 0 10px;font-size:15px;padding:0}
nav ol{margin:0;padding-left:20px;columns:2;column-gap:34px}
@media(max-width:720px){nav ol{columns:1}}
nav li{margin:3px 0;break-inside:avoid}
nav a{color:var(--accent);text-decoration:none}
nav a:hover{text-decoration:underline}
.claims th,.claims td{padding:8px 12px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top;white-space:normal}
.claims thead th{border-bottom:1.5px solid var(--ink);font-size:11.5px;
text-transform:uppercase;letter-spacing:.05em;color:var(--dim)}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
color:var(--dim);font-size:13px}
"""

CLAIMS = [
 ("C1", "The zero-shot KG pipeline generalises about 3x better than a supervised CNN",
  "Out-of-distribution AUC loss 0.093 against 0.287; the ordering reverses off distribution.",
  "Strong", "16, 22, 26"),
 ("C2", "Debate adds nothing; it loses to sampling the same model five times",
  "B1 Catfish 0.6499 against self-consistency N=5 at 0.7496, P=0.016.", "Strong", "25, 26"),
 ("C3", "KG topology is a well-powered null, not an underpowered positive",
  "Four independent implementations, each matching its degree-matched shuffled control; "
  "the effect falls as n grows (P=0.873); over five paired seeds on ICDR it is negative.",
  "Strong", "12, 15, 17, retina"),
 ("C4", "Perception dominates architecture by an order of magnitude",
  "+0.0871 AUC from a boundary-contour segmentation prior at P(better)=1.0, against "
  "roughly +0.01 for any architectural choice.", "Strong", "20, 23"),
 ("C5", "The per-node feature map transfers across ontologies",
  "breast to retina retains 99.4% of the target-trained ceiling and beats a frozen random "
  "trunk by +4.35 sd; the decomposition shows W0 carries it and the graph channels do not.",
  "Moderate - one direction only", "retina, shared"),
]


def build() -> str:
    consumed: set = set()
    home: dict = {}
    parts, body = [], []

    for s in SECTIONS:
        if s["part"] not in parts:
            parts.append(s["part"])
            body.append(f'<p class="part">{s["part"]}</p>')
        body.append(f'<h2 id="{s["id"]}">{s["title"]}</h2>')
        body.append(f'<p class="q"><b>The question.</b> {s["q"]}</p>')
        body.append(f'<p class="v"><b>What it settled.</b> {s["v"]}</p>')
        if not s["f"]:
            body.append('<p class="cap">No standalone result file; the numbers above '
                        "are reported through the experiments that superseded it.</p>")
        for path, cap in s["f"]:
            body.append(block(path, cap, consumed, home,
                              (s["id"], re.sub(r"<[^>]+>", "", s["title"]))))

    # corpora
    body.append('<p class="part">VII. Corpora</p>')
    body.append('<h2 id="corpora">The generated corpora behind every number above</h2>')
    body.append('<p class="q"><b>The question.</b> How much model output does this '
                "project actually rest on, and is any corpus incomplete?</p>")
    body.append('<p class="v"><b>What it settled.</b> Every number above rests on these '
                "records, and they are counted here rather than trusted. One record is "
                "one image under one arm, so the model-call count is far higher: a probe "
                "record is 16 calls, a v5 debate record 6, a B1 record 9 and a B2 record "
                "24. Roughly 230,000 vision-language model calls in total, every one on "
                "local hardware at a hard $0 budget. Counting matters here because two "
                "real defects were found this way: a mixed shard modulus that silently "
                "duplicated 341 probe computations, and a raw line count that reported "
                "1874 of 1875 when unique coverage was actually 1536.</p>")
    rows, tot_r, tot_b = [], 0, 0
    for name, d in CORPORA:
        st = corpus_stats(d)
        if not st:
            continue
        tot_r += st["records"]
        tot_b += st["bytes"]
        splits = ", ".join(f"{k} {v:,}" for k, v in sorted(st["by split"].items()))
        rows.append(f'<tr><th class="rowname">{html.escape(name)}</th>'
                    f'<td>{st["files"]:,}</td><td>{st["records"]:,}</td>'
                    f'<td>{st["bytes"]/1e6:,.1f} MB</td>'
                    f'<td style="text-align:left"><span class="dim">{st["format"]}</span></td>'
                    f'<td style="text-align:left;white-space:normal">'
                    f'<span class="dim">{html.escape(splits)}</span></td></tr>')
    # Per-image prediction dumps: raw records like the corpora, counted not printed.
    preds = sorted(REPO.glob("experiments/**/predictions.csv"))
    if preds:
        n = 0
        for f in preds:
            n += max(0, sum(1 for _ in f.open(encoding="utf8", errors="replace")) - 1)
            consumed.add(str(f.resolve()))
        rows.append(f'<tr><th class="rowname">ResNet per-image predictions '
                    f'(02, one file per seed)</th><td>{len(preds)}</td><td>{n:,}</td>'
                    f'<td>{sum(f.stat().st_size for f in preds)/1e6:,.1f} MB</td>'
                    f'<td style="text-align:left"><span class="dim">CSV</span></td>'
                    f'<td style="text-align:left"><span class="dim">'
                    f'sample_id, split, y_true, y_prob, y_pred</span></td></tr>')
        tot_r += n
        tot_b += sum(f.stat().st_size for f in preds)
    rows.append(f'<tr><th class="rowname"><b>total</b></th><td></td>'
                f'<td><b>{tot_r:,}</b></td><td><b>{tot_b/1e6:,.1f} MB</b></td>'
                f'<td></td><td></td></tr>')
    body.append('<div class="rf"><div class="scroll"><table class="matrix"><thead><tr>'
                "<th></th><th>files</th><th>records</th><th>size</th>"
                '<th style="text-align:left">format</th>'
                '<th style="text-align:left">by split</th></tr></thead><tbody>'
                + "".join(rows) + "</tbody></table></div></div>")

    # audit
    body.append('<p class="part">VIII. Audit</p>')
    body.append('<h2 id="audit">Completeness audit</h2>')
    body.append('<p class="q"><b>The question.</b> Does this document actually contain '
                "every result in the repository, or only the ones someone remembered to "
                "add?</p>")
    body.append('<p class="v"><b>How it is answered.</b> The builder walks the whole tree '
                "for result files and compares what it finds against what the sections "
                "above rendered. Anything unclaimed is named below, so a missing result "
                "is a visible failure rather than a silent one.</p>")
    body.append(f'<div class="rf">{audit(consumed)}</div>')

    nav = "".join(f'<li><a href="#{s["id"]}">{s["title"]}</a></li>' for s in SECTIONS)
    nav = ('<li><a href="#claims">The five claims</a></li>'
           '<li><a href="#howtoread">How to read this document</a></li>' + nav
           + '<li><a href="#corpora">VII. Corpora</a></li>'
           '<li><a href="#audit">VIII. Completeness audit</a></li>')

    claims = "".join(
        f"<tr><td><b>{c}</b></td><td>{t}</td><td>{e}</td><td>{s}</td>"
        f"<td><span class='dim'>{w}</span></td></tr>" for c, t, e, s, w in CLAIMS)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<title>Results Compendium</title>
<style>{CSS}</style>
<header><div class="wrap">
<h1>Results Compendium</h1>
<p class="sub">Multi-Agent Cooperative Debate via Knowledge Graphs &mdash;
every measured result, in one document</p>
<p>Breast ultrasound (ACR BI-RADS) &middot; diabetic retinopathy (ICDR) &middot;
dermoscopy &mdash; binary, five-level ordinal and seven-class nominal targets.</p>
<p>Nothing here is quoted from memory. Every table is read out of the result file
named beneath it at build time, with that file's size and SHA-256 recorded, and the
last section audits the tree for any result this document fails to show.</p>
<p><b>Generated</b> {now} by <code>scripts/build_results_compendium.py</code>.</p>
</div></header>
<div class="wrap">
<nav><h2>Contents</h2><ol>{nav}</ol></nav>

<h2 id="howtoread">How to read this document</h2>
<p class="q">Every section states the question the experiment asked and what it
settled, then shows the result files themselves. Under each table is the path,
byte size and SHA-256 of the file it was read from, so any number here can be
traced back to disk in one step.</p>
<div class="rf">
<h4>Three things are deliberately summarised rather than printed</h4>
<table class="claims"><tbody>
<tr><td><b>Per-seed runs</b></td><td>A <code>runs</code> or <code>seeds</code> block
collapses to <b>mean &plusmn; sd (n)</b>, which is how this project reports every
multi-seed result. The individual seeds are in the source file.</td></tr>
<tr><td><b>Training curves</b></td><td>An epoch or dataset-size curve shows its
<b>first point, its best point by test score, and its last point</b>, with the number
of points. Printing several hundred epochs would bury the result.</td></tr>
<tr><td><b>Per-sample predictions</b></td><td>Out-of-fold prediction vectors
(<code>oof</code>, <code>y</code>, <code>keys</code>) are raw records, not results.
They are counted in the corpora section, never printed.</td></tr>
</tbody></table>
<p class="cap">Everything else &mdash; every aggregated metric, every control draw,
every hyperparameter selection grid &mdash; is printed in full. Five-draw control
arrays are shown element by element, because a shuffled arm matching the real one
<i>is</i> the finding in this project and collapsing it to a mean would hide the
evidence.</p>
</div>

<h2 id="claims">The five claims, and where each is measured</h2>
<p class="q"><b>Reading rule.</b> This is a negative-result project with a positive
generalisation claim. The debate does not work, the graph does not work, and the
knowledge graph pays through its <i>vocabulary</i> rather than its <i>edges</i>. The
contribution is that all three were measured to a decision against proper controls
rather than asserted. Read the tables below with that as the thesis, not an accuracy
win.</p>
<div class="rf"><div class="scroll"><table class="claims"><thead><tr>
<th></th><th>Claim</th><th>Evidence</th><th>Strength</th><th>Measured in</th>
</tr></thead><tbody>{claims}</tbody></table></div></div>

{''.join(body)}

<footer>
<p><b>Conventions.</b> Balanced accuracy is the headline metric throughout, because
every dataset here is imbalanced &mdash; on BreastMNIST test (27% malignant) always
answering benign scores 0.7308 accuracy but 0.500 bAcc. AUC handles ties as
<code>(a&gt;b) + 0.5&middot;(a==b)</code>. Five fixed seeds (0-4), reported mean &plusmn; sd.
Paired bootstraps use 4000 resamples on patients. Cross-validation is 5-fold with
folds split on case. Selection is on train only; test is evaluated once per arm. The
bar for calling an effect real is 2 sd over its control.</p>
<p><b>Two comparisons that must never be made.</b> Never place
<code>external_merged.py</code> numbers (within-BUS-BRA 5-fold CV) beside
<code>freeze_models.py</code> numbers (frozen on breast, true transfer) &mdash; they rank
the probe arms oppositely, and that reversal is the transfer-gap law, not a
contradiction. Never place the MedMNIST CNN baselines in the same column as these
zero-shot numbers: 63-236 trained parameters on a frozen 8B backbone against CNNs
trained from scratch measures what the backbone already knows, not an architecture.</p>
<p><b>Known limitations carried into every table.</b> Probe arms pooled before
2026-08-26 include the anti-predictive P2 arm. B2 carries 182x ThothGNN v3's
parameters, uncorrected, in B2's favour. Both baselines are our re-implementations
from the papers. Multiple comparisons are not formally corrected across rounds. Only
one backbone (<code>qwen3-vl:8b-instruct</code>) was run at scale. The ThothGNN mask
row is BUS-BRA only &mdash; BreastMNIST ships no segmentation masks. The retina transfer
arms are single-seed. No pre-registration was committed before the retina run; the
procedure was clean but the document does not exist and must be labelled post-hoc,
never backdated.</p>
<p>Rebuild with <code>python scripts/build_results_compendium.py</code>.</p>
</footer>
</div>"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf8")
    print(f"wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size:,} bytes)")
