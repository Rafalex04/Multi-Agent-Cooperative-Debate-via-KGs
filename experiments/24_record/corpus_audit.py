"""Corpus and probe-run integrity appendix, recomputed from the files themselves."""
from __future__ import annotations
import json, glob, re, os
from pathlib import Path
import numpy as np
_H = Path(__file__).resolve().parent
_R = _H.parents[1]


def debates(root, label):
    fs = sorted(glob.glob(str(root / "*.json")))
    bad = empty = noclaim = 0
    cl, ed, ids = [], [], set()
    for f in fs:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            bad += 1; continue
        c = d.get("claims") or []
        if not c: noclaim += 1
        if c and not any((x.get("text") or "").strip() for x in c): empty += 1
        cl.append(len(c))
        ed.append(sum(len(x.get("addressed_ids") or []) for x in c))
        ids.add(d.get("sample_id"))
    r = {"label": label, "files": len(fs), "unparseable": bad, "zero_claim": noclaim,
         "all_text_empty": empty, "unique_sample_ids": len(ids),
         "claims_mean": float(np.mean(cl)) if cl else 0,
         "claims_min": int(min(cl)) if cl else 0, "claims_max": int(max(cl)) if cl else 0,
         "edges_mean": float(np.mean(ed)) if ed else 0}
    print(f"  {label:34s} files {r['files']:5d}  unparseable {bad}  zero-claim {noclaim}  "
          f"empty-text {empty}  uniq-ids {len(ids):5d}  claims {r['claims_mean']:.1f} "
          f"[{r['claims_min']}-{r['claims_max']}]  edges {r['edges_mean']:.1f}")
    return r


def probes(pat, label):
    n = nul = dup = 0; seen = set()
    for f in sorted(glob.glob(pat)):
        for ln in Path(f).read_text(errors="replace").splitlines():
            if not ln.strip(): continue
            try: r = json.loads(ln)
            except Exception: continue
            n += 1
            # key on the FULL tag: files differ by shard suffix only, and the two
            # phrasings share a directory, so a looser key invents duplicates
            k = (re.sub(r"_[0-9-]+\.jsonl$", "", os.path.basename(f)), r.get("index"))
            if k in seen: dup += 1
            seen.add(k)
            v = r.get("p_yes") or {}
            if v and sum(1 for x in v.values() if x is not None) == 0: nul += 1
    print(f"  {label:34s} records {n:6d}  all-null {nul}  duplicate-index {dup}")
    return {"label": label, "records": n, "all_null": nul, "duplicate_index": dup}


def main():
    out = {"debates": [], "probes": []}
    print("DEBATE CORPORA")
    D = _R / "breastMnist/data/breast"
    for sub, lab in (("debates_v5q", "BreastMNIST v5q (in-domain)"),
                     ("debates_busbra/all", "BUS-BRA (external)")):
        p = D / sub
        if p.exists():
            out["debates"].append(debates(p, lab))
        else:
            for q in sorted(D.glob("debates_v5*")):
                for r in sorted(q.glob("*")):
                    if r.is_dir():
                        out["debates"].append(debates(r, f"{q.name}/{r.name}"))
                break
    print("\nPROBE RUNS")
    E = _R / "experiments"
    for pat, lab in (
        (str(E / "15_kggnn/results/probeneg_*.jsonl"), "BreastMNIST P2 negated"),
        (str(E / "15_kggnn/results/probep3_*.jsonl"), "BreastMNIST P3 verification"),
        (str(E / "16_external/results/busbra_p2_*.jsonl"), "BUS-BRA P2 (archived)"),
        (str(E / "16_external/results/busbra_p3_*.jsonl"), "BUS-BRA P3 (archived)"),
        (str(E / "16_external/results/brst_p*.jsonl"), "BrEaST P2+P3"),
        (str(E / "20_perception/results/pad*_p*.jsonl"), "BrEaST crop sweep"),
        (str(E / "20_perception/results/mask*_p*.jsonl"), "BrEaST mask arms"),
        (str(E / "23_masktest/results/*_p3_*.jsonl"), "BUS-BRA mask arms (P3)"),
        (str(E / "23_masktest/results/bring_p2_*.jsonl"), "BUS-BRA maskring negated"),
        (str(E / "18_lesion/results/lesion_*.jsonl"), "BreastMNIST lesion probes"),
        (str(E / "18_lesion/results/leslbus_*.jsonl"), "BUS-BRA lesion probes"),
    ):
        if glob.glob(pat):
            out["probes"].append(probes(pat, lab))
    (_H / "results/corpus_audit.json").write_text(json.dumps(out, indent=1))
    print("\nwrote results/corpus_audit.json")


if __name__ == "__main__":
    main()
