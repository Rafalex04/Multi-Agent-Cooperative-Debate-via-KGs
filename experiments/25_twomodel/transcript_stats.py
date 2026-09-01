"""Secondary endpoints: properties of the transcript, not of the label.

These are the quantities each architecture's post-mortem named as its failure:

  v4  "not one claim in 1064 from the benign advocate ever conceded"  -> concessions
  v3  "743 of 6249 claims identical"                                  -> uniqueness
  v2  "both agents cite the same saturated features"                  -> citation overlap
  v5  "agents split on 44.4% of cases, worth 0.0000 AUC"              -> split rate

They are measured per corpus so HOM and HET are directly comparable, and they
do not require the debate to carry label signal - a debate can improve as a
debate while staying diagnostically inert, which is the outcome predicted in
the pre-registration.

  python transcript_stats.py --corpora hom=<dir> het=<dir> --out stats.json
"""
from __future__ import annotations

import argparse, glob, json
from collections import Counter
from pathlib import Path


def norm(t):
    return " ".join((t or "").lower().split())


def stats(root):
    files = sorted(glob.glob(str(Path(root) / "*" / "*.json")))
    n_deb = 0
    claims_tot = 0
    verdicts = Counter()
    per_agent_verdict = {}
    per_agent_label = {}
    texts = []
    cites = {}
    split_cases = 0
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        cl = d.get("claims") or []
        if not cl:
            continue
        n_deb += 1
        claims_tot += len(cl)
        opens = {}
        for c in cl:
            a = c.get("expert_id", "?")
            v = c.get("stance_verdict")
            verdicts[v] += 1
            per_agent_verdict.setdefault(a, Counter())[v] += 1
            per_agent_label.setdefault(a, Counter())[c.get("label")] += 1
            texts.append(norm(c.get("text")))
            for ft in (c.get("cited_features") or []):
                cites.setdefault(a, Counter())[ft] += 1
            if c.get("round_idx") == 0 and a not in opens:
                opens[a] = c.get("label")
        if len(opens) == 2 and len(set(opens.values())) == 2:
            split_cases += 1

    uniq = len(set(texts)) / max(1, len(texts))
    agree = verdicts["AGREE"]
    disagree = verdicts["DISAGREE"]
    out = {
        "debates": n_deb,
        "claims": claims_tot,
        "claims_per_debate": claims_tot / max(1, n_deb),
        "claim_uniqueness": uniq,
        "agree": agree, "disagree": disagree,
        "concession_rate": agree / max(1, agree + disagree),
        "opening_split_rate": split_cases / max(1, n_deb),
        "per_agent": {},
    }
    for a in sorted(per_agent_verdict):
        v = per_agent_verdict[a]; lb = per_agent_label[a]
        tot = v["AGREE"] + v["DISAGREE"]
        nl = sum(lb.values())
        out["per_agent"][a] = {
            "claims": nl,
            "agree": v["AGREE"], "disagree": v["DISAGREE"],
            "concession_rate": v["AGREE"] / max(1, tot),
            "p_malignant": lb.get("MALIGNANT", 0) / max(1, nl),
        }
    # how much do the two agents cite the SAME findings? (v2's failure)
    ags = sorted(cites)
    if len(ags) == 2:
        a, b = (Counter(cites[ags[0]]), Counter(cites[ags[1]]))
        ta, tb = sum(a.values()) or 1, sum(b.values()) or 1
        keys = set(a) | set(b)
        # Bhattacharyya overlap of the two citation distributions
        ov = sum(((a[k] / ta) * (b[k] / tb)) ** 0.5 for k in keys)
        out["citation_overlap"] = ov
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", required=True, help="name=dir ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = {}
    for spec in args.corpora:
        name, d = spec.split("=", 1)
        res[name] = stats(d)

    keys = [("debates", "{:d}"), ("claims", "{:d}"), ("claims_per_debate", "{:.2f}"),
            ("claim_uniqueness", "{:.4f}"), ("concession_rate", "{:.4f}"),
            ("opening_split_rate", "{:.4f}"), ("citation_overlap", "{:.4f}")]
    names = list(res)
    print(f"{'metric':22s}" + "".join(f"{n:>16s}" for n in names))
    for k, fmt in keys:
        row = f"{k:22s}"
        for n in names:
            v = res[n].get(k)
            row += f"{(fmt.format(v) if v is not None else '-'):>16s}"
        print(row)
    print()
    for n in names:
        print(f"{n}:")
        for a, s in res[n]["per_agent"].items():
            print(f"  {a:9s} claims {s['claims']:6d}  concession {s['concession_rate']:.4f}"
                  f"  p(claim=MALIGNANT) {s['p_malignant']:.4f}")
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
