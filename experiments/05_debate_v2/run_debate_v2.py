"""Evidence-grounded adversarial debate — dataset generator v2.

WHY v2 EXISTS
-------------
`dataset_full` (780 graphs) carries essentially no learnable signal:

  mal_share AUC = 0.5007   claim labels are uncorrelated with the gold label
  best feature AUC = 0.5475
  verdict balanced accuracy = 0.4916   worse than chance
  expert_gap = 0.997 for BOTH classes

The last number is the diagnosis. The two agents always took maximally opposite
positions regardless of what the image contained, so the debate was role-play,
not evidence. Compounding it, every graph was built from a 28x28 thumbnail
upscaled to 224 (the loader bug fixed in b47893f), so there was very little to
see in the first place.

WHAT CHANGES HERE
-----------------
1. Native 224x224 images.
2. A measurement phase runs BEFORE the debate. Each KG stance feature becomes a
   yes/no visual probe and p(yes) is read from first-token logprobs. On the full
   156-sample test split this evidence alone reaches AUC 0.6685, versus 0.6011
   for direct zero-shot classification. (An early 39-sample prefix suggested
   ~0.80; that did not survive the full split.)
3. Both agents argue from the SAME measured evidence table. They still take
   opposite sides, but each claim must cite a feature that was actually
   measured, so disagreement now tracks the image instead of the role.
4. Claim nodes carry the cited feature's probe probability and KG stance weight
   as continuous node attributes. This is what gives a GNN something to learn:
   node features that correlate with the label, rather than labels alone.

Graph JSON schema matches dataset_full so downstream loaders keep working;
claim nodes gain `cited_feature`, `p_yes` and `stance_weight`.

Usage:
  python run_debate_v2.py --split test --limit 12 --out-dir .../dataset_v2
  python run_debate_v2.py --split train --shard 0 --num-shards 4 --out-dir ...
"""
from __future__ import annotations

import argparse, base64, io, json, logging, math, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

from run_kg_featureprobe import build_probes, _p_yes, _b64  # noqa: E402

_LABEL_MAP = {0: "MALIGNANT", 1: "BENIGN"}

_CLAIM_SPLIT_RE = re.compile(r"\[CLAIM\]")
_ADDR_RE  = re.compile(r"\[ADDRESSED:\s*([^\]]+?)\s*\]\s*\[\s*(AGREE|DISAGREE)\s*\]", re.I)
_ADDR_ONLY_RE = re.compile(r"\[ADDRESSED:\s*(c\d+)\s*\]", re.I)
_LABEL_RE = re.compile(r"\[LABEL:\s*(BENIGN|MALIGNANT)\s*\]", re.I)
_CITED_RE = re.compile(r"\[CITED:\s*([^\]]+?)\s*\]", re.I)
_TAG_RE   = re.compile(r"\[(?:ADDRESSED:[^\]]*|AGREE|DISAGREE|LABEL:[^\]]*|CITED:[^\]]*)\]", re.I)


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image=None, num_predict=300, logprobs=False,
             num_ctx=4096, temperature=0.0, seed=None):
        msg = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [image]
        opts = {"temperature": temperature, "num_ctx": num_ctx,
                "num_predict": num_predict}
        if seed is not None:
            opts["seed"] = seed
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": opts}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 10
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


# --------------------------------------------------------------------------
# Phase 1 — measurement
# --------------------------------------------------------------------------

def measure(b64: str, probes: list[dict], client: Client, workers: int = 4) -> list[dict]:
    """Run every KG feature probe against the image, concurrently."""
    def one(pr):
        try:
            out = client.call(pr["question"], image=b64, num_predict=3,
                              logprobs=True, num_ctx=2048)
            lp = out.get("logprobs") or out["message"].get("logprobs")
            p  = _p_yes(lp)
        except Exception as exc:                      # network / decode hiccup
            logger.warning("probe %s failed: %s", pr["feature"], exc)
            p = None
        return {"feature": pr["feature"], "weight": pr["weight"], "p_yes": p}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, probes))


def evidence_score(ev: list[dict]) -> float | None:
    num = den = 0.0
    for e in ev:
        if e["p_yes"] is None:
            continue
        num += e["weight"] * e["p_yes"]
        den += abs(e["weight"])
    return None if den == 0 else (num / den + 1.0) / 2.0


def evidence_table(ev: list[dict], side: str | None = None) -> str:
    """Render measurements, split by whether they help or hurt `side`.

    Showing an agent which measurements cut against it is what stops the debate
    collapsing into role-play: it has to argue around real numbers rather than
    cite whatever feature matches its assigned label.
    """
    present = [e for e in ev if e["p_yes"] is not None and e["p_yes"] >= 0.5]
    absent  = [e for e in ev if e["p_yes"] is not None and e["p_yes"] < 0.5]

    def fmt(e):
        return f"- {e['feature']}: confidence {e['p_yes']:.2f}"

    if side is None:
        return ("PRESENT in this image:\n" +
                ("\n".join(fmt(e) for e in sorted(present, key=lambda x: -x["p_yes"]))
                 or "  (none)") +
                "\n\nABSENT from this image:\n" +
                ("\n".join(fmt(e) for e in sorted(absent, key=lambda x: x["p_yes"]))
                 or "  (none)"))

    want = 1.0 if side == "MALIGNANT" else -1.0
    helps  = [e for e in present if e["weight"] * want > 0]
    hurts  = [e for e in present if e["weight"] * want < 0]
    return ("MEASURED PRESENT and supporting your position:\n" +
            ("\n".join(fmt(e) for e in sorted(helps, key=lambda x: -x["p_yes"]))
             or "  (none — your position is not supported by the measurements)") +
            "\n\nMEASURED PRESENT but contradicting your position:\n" +
            ("\n".join(fmt(e) for e in sorted(hurts, key=lambda x: -x["p_yes"]))
             or "  (none)") +
            "\n\nMEASURED ABSENT (do not claim these are present):\n" +
            ("\n".join(f"- {e['feature']}: {e['p_yes']:.2f}"
                       for e in sorted(absent, key=lambda x: x["p_yes"])[:6]) or "  (none)"))


# --------------------------------------------------------------------------
# Phase 2 — debate
# --------------------------------------------------------------------------

_ROLE = {
    "expert_a": ("MALIGNANT",
                 "You argue this mass is MALIGNANT. Build the strongest honest case "
                 "for malignancy from the measured evidence."),
    "expert_b": ("BENIGN",
                 "You argue this mass is BENIGN. Build the strongest honest case "
                 "for a benign diagnosis from the measured evidence."),
}

_OPEN_FORMAT = """\
Reply with exactly 3 claims and nothing else:

[CLAIM] <one sentence naming one measured feature and what it implies> [LABEL: MALIGNANT]

Rules:
- Name exactly one feature per claim, and a different feature in each claim.
- Only call a feature present if it appears in a MEASURED PRESENT list above.
- Set LABEL to what that feature implies on its own, not to your assigned side.
  If a measurement contradicts your position, one claim must concede it and
  carry the opposite LABEL."""

_REBUT_FORMAT = """\
Reply with exactly 3 claims and nothing else. Each one answers a specific claim \
of your opponent:

[CLAIM] [ADDRESSED: c2] <your counter-argument in one sentence> [LABEL: MALIGNANT]

Rules:
- [ADDRESSED: cN] is required on every claim, using an id listed above.
- Write a NEW sentence. Never repeat or paraphrase back a sentence that already
  appears above, from either side.
- Explain why the measured confidence value does or does not support what your
  opponent concluded from it.
- Set LABEL to what the evidence implies, not to your assigned side."""


def opening_prompt(role: str, ev: list[dict]) -> str:
    side, instruction = _ROLE[role]
    return (f"You are a radiologist reviewing a breast ultrasound image.\n\n"
            f"{instruction}\n\n"
            f"Automated measurements of this image, as confidence 0-1 that each "
            f"feature is present:\n\n{evidence_table(ev, side)}\n\n"
            f"Look at the image and the measurements together.\n\n{_OPEN_FORMAT}")


def rebuttal_prompt(role: str, ev: list[dict], opponent: list[dict],
                    own: list[dict]) -> str:
    side, instruction = _ROLE[role]
    # Only the most recent round is shown, to keep the agent attacking something
    # fresh rather than restating its opening.
    last = max((c["round_idx"] for c in opponent), default=0)
    opp = "\n".join(f"[{c['node_id']}] {c['text']} -> they labelled this {c['label']}"
                    for c in opponent if c["round_idx"] == last) or "(nothing yet)"
    mine = "\n".join(f"- {c['text']}" for c in own) or "(nothing yet)"
    return (f"You are a radiologist reviewing a breast ultrasound image.\n\n"
            f"{instruction}\n\n"
            f"Measurements of this image:\n\n{evidence_table(ev, side)}\n\n"
            f"YOUR OPPONENT'S LATEST CLAIMS — respond to these by id:\n{opp}\n\n"
            f"Sentences you have ALREADY used (do not repeat any of them):\n{mine}\n\n"
            f"{_REBUT_FORMAT}")


_STOP = {"the", "a", "an", "is", "are", "of", "in", "and", "or", "with", "to",
         "this", "that", "it", "its", "mass", "lesion", "which", "was", "be",
         "on", "for", "as", "at", "by", "present", "shows", "appears", "seen"}


def _feature_tokens(feature: str) -> set[str]:
    """Content words identifying a feature, from its name and its phrasing."""
    from run_kg_featureprobe import _PHRASING
    words = set(feature.split("_"))
    phrase = _PHRASING.get(feature, "")
    # Only the phrasing's head clause names the feature; the rest explains it.
    words |= set(re.findall(r"[a-z]+", phrase.split(",")[0].lower()))
    return {w for w in words if w not in _STOP and len(w) > 2}


def match_feature(text: str, valid: set[str],
                  toks: dict[str, set[str]]) -> str | None:
    """Infer which measured feature a claim is about.

    A 4B model emits [CITED:] tags unreliably, so the citation is recovered from
    the prose instead. Exact name mentions win; otherwise the feature sharing the
    most content words with the sentence does, provided the overlap is decisive.
    """
    low = " " + re.sub(r"[^a-z_ ]", " ", text.lower()) + " "
    for f in sorted(valid, key=len, reverse=True):
        if f in low or " " + f.replace("_", " ") + " " in low:
            return f
    words = {w for w in re.findall(r"[a-z]+", low) if w not in _STOP and len(w) > 2}
    best, best_n = None, 0
    for f in valid:
        n = len(words & toks[f])
        if n > best_n:
            best, best_n = f, n
    # One shared content word is coincidence; two or more identifies the feature.
    return best if best_n >= 2 else None


def parse_claims(text: str, expert_id: str, round_idx: int,
                 counter: list[int], valid_features: set[str],
                 toks: dict[str, set[str]] | None = None) -> list[dict]:
    claims = []
    for part in _CLAIM_SPLIT_RE.split(text)[1:]:
        # Strip every bracket tag wherever it sits; the prose is what remains.
        # Splitting at the first tag would return nothing now that rebuttals
        # lead with [ADDRESSED: cN].
        body = " ".join(_TAG_RE.sub(" ", part).split()).strip()
        if len(body.split()) < 3:
            continue
        lm = _LABEL_RE.search(part)
        cm = _CITED_RE.search(part)
        cited = cm.group(1).strip().lower().replace(" ", "_") if cm else None
        if cited not in valid_features:
            cited = match_feature(body, valid_features, toks or {})
        counter[0] += 1
        claims.append({
            "node_id":   f"c{counter[0]}",
            "type":      "claim",
            "text":      body,
            "label":     lm.group(1).upper() if lm else None,
            "label_explicit": lm is not None,
            "expert_id": expert_id,
            "round_idx": round_idx,
            "cited_feature": cited,
            "addressed": [(a.strip(), b.upper())
                          for a, b in _ADDR_RE.findall(part)],
            "addressed_ids": [a.strip().lower()
                              for a in _ADDR_ONLY_RE.findall(part)],
        })
    return claims


def debate(b64: str, ev: list[dict], client: Client, rounds: int = 3,
           temperature: float = 0.4) -> list[dict]:
    valid = {e["feature"] for e in ev}
    toks = {f: _feature_tokens(f) for f in valid}
    counter = [0]
    by_expert: dict[str, list[dict]] = {"expert_a": [], "expert_b": []}
    # Global, not per-expert: agents were echoing the opponent's sentence back
    # verbatim, which manufactured edges without adding any argument.
    seen: set[str] = set()
    all_claims: list[dict] = []

    for r in range(rounds):
        for role in ("expert_a", "expert_b"):
            other = "expert_b" if role == "expert_a" else "expert_a"
            prompt = (opening_prompt(role, ev) if r == 0 else
                      rebuttal_prompt(role, ev, by_expert[other], by_expert[role]))
            try:
                # Greedy decoding made each round restate the previous one almost
                # verbatim. A little temperature, with a fixed per-turn seed to
                # keep runs reproducible, gives the rounds distinct content.
                out = client.call(prompt, image=b64, num_predict=320,
                                  temperature=temperature,
                                  seed=1000 * r + (0 if role == "expert_a" else 1))
                txt = out["message"]["content"]
            except Exception as exc:
                logger.warning("debate turn %s r%d failed: %s", role, r, exc)
                continue
            # Drop verbatim restatements so repeated turns cannot inflate the graph.
            fresh = []
            for c in parse_claims(txt, role, r, counter, valid, toks):
                key = re.sub(r"[^a-z ]", "", c["text"].lower()).strip()
                if key in seen:
                    continue
                seen.add(key)
                fresh.append(c)
            by_expert[role] += fresh
            all_claims += fresh
    return all_claims


# --------------------------------------------------------------------------
# Phase 3 — graph assembly
# --------------------------------------------------------------------------

def build_graph(sample_id: str, gold: str, claims: list[dict], ev: list[dict],
                triples: list[dict], rounds: int) -> dict:
    ev_by_feat = {e["feature"]: e for e in ev}
    triple_by_subj: dict[str, list[dict]] = {}
    for t in triples:
        triple_by_subj.setdefault(t.get("subject", ""), []).append(t)

    claim_nodes, used_triples = [], {}
    claim_triple_edges = []
    for c in claims:
        e = ev_by_feat.get(c["cited_feature"] or "", None)
        label = c["label"]
        # A missing LABEL tag is filled from the KG polarity of the feature the
        # claim is about, so an unparsed tag does not silently drop a node.
        if label is None and e is not None:
            label = "MALIGNANT" if e["weight"] > 0 else "BENIGN"
        claim_nodes.append({
            "node_id":   c["node_id"],
            "type":      "claim",
            "text":      c["text"],
            "label":     label,
            "label_int": {"MALIGNANT": 1, "BENIGN": 0}.get(label, -1),
            "label_explicit": c["label_explicit"],
            "expert_id": c["expert_id"],
            "round_idx": c["round_idx"],
            "cited_feature": c["cited_feature"],
            "p_yes":        e["p_yes"] if e else None,
            "stance_weight": e["weight"] if e else 0.0,
        })
        # Link the claim to every KG triple describing the feature it cited.
        for t in triple_by_subj.get(c["cited_feature"] or "", []):
            used_triples[t["id"]] = t
            claim_triple_edges.append(
                {"src": c["node_id"], "dst": t["id"], "type": "cited"})

    triple_nodes = [{
        "node_id": t["id"], "type": "triple",
        "subject": t.get("subject", ""), "predicate": t.get("relation", ""),
        "object": t.get("object", ""),
        "text": f"{t.get('subject','')} {t.get('relation','').replace('_',' ')} {t.get('object','')}".strip(),
    } for t in used_triples.values()]

    node_by_id = {c["node_id"]: c for c in claim_nodes}
    claim_claim, seen_edges = [], set()
    for c in claims:
        explicit = {a.strip().lower(): k for a, k in c["addressed"]}
        for tgt in set(c["addressed_ids"]) | set(explicit):
            if tgt not in node_by_id or tgt == c["node_id"]:
                continue
            if (c["node_id"], tgt) in seen_edges:
                continue
            seen_edges.add((c["node_id"], tgt))
            kind = explicit.get(tgt)
            if kind is None:
                # The model reliably emits [ADDRESSED:] but not [AGREE|DISAGREE];
                # the stance follows from whether the two claims share a label.
                src_lab = node_by_id[c["node_id"]]["label"]
                dst_lab = node_by_id[tgt]["label"]
                kind = "AGREE" if (src_lab and src_lab == dst_lab) else "DISAGREE"
            claim_claim.append({"src": c["node_id"], "dst": tgt,
                                "sign": "+" if kind == "AGREE" else "-",
                                "type": kind})

    triple_triple = []
    tids = list(used_triples)
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            ta, tb = used_triples[a], used_triples[b]
            if ta.get("subject") == tb.get("subject") or \
               ta.get("object") == tb.get("object"):
                triple_triple.append({"src": a, "dst": b, "type": "shared_entity"})

    score = evidence_score(ev)
    # No judge: the verdict is the measured evidence, with the debate's final
    # round breaking ties when the evidence sits near the decision boundary.
    last = [c for c in claim_nodes if c["round_idx"] == rounds - 1]
    lab = [c["label"] for c in last if c["label"] in ("MALIGNANT", "BENIGN")]
    debate_share = (sum(1 for l in lab if l == "MALIGNANT") / len(lab)) if lab else 0.5
    verdict = None if score is None else ("MALIGNANT" if score >= 0.5 else "BENIGN")

    return {
        "sample_id": sample_id,
        "gold_label": gold,
        "gold_label_int": 1 if gold == "MALIGNANT" else 0,
        "verdict": verdict,
        "correct": verdict == gold,
        "rounds_used": rounds,
        "evidence": ev,
        "evidence_score": score,
        "debate_mal_share": debate_share,
        "nodes": {"claims": claim_nodes, "triples": triple_nodes},
        "edges": {"claim_claim": claim_claim,
                  "claim_triple": claim_triple_edges,
                  "triple_triple": triple_triple},
        "stats": {
            "num_claims": len(claim_nodes),
            "num_triples": len(triple_nodes),
            "num_claim_claim_edges": len(claim_claim),
            "num_claim_triple_edges": len(claim_triple_edges),
            "num_triple_triple_edges": len(triple_triple),
            "num_cited": sum(1 for c in claim_nodes if c["cited_feature"]),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="medgemma:4b")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--rounds",     type=int, default=3)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--shard",      type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--reverse", action="store_true",
                   help="walk samples last-to-first; lets a fast sweeper fill "
                        "gaps from the far end instead of racing slow workers")
    p.add_argument("--workers",    type=int, default=4, help="concurrent probes")
    p.add_argument("--temperature", type=float, default=0.4,
                   help="debate sampling temperature; probes stay greedy")
    p.add_argument("--url",        default="http://localhost:11434/api/chat")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--kg-root",
        default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np

    root    = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    probes  = build_probes(triples, schema)

    # Pre-exported at native 224px so worker nodes need only numpy + PIL.
    npz = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = npz["imgs"], npz["labels"]
    n  = len(imgs) if args.limit is None else min(args.limit, len(imgs))
    idxs = [i for i in range(n) if i % args.num_shards == args.shard]
    if args.reverse:
        idxs.reverse()

    out_dir = Path(args.out_dir) / args.split / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = Client(args.url, args.model)
    logger.info("debate v2 | split=%s shard=%d/%d | %d samples | %d probes | %d rounds",
                args.split, args.shard, args.num_shards, len(idxs), len(probes), args.rounds)

    t0 = time.time()
    for k, idx in enumerate(idxs, 1):
        sid  = f"{idx:03d}"
        dest = out_dir / f"sample_{sid}.json"
        if dest.exists():
            continue
        gold = _LABEL_MAP[int(labels[idx][0])]
        b64  = _b64(imgs[idx], args.image_size)

        ts = time.time()
        ev = measure(b64, probes, client, workers=args.workers)
        claims = debate(b64, ev, client, rounds=args.rounds,
                        temperature=args.temperature)
        g = build_graph(sid, gold, claims, ev, triples, args.rounds)
        # Write-then-rename: sweeper workers may race on the same sample once
        # the fast nodes start filling gaps left by the slow ones, and a
        # half-written graph would be worse than a duplicated one.
        tmp = dest.with_suffix(f".{args.shard}.tmp")
        tmp.write_text(json.dumps(g, indent=1))
        tmp.replace(dest)

        el = time.time() - ts
        rate = (time.time() - t0) / k
        logger.info("  [%3d/%3d] %s gold=%-9s verdict=%-9s score=%s | "
                    "%d claims (%d cited) %d cc-edges | %.0fs | ETA %.1fh",
                    k, len(idxs), sid, gold, g["verdict"],
                    f"{g['evidence_score']:.3f}" if g["evidence_score"] else "n/a",
                    g["stats"]["num_claims"], g["stats"]["num_cited"],
                    g["stats"]["num_claim_claim_edges"], el,
                    rate * (len(idxs) - k) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
