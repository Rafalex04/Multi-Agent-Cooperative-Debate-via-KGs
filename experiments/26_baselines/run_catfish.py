"""B1 - our implementation of the Catfish Agent (Wang et al., arXiv 2505.21503).

Multi-agent medical debate with an injected dissenting agent. No graph, no GNN.
Re-implemented from the paper's description; the released code is not used.

Four roles on one backbone. A and B see the image and the 16 BI-RADS findings,
independently shuffled. The CATFISH SEES THE TRANSCRIPT ONLY, never the image -
the information-asymmetry condition. Without it the Catfish is a third observer,
which is the redundancy that made our own debate channel harmful. A Moderator
consolidates.

GENERATION vs GATING. The complexity-aware trigger is not applied here. Every
case gets a Catfish round, and the trigger's INPUTS are recorded per sample
(silent_agreement, mean_conf). The gate is then applied at analysis time, which
is equivalent, lets tau_conf be cross-validated without regenerating the corpus,
and makes `B1-always-on` fall out of the same data. Both tones are generated
from a SHARED rounds-0/1 prefix, so tone costs one extra branch, not a re-run.

  python run_catfish.py --split test --out-dir .../catfish --num-shards 8 --shard 0
"""
from __future__ import annotations

import argparse, json, logging, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "10_debate_v5"))
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

from run_debate_v5 import (                                      # noqa: E402
    Client, _LABEL_MAP, _b64, _mark_repeat, _norm, all_findings,
    opening_prompt, parse_claims, rebuttal_prompt, shuffled,
)

TONES = ("collaborative", "adversarial")

_TONE_TEXT = {
    # the paper's point is that pure adversarial framing over-corrects
    "collaborative":
        "Identify the claims that are least supported by what a radiologist "
        "could actually see, and explain what additional evidence would be "
        "needed to settle each one.",
    "adversarial":
        "Argue against the emerging consensus. Attack the position the two "
        "radiologists are converging on.",
}


def catfish_prompt(claims, tone, sees_image=False):
    """The Catfish receives the transcript only, unless the asymmetry ablation
    is on (`--catfish-sees-image`), in which case it also gets the image and is
    told so - otherwise the prompt would contradict the input."""
    lines = "\n".join(f"  {c['node_id']} | {c['label']} | {c['text']}" for c in claims)
    mal = sum(1 for c in claims if c["label"] == "MALIGNANT")
    lean = ("MALIGNANT" if mal > len(claims) / 2 else
            "BENIGN" if mal < len(claims) / 2 else "SPLIT")
    head = ("You are a senior radiologist reviewing a written case discussion "
            "between two colleagues about a breast ultrasound. You have NOT seen "
            "the image. You are reviewing only what they wrote."
            if not sees_image else
            "You are a senior radiologist joining a case discussion between two "
            "colleagues about a breast ultrasound. The image is attached and you "
            "can examine it yourself.")
    tail = ("- You cannot see the image, so never assert what the image shows. "
            "Challenge the\n  REASONING and the confidence, not the pixels."
            if not sees_image else
            "- Examine the image yourself and challenge claims it does not support.")
    return f"""\
{head}

Two radiologists have been debating. Their claims so far:

{lines}

Their discussion is leaning: {lean}

{_TONE_TEXT[tone]}

Premature agreement is the failure you exist to catch. Reply with between 2 and \
4 lines and nothing else, each with three fields separated by | :

  their claim id | CHALLENGE | what is weak about that claim

Rules:
{tail}
- Pick the claims carrying the most weight in their conclusion.
- One sentence per line. Do not repeat their wording."""


def parse_catfish(text, round_idx, counter):
    out = []
    for ln in (text or "").splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 3 or "CHALLENGE" not in parts[1].upper():
            continue
        counter[0] += 1
        out.append({"node_id": f"c{counter[0]}", "text": parts[2],
                    "target_id": parts[0], "expert_id": "catfish",
                    "round_idx": round_idx, "phase": "catfish",
                    "label": None, "stance_verdict": "DISAGREE",
                    "addressed_ids": [parts[0]], "cited_features": []})
    return out


def catfish_reply_prompt(kit, challenges, own):
    ch = "\n".join(f"  {c['node_id']} | {c['text']}" for c in challenges) or "  (none)"
    mine = "\n".join(f"  - {c['text']}" for c in own) or "  (nothing yet)"
    from run_debate_v5 import _block
    ids = [c["node_id"] for c in challenges] or ["c1", "c2"]
    ex = "\n".join(f"  {i} | {v} | f{k+1} | {l} | {r}" for k, (i, v, l, r) in
                   enumerate(zip((ids + ids)[:2], ("DISAGREE", "AGREE"),
                                 ("MALIGNANT", "BENIGN"),
                                 ("what they missed in the image",
                                  "why their challenge is fair"))))
    return f"""\
You are a radiologist examining a breast ultrasound image, taking part in a \
debate about whether this mass is malignant or benign.

A senior colleague who has NOT seen the image has challenged the discussion:

{ch}

They cannot see the image. You can. Answer their challenge by looking again at \
the image: either show what they missed, or concede the point if they are right.

Findings that matter for this decision:
{_block(kit)}

Sentences you have already used, do not repeat any:
{mine}

Reply with between 2 and 4 lines and nothing else, each with five fields \
separated by | :

  their challenge id | AGREE or DISAGREE | finding tag | MALIGNANT or BENIGN | your reason

Worked examples of the format only:

{ex}

Rules:
- AGREE means you concede their challenge. Concede when they are right.
- Start every line with the challenge id you are answering.
- Point at the image. One sentence per line, no repeats."""


def moderator_prompt(claims):
    lines = "\n".join(f"  {c['node_id']} | {c['label'] or '-'} | {c['text']}"
                      for c in claims)
    return f"""\
You are the moderator of a radiology case discussion about a breast ultrasound. \
You have not seen the image. Below is every claim made.

{lines}

Decide, for each claim that describes an observation of the image, what that \
observation implies ON ITS OWN. Drop claims that are pure procedural challenge \
and make no observation.

Reply with one line per surviving claim and nothing else, two fields separated \
by | :

  claim id | MALIGNANT or BENIGN

Rules:
- Judge each observation independently. Mixed evidence is normal.
- Do not add claims. Do not explain."""


def parse_moderator(text, valid):
    out = {}
    for ln in (text or "").splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 2:
            continue
        cid, lab = parts[0], parts[1].upper()
        if cid in valid and lab in ("MALIGNANT", "BENIGN"):
            out[cid] = lab
    return out


def run_one(b64, clients, kits, tag2feats, temperature, run_id, conf,
            catfish_sees_image=False):
    """rounds 0-1 shared; then a Catfish branch per tone."""
    counter = [0]
    by_agent = {"agent_1": [], "agent_2": []}
    seen = {_norm(d) for k in kits.values() for _, _, d in k}
    base = []
    for r in (0, 1):
        for agent in ("agent_1", "agent_2"):
            other = "agent_2" if agent == "agent_1" else "agent_1"
            kit, tag2feat = kits[agent], tag2feats[agent]
            cl = clients[agent]
            prompt = (opening_prompt(kit) if r == 0 else
                      rebuttal_prompt(kit, by_agent[other][-3:], by_agent[agent]))
            try:
                txt = cl.call(prompt, image=b64, temperature=temperature,
                              seed=100000 * run_id + 1000 * r
                                   + (0 if agent == "agent_1" else 1)
                              )["message"]["content"]
            except Exception as exc:
                logger.warning("base %s r%d failed: %s", agent, r, exc)
                continue
            fresh = [c for c in parse_claims(txt, agent, r, counter, tag2feat,
                                             max_claims=4 if r == 0 else 3)
                     if not _mark_repeat(c, seen)]
            for c in fresh:
                c["phase"] = "base"; c["model"] = cl.model
            by_agent[agent] += fresh
            base += fresh

    # trigger inputs, recorded not applied
    o1 = next((c["label"] for c in base
               if c["expert_id"] == "agent_1" and c["round_idx"] == 0), None)
    o2 = next((c["label"] for c in base
               if c["expert_id"] == "agent_2" and c["round_idx"] == 0), None)
    trigger = {"silent_agreement": bool(o1 and o2 and o1 == o2),
               "opening_1": o1, "opening_2": o2, "mean_conf": conf}

    branches = {}
    for tone in TONES:
        bc = [dict(c) for c in base]
        cnt = [counter[0]]
        try:
            ctxt = clients["catfish"].call(
                catfish_prompt(base, tone, catfish_sees_image),
                image=b64 if catfish_sees_image else None,
                temperature=temperature,
                seed=100000 * run_id + 7001)["message"]["content"]
        except Exception as exc:
            logger.warning("catfish(%s) failed: %s", tone, exc)
            ctxt = ""
        ch = parse_catfish(ctxt, 2, cnt)
        for c in ch:
            c["model"] = clients["catfish"].model
        resp = []
        for agent in ("agent_1", "agent_2"):
            if not ch:
                break
            cl = clients[agent]
            try:
                txt = cl.call(catfish_reply_prompt(kits[agent], ch, by_agent[agent]),
                              image=b64, temperature=temperature,
                              seed=100000 * run_id + 8000
                                   + (0 if agent == "agent_1" else 1)
                              )["message"]["content"]
            except Exception as exc:
                logger.warning("reply %s failed: %s", agent, exc)
                continue
            fresh = parse_claims(txt, agent, 3, cnt, tag2feats[agent], max_claims=3)
            for c in fresh:
                c["phase"] = "response"; c["model"] = cl.model
            resp += fresh
        allc = bc + ch + resp
        valid = {c["node_id"] for c in allc}
        try:
            mtxt = clients["moderator"].call(moderator_prompt(allc), image=None,
                                             temperature=0.0,
                                             seed=100000 * run_id + 9001
                                             )["message"]["content"]
        except Exception as exc:
            logger.warning("moderator(%s) failed: %s", tone, exc)
            mtxt = ""
        branches[tone] = {"catfish": ch, "response": resp,
                          "moderator": parse_moderator(mtxt, valid)}
    return base, trigger, branches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    p.add_argument("--split", default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--catfish-sees-image", action="store_true",
                   help="ablation: remove the information asymmetry")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--kg-root", default=str(_HERE.parents[2] / "breastMnist"))
    p.add_argument("--npz", default=None)
    p.add_argument("--probe-tag", default="probep3")
    args = p.parse_args()

    import numpy as np
    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema = json.loads((root / "data/breast/schema.json").read_text())
    findings = all_findings(triples, schema)
    names = [f for f, _, _ in findings]

    # probe confidence per image -> the trigger's ambiguity term
    conf = {}
    pd = _HERE.parents[1] / "15_kggnn/results"
    for f in sorted(pd.glob(f"{args.probe_tag}_{args.split}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            v = r.get("p_yes") or {}
            vals = [float(v[n]) for n in names if v.get(n) is not None]
            if vals:
                conf[int(r["index"])] = float(np.mean([abs(2 * x - 1) for x in vals]))

    z = np.load(Path(args.npz) if args.npz
                else root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards == args.shard]

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    cl = Client(args.url, args.model)
    clients = {"agent_1": cl, "agent_2": cl, "catfish": cl, "moderator": cl}
    logger.info("B1 catfish | split=%s shard=%d/%d | %d samples | asymmetry=%s",
                args.split, args.shard, args.num_shards, len(idxs),
                "OFF" if args.catfish_sees_image else "ON")

    t0 = time.time()
    width = max(3, len(str(len(imgs) - 1)))
    for k, idx in enumerate(idxs, 1):
        sid = f"{idx:0{width}d}"
        dest = out_dir / f"debate_{sid}.json"
        if dest.exists():
            continue
        b64 = _b64(imgs[idx], args.image_size)
        k1, t1 = shuffled(findings, sid, 2 * args.run_id)
        k2, t2 = shuffled(findings, sid, 2 * args.run_id + 1)
        base, trig, branches = run_one(
            b64, clients, {"agent_1": k1, "agent_2": k2},
            {"agent_1": t1, "agent_2": t2}, args.temperature, args.run_id,
            conf.get(idx), catfish_sees_image=args.catfish_sees_image)
        tmp = dest.with_suffix(f".{args.shard}.tmp")
        tmp.write_text(json.dumps({
            "sample_id": sid, "gold_label": _LABEL_MAP[int(labels[idx][0])],
            "model": args.model, "asymmetry": not args.catfish_sees_image,
            "trigger": trig, "base_claims": base, "branches": branches,
        }, indent=1))
        tmp.replace(dest)
        nb = len(base)
        nc = len(branches["collaborative"]["catfish"])
        logger.info("  [%3d/%3d] %s %-9s | base %2d | catfish %d | silent=%s | %.0fs",
                    k, len(idxs), sid, _LABEL_MAP[int(labels[idx][0])], nb, nc,
                    trig["silent_agreement"], (time.time() - t0) / k)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
