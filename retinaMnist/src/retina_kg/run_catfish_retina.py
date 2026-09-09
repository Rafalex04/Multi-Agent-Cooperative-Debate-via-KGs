"""B1 - Catfish Agent (arXiv 2505.21503) on RetinaMNIST / ICDR.

Our implementation from the paper, ported from `26_baselines/run_catfish.py`.
Label every result "our implementation of Catfish Agent".

Protocol is unchanged from the breast version: four roles (agent_1, agent_2,
catfish, moderator); rounds 0-1 shared; a catfish branch per tone; the catfish
sees the TRANSCRIPT ONLY -- that information asymmetry is the paper's mechanism
and `--catfish-sees-image` is the ablation that removes it. The complexity
trigger is recorded at generation and gated at analysis, so `tau_conf` is
cross-validatable without regenerating and `B1-always-on` falls out of the same
corpus.

What changes for a five-level ordinal scale, exactly as in run_debate_retina:
a claim's stance is an ICDR GRADE 0-4, not MALIGNANT/BENIGN, and the discussion's
"lean" is a mean grade rather than a majority side. The moderator assigns a grade
per surviving claim.

Output is JSONL through corpus_io -- one file per (split, shard), not one JSON
per image, for the inode quota.

  python run_catfish_retina.py --split test --num-shards 8 --shard 0
"""
from __future__ import annotations

import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("10_debate_v5", "26_baselines")

from run_debate_v5 import Client, _b64, _mark_repeat, _norm, shuffled  # noqa: E402
from corpus_io import append, done_indices                            # noqa: E402
from run_debate_retina import (                                       # noqa: E402
    _GRADE_RE, _SCALE, opening_prompt, parse_claims, rebuttal_prompt, _block,
)

TONES = ("collaborative", "adversarial")

_TONE_TEXT = {
    # the paper's point is that pure adversarial framing over-corrects
    "collaborative":
        "Identify the claims that are least supported by what an ophthalmologist "
        "could actually see on a fundus photograph, and explain what additional "
        "evidence would be needed to settle each one.",
    "adversarial":
        "Argue against the emerging consensus. Attack the position the two "
        "ophthalmologists are converging on.",
}


def _lean(claims):
    g = [c["grade"] for c in claims if c.get("grade") is not None]
    return f"mean ICDR grade {np.mean(g):.1f}" if g else "no grade stated"


def catfish_prompt(claims, tone, sees_image=False):
    lines = "\n".join(f"  {c['node_id']} | {c['grade']} | {c['text']}" for c in claims)
    head = ("You are a senior ophthalmologist reviewing a written case discussion "
            "between two colleagues about a colour fundus photograph. You have NOT "
            "seen the image. You are reviewing only what they wrote."
            if not sees_image else
            "You are a senior ophthalmologist joining a case discussion between two "
            "colleagues about a colour fundus photograph. The image is attached and "
            "you can examine it yourself.")
    tail = ("- You cannot see the image, so never assert what the image shows. "
            "Challenge the\n  REASONING and the confidence, not the pixels."
            if not sees_image else
            "- Examine the image yourself and challenge claims it does not support.")
    return f"""\
{head}

{_SCALE}

Two ophthalmologists have been debating. Their claims so far:

{lines}

Their discussion is leaning: {_lean(claims)}

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
                    "grade": None, "stance_verdict": "DISAGREE",
                    "addressed_ids": [parts[0]], "cited_features": []})
    return out


def catfish_reply_prompt(kit, challenges, own):
    ch = "\n".join(f"  {c['node_id']} | {c['text']}" for c in challenges) or "  (none)"
    mine = "\n".join(f"  - {c['text']}" for c in own) or "  (nothing yet)"
    ids = [c["node_id"] for c in challenges] or ["c1", "c2"]
    ex = "\n".join(f"  {i} | {v} | f{k+1} | {g} | {r}" for k, (i, v, g, r) in
                   enumerate(zip((ids + ids)[:2], ("DISAGREE", "AGREE"), (3, 1),
                                 ("what they missed in the image",
                                  "why their challenge is fair"))))
    return f"""\
You are an ophthalmologist examining a colour fundus photograph, taking part in a \
debate about how severe this eye's diabetic retinopathy is on the ICDR scale.

A senior colleague who has NOT seen the image has challenged the discussion:

{ch}

They cannot see the image. You can. Answer their challenge by looking again at \
the image: either show what they missed, or concede the point if they are right.

{_SCALE}

Findings that matter for this decision:
{_block(kit)}

Sentences you have already used, do not repeat any:
{mine}

Reply with between 2 and 4 lines and nothing else, each with five fields \
separated by | :

  their challenge id | AGREE or DISAGREE | finding tag | ICDR grade 0-4 | your reason

Worked examples of the format only:

{ex}

Rules:
- AGREE means you concede their challenge. Concede when they are right.
- Start every line with the challenge id you are answering.
- Point at the image. One sentence per line, no repeats."""


def moderator_prompt(claims):
    lines = "\n".join(f"  {c['node_id']} | {c.get('grade') if c.get('grade') is not None else '-'} | {c['text']}"
                      for c in claims)
    return f"""\
You are the moderator of an ophthalmology case discussion about a colour fundus \
photograph. You have not seen the image. Below is every claim made.

{_SCALE}

{lines}

Decide, for each claim that describes an observation of the image, what ICDR \
grade that observation implies ON ITS OWN. Drop claims that are pure procedural \
challenge and make no observation.

Reply with one line per surviving claim and nothing else, two fields separated \
by | :

  claim id | ICDR grade 0-4

Rules:
- Judge each observation independently. Mixed evidence is normal.
- Do not add claims. Do not explain."""


def parse_moderator(text, valid):
    out = {}
    for ln in (text or "").splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 2:
            continue
        cid = parts[0]
        m = _GRADE_RE.search(parts[1])
        if cid in valid and m:
            out[cid] = int(m.group(1))
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

    g1 = next((c["grade"] for c in base
               if c["expert_id"] == "agent_1" and c["round_idx"] == 0), None)
    g2 = next((c["grade"] for c in base
               if c["expert_id"] == "agent_2" and c["round_idx"] == 0), None)
    trigger = {"silent_agreement": bool(g1 is not None and g1 == g2),
               "opening_1": g1, "opening_2": g2, "mean_conf": conf}

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
    p.add_argument("--pack", default=str(P.PACK))
    p.add_argument("--npz", default=None)
    p.add_argument("--probe-tag", default="retina_p3")
    p.add_argument("--probe-dir", default=str(P.RESULTS))
    p.add_argument("--out-dir", default=str(P.PACK / "catfish_b1"))
    args = p.parse_args()

    pack = Path(args.pack)
    findings = [tuple(x) for x in
                json.loads((pack / "probe_findings.json").read_text())]
    names = [f for f, _, _ in findings]

    # probe confidence per image -> the trigger's ambiguity term
    conf = {}
    for f in sorted(Path(args.probe_dir).glob(f"{args.probe_tag}_{args.split}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            v = r.get("p_yes") or {}
            vals = [float(v[n]) for n in names if v.get(n) is not None]
            if vals:
                conf[int(r["index"])] = float(np.mean([abs(2 * x - 1) for x in vals]))

    z = np.load(Path(args.npz) if args.npz else pack / f"images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards == args.shard]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    done = done_indices(out_dir, args.split)
    todo = [i for i in idxs if i not in done]
    cl = Client(args.url, args.model)
    clients = {"agent_1": cl, "agent_2": cl, "catfish": cl, "moderator": cl}
    logger.info("B1 catfish retina | %s shard %d/%d | %d samples (%d done) | asymmetry=%s",
                args.split, args.shard, args.num_shards, len(todo), len(done),
                "OFF" if args.catfish_sees_image else "ON")

    t0 = time.time()
    for k, idx in enumerate(todo, 1):
        sid, gold, ts = idx, int(labels[idx][0]), time.time()
        b64 = _b64(imgs[idx], args.image_size)
        k1, t1 = shuffled(findings, idx, 2 * args.run_id)
        k2, t2 = shuffled(findings, idx, 2 * args.run_id + 1)
        base, trig, branches = run_one(
            b64, clients, {"agent_1": k1, "agent_2": k2},
            {"agent_1": t1, "agent_2": t2}, args.temperature, args.run_id,
            conf.get(idx), catfish_sees_image=args.catfish_sees_image)
        append(out_dir, args.split, args.shard, {
            "sample_id": sid, "gold_grade": gold, "model": args.model,
            "asymmetry": not args.catfish_sees_image,
            "trigger": trig, "base_claims": base, "branches": branches,
        })
        rate = (time.time() - t0) / k
        logger.info("  [%3d/%3d] %04d gold=%d | base %2d | catfish %d | %.0fs | ETA %.1fh",
                    k, len(todo), idx, gold, len(base),
                    len(branches["collaborative"]["catfish"]),
                    time.time() - ts, rate * (len(todo) - k) / 3600)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
