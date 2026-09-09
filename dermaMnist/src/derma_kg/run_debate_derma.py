"""Open-stance two-agent debate over dermoscopic findings, on DermaMNIST.

Protocol is v5's, unchanged from breast and retina: no assigned side, both agents
see the image and the same shuffled finding kit, explicit AGREE/DISAGREE on a
named claim id, three rounds, claims capped per turn. Transport, image encoding,
tag shuffling and repeat marking are IMPORTED from run_debate_v5 so only what is
genuinely dataset-specific lives here.

WHAT CHANGES FOR A NOMINAL LABEL SPACE. Retina's claims carried an ICDR grade
0-4, an ordered scale, so "net stance" was a position on a ladder. DermaMNIST has
SEVEN UNORDERED classes -- melanoma is not "more" than dermatofibroma -- so a
claim carries a CLASS NAME and the finding-node channel becomes a per-class
argument mass vector rather than a signed scalar. There is no meaningful
"direction" to lean in.

Output is JSONL through corpus_io, one file per (split, shard).

  python run_debate_derma.py --split test --num-shards 32 --shard 0
"""
from __future__ import annotations

import argparse, json, logging, re, sys, time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("10_debate_v5", "26_baselines")

from run_debate_v5 import Client, _b64, _mark_repeat, _norm, shuffled  # noqa: E402
from corpus_io import append, done_indices                            # noqa: E402

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
_CLASS_RE = re.compile(r"\b(akiec|bcc|bkl|df|mel|nv|vasc)\b", re.I)

_HEAD = ("You are a dermatologist examining a dermoscopic image, taking part in "
         "a discussion about which lesion class it shows.")

_SCALE = """\
The seven possible classes are:
  akiec  actinic keratosis / intraepithelial carcinoma
  bcc    basal cell carcinoma
  bkl    benign keratosis-like lesion
  df     dermatofibroma
  mel    melanoma
  nv     melanocytic nevus
  vasc   vascular lesion"""

_EXAMPLES = (
    "a fine regular brown grid covering most of the lesion with a paler centre",
    "several large branching vessels running across a pearly pink background",
    "sharply bordered red-purple round pools separated by pale walls",
)


def _block(kit):
    return "\n".join(f"[{tag}] {desc}" for tag, _, desc in kit)


def opening_prompt(kit, no_kg=False):
    if no_kg:
        return f"""\
{_HEAD}

You have not been assigned a class. Decide for yourself from the image.

{_SCALE}

Report ONLY what you can actually see in this image. Reply with between 2 and 4 \
lines and nothing else, each with two fields separated by | :

  class code | what you actually see in this image

Worked examples of the format only - write your own observations:

  nv | {_EXAMPLES[0]}
  bcc | {_EXAMPLES[1]}
  vasc | {_EXAMPLES[2]}

Rules:
- Describe a different observation on each line.
- The first field is the class that ONE observation points to on its own, not
  your overall diagnosis. Mixed evidence is normal and expected.
- One sentence per line."""
    return f"""\
{_HEAD}

You have not been assigned a class. Decide for yourself from the image.

{_SCALE}

These are the dermoscopic findings that matter for this decision, each described \
by how it looks under dermoscopy:
{_block(kit)}

Check them against the image and report ONLY the ones you can actually see. \
Reply with between 2 and 4 lines and nothing else, each with three fields \
separated by | :

  finding tag | class code | what you actually see in this image

Worked examples of the format only - write your own observations:

  f1 | nv | {_EXAMPLES[0]}
  f2 | bcc | {_EXAMPLES[1]}
  f3 | vasc | {_EXAMPLES[2]}

Rules:
- Never copy a finding's wording. Say what THIS image looks like.
- The middle field is the class that ONE observation points to on its own, not
  your overall diagnosis. Mixed evidence is normal and expected.
- Use a different finding tag on each line. One sentence per line."""


def rebuttal_prompt(kit, opponent, own, no_kg=False):
    last = max((c["round_idx"] for c in opponent), default=0)
    recent = [c for c in opponent if c["round_idx"] == last]
    opp = "\n".join(f"  {c['node_id']} | {c['label']} | {c['text']}"
                    for c in recent) or "  (nothing yet)"
    mine = "\n".join(f"  - {c['text']}" for c in own) or "  (nothing yet)"
    ids = [c["node_id"] for c in recent] or ["c1", "c2"]
    reasons = ("what they missed or misread in the image",
               "why you cannot honestly dispute this one")
    pairs = list(zip((ids + ids)[:2], ("DISAGREE", "AGREE"), ("mel", "nv"), reasons))
    if no_kg:
        ex = "\n".join(f"  {cid} | {v} | {lab} | {t}" for cid, v, lab, t in pairs)
        fields = ("four fields",
                  "their claim id | AGREE or DISAGREE | class code | your reason")
        kg_block = ""
    else:
        ex = "\n".join(f"  {cid} | {v} | f{i+1} | {lab} | {t}"
                       for i, (cid, v, lab, t) in enumerate(pairs))
        fields = ("five fields",
                  "their claim id | AGREE or DISAGREE | finding tag | class code | your reason")
        kg_block = f"\nFindings that matter for this decision:\n{_block(kit)}\n"
    return f"""\
{_HEAD}

This is a DISCUSSION. You are expected to CHALLENGE what the other dermatologist \
claims. Look at the image again and find what they got wrong, overstated, or \
missed. Only AGREE with a claim when you genuinely cannot find any grounds to \
dispute it.

{_SCALE}
{kg_block}
THE OTHER DERMATOLOGIST JUST CLAIMED:
{opp}

YOU PREVIOUSLY SAID:
{mine}

Reply with between 2 and 4 lines and nothing else, each with {fields[0]} \
separated by | :

  {fields[1]}

Worked examples of the format only - write your own:

{ex}

Rules:
- Address a specific claim id on every line.
- The class field is what YOUR observation points to on its own.
- One sentence per line."""


def parse_claims(text, agent, round_idx, counter, tag2feat, max_claims=4, no_kg=False):
    """Same field grammar as v5, with a CLASS CODE where the label was."""
    out = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        if "|" not in line:
            continue
        parts = [x.strip() for x in line.split("|")]
        addressed, verdict = [], None
        if parts and re.fullmatch(r"c\d+", parts[0], re.I):
            addressed = [parts[0].lower()]; parts = parts[1:]
        if parts and re.fullmatch(r"(AGREE|DISAGREE)", parts[0], re.I):
            verdict = parts[0].upper(); parts = parts[1:]
        if no_kg:
            if len(parts) >= 3 and not _CLASS_RE.fullmatch(parts[0].strip()):
                parts = parts[1:]
            if len(parts) < 2:
                continue
            tag, lab, body = "", parts[0], " | ".join(parts[1:])
        else:
            if len(parts) < 3:
                continue
            tag, lab, body = parts[0], parts[1], " | ".join(parts[2:])
        feat = ([tag2feat[tag.strip().lower()]]
                if tag.strip().lower() in tag2feat else [])
        m = _CLASS_RE.search(lab)
        label = m.group(1).lower() if m else None

        body = re.sub(r"\b(AGREE|DISAGREE)\b\s*$", "", body, flags=re.I)
        body = " ".join(body.split()).strip(" .|") + "."
        if len(body.split()) < 4:
            continue
        if len(out) >= max_claims:
            break
        counter[0] += 1
        out.append({"node_id": f"c{counter[0]}", "text": body, "label": label,
                    "label_explicit": label is not None, "expert_id": agent,
                    "round_idx": round_idx, "cited_features": feat,
                    "addressed_ids": addressed, "stance_verdict": verdict,
                    "is_repeat": False})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--split", default="test")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    p.add_argument("--pack", default=str(P.PACK))
    p.add_argument("--npz", default=None)
    p.add_argument("--no-kg", action="store_true")
    p.add_argument("--out-dir", default=str(P.PACK / "debates_d1"))
    args = p.parse_args()

    pack = Path(args.pack)
    findings = [(f[0], f[1]) for f in
                json.loads((pack / "probe_findings.json").read_text())]
    z = np.load(Path(args.npz) if args.npz else pack / f"images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards == args.shard]
    done = done_indices(args.out_dir, args.split)
    todo = [i for i in idxs if i not in done]

    cl = Client(args.url, args.model)
    logger.info("derma debate | %s shard %d/%d | %d findings | %d samples (%d done)",
                args.split, args.shard, args.num_shards, len(findings),
                len(todo), len(done))

    t0 = time.time()
    for k, idx in enumerate(todo, 1):
        gold, ts = int(labels[idx][0]), time.time()
        b64 = _b64(imgs[idx], args.image_size)
        kit_a, map_a = shuffled([(f, d, "") for f, d in findings], idx, 0)
        kit_b, map_b = shuffled([(f, d, "") for f, d in findings], idx, 1)
        counter, claims = [0], []
        seen = ({_norm(d) for _, _, d in kit_a} | {_norm(d) for _, _, d in kit_b}
                | {_norm(x) for x in _EXAMPLES})
        by = {"agent_1": [], "agent_2": []}

        for r in range(args.rounds):
            for agent, kit, tmap in (("agent_1", kit_a, map_a),
                                     ("agent_2", kit_b, map_b)):
                other = "agent_2" if agent == "agent_1" else "agent_1"
                prompt = (opening_prompt(kit, args.no_kg) if r == 0 else
                          rebuttal_prompt(kit, by[other], by[agent], args.no_kg))
                try:
                    txt = cl.call(prompt, b64, num_predict=360)["message"]["content"]
                except Exception as e:
                    logger.warning("  %04d r%d %s failed: %s", idx, r, agent, e)
                    continue
                got = parse_claims(txt, agent, r, counter,
                                   {t.lower(): f for t, f, _ in kit},
                                   no_kg=args.no_kg)
                for c in [x for x in got if not _mark_repeat(x, seen)]:
                    claims.append(c); by[agent].append(c)

        append(args.out_dir, args.split, args.shard,
               {"sample_id": idx, "gold_class": gold,
                "gold_code": CLASSES[gold], "rounds_used": args.rounds,
                "model": args.model, "claims": claims})
        labs = [c["label"] for c in claims if c["label"]]
        top = max(set(labs), key=labs.count) if labs else "-"
        rate = (time.time() - t0) / k
        logger.info("  [%4d/%4d] %04d gold=%-5s | %2d claims | modal %-5s | %.0fs | ETA %.1fh",
                    k, len(todo), idx, CLASSES[gold], len(claims), top,
                    time.time() - ts, rate * (len(todo) - k) / 3600)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, args.out_dir)


if __name__ == "__main__":
    main()
