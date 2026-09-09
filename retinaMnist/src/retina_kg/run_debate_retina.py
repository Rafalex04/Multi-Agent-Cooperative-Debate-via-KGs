"""Open-stance two-agent debate over ICDR findings, on RetinaMNIST.

The protocol is v5's, unchanged: no assigned side, both agents see the image and
the same shuffled finding kit, explicit AGREE/DISAGREE on a named claim id, three
rounds, claims capped at four per turn. The transport, image encoding, tag
shuffling and repeat marking are IMPORTED from run_debate_v5 rather than copied,
so only what is genuinely ordinal lives here.

What changes for a five-level scale:

  * a claim's middle field is an ICDR GRADE 0-4, not MALIGNANT/BENIGN. It is what
    that one observation implies on its own, which is what makes `net stance` a
    position on the severity ladder rather than a sign.
  * the finding kit carries no polarity. On breast every finding was pre-labelled
    as pointing malignant or benign; here saying "microaneurysm means grade 1"
    inside the prompt would hand the agent the answer, so the kit lists only what
    each finding LOOKS like and the agent supplies the grade.

Output is JSONL, one file per (split, shard), through corpus_io -- not one JSON
per image. 1600 retina samples at the old layout is 1600 inodes against a home
quota with roughly 4,900 free.

  python run_debate_retina.py --split test --num-shards 16 --shard 0
"""
from __future__ import annotations

import argparse, json, logging, random, re, sys, time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("10_debate_v5", "03_kg_grounded_vlm", "26_baselines")

from run_debate_v5 import Client, _b64, _mark_repeat, _norm, shuffled  # noqa: E402
from corpus_io import append, done_indices                            # noqa: E402

GRADES = (0, 1, 2, 3, 4)
_GRADE_RE = re.compile(r"\b([0-4])\b")

_HEAD = ("You are an ophthalmologist examining a colour fundus photograph, "
         "taking part in a debate about how severe this eye's diabetic "
         "retinopathy is on the ICDR scale.")

_SCALE = """\
The ICDR scale runs 0 to 4:
  0  no apparent retinopathy      3  severe non-proliferative
  1  mild non-proliferative       4  proliferative
  2  moderate non-proliferative"""

_EXAMPLES = (
    "a few small round red dots scattered in the temporal periphery, nothing else",
    "several deep round haemorrhages in two quadrants alongside yellow deposits",
    "new fine vessel tufts arcing off the disc margin",
)


def _block(kit):
    return "\n".join(f"[{tag}] {desc}" for tag, _, desc in kit)


def opening_prompt(kit, no_kg=False):
    if no_kg:
        return f"""\
{_HEAD}

You have not been assigned a grade. Decide for yourself from the image.

{_SCALE}

Report ONLY what you can actually see in this image. Reply with between 2 and 4 \
lines and nothing else, each with two fields separated by | :

  ICDR grade 0-4 | what you actually see in this image

Worked examples of the format only - write your own observations:

  1 | {_EXAMPLES[0]}
  2 | {_EXAMPLES[1]}
  4 | {_EXAMPLES[2]}

Rules:
- Describe a different observation on each line.
- The first field is the grade that ONE observation implies on its own, not your
  overall grade for the eye. Mixed evidence is normal and expected.
- One sentence per line."""
    return f"""\
{_HEAD}

You have not been assigned a grade. Decide for yourself from the image.

{_SCALE}

These are the retinal findings that matter for this decision, each described by \
how it looks on a fundus photograph:
{_block(kit)}

Check them against the image and report ONLY the ones you can actually see. \
Reply with between 2 and 4 lines and nothing else, each with three fields \
separated by | :

  finding tag | ICDR grade 0-4 | what you actually see in this image

Worked examples of the format only - write your own observations:

  f1 | 1 | {_EXAMPLES[0]}
  f2 | 2 | {_EXAMPLES[1]}
  f3 | 4 | {_EXAMPLES[2]}

Rules:
- Never copy a finding's wording. Say what THIS image looks like.
- The middle field is the grade that ONE observation implies on its own, not your
  overall grade for the eye. Mixed evidence is normal and expected.
- Use a different finding tag on each line. One sentence per line."""


def rebuttal_prompt(kit, opponent, own, no_kg=False):
    last = max((c["round_idx"] for c in opponent), default=0)
    recent = [c for c in opponent if c["round_idx"] == last]
    opp = "\n".join(f"  {c['node_id']} | {c['grade']} | {c['text']}"
                    for c in recent) or "  (nothing yet)"
    mine = "\n".join(f"  - {c['text']}" for c in own) or "  (nothing yet)"
    ids = [c["node_id"] for c in recent] or ["c1", "c2"]
    reasons = ("what they missed or misread in the image",
               "why you cannot honestly dispute this one")
    pairs = list(zip((ids + ids)[:2], ("DISAGREE", "AGREE"), (3, 1), reasons))
    if no_kg:
        ex = "\n".join(f"  {cid} | {v} | {g} | {t}" for cid, v, g, t in pairs)
        fields = ("four fields",
                  "their claim id | AGREE or DISAGREE | ICDR grade 0-4 | your reason")
        kg_block = ""
    else:
        ex = "\n".join(f"  {cid} | {v} | f{i+1} | {g} | {t}"
                       for i, (cid, v, g, t) in enumerate(pairs))
        fields = ("five fields",
                  "their claim id | AGREE or DISAGREE | finding tag | "
                  "ICDR grade 0-4 | your reason")
        kg_block = f"\nFindings that matter for this decision:\n{_block(kit)}\n"
    return f"""\
{_HEAD}

This is a DEBATE. You are expected to CHALLENGE what the other ophthalmologist \
claims. Look at the image again and find what they got wrong, overstated, or \
missed. Only AGREE with a claim when you genuinely cannot find any grounds to \
dispute it.

{_SCALE}
{kg_block}
THE OTHER OPHTHALMOLOGIST JUST CLAIMED:
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
- The grade field is what YOUR observation implies on its own.
- One sentence per line."""


def parse_claims(text, agent, round_idx, counter, tag2feat, max_claims=4, no_kg=False):
    """Same field grammar as v5, with an ICDR grade where the label was."""
    out = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        if "|" not in line:
            continue
        parts = [x.strip() for x in line.split("|")]
        addressed, verdict = [], None
        if parts and re.fullmatch(r"c\d+", parts[0], re.I):
            addressed = [parts[0].lower()]
            parts = parts[1:]
        if parts and re.fullmatch(r"(AGREE|DISAGREE)", parts[0], re.I):
            verdict = parts[0].upper()
            parts = parts[1:]
        if no_kg:
            if len(parts) >= 3 and not _GRADE_RE.fullmatch(parts[0].strip()):
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
        gm = _GRADE_RE.search(lab)
        grade = int(gm.group(1)) if gm else None

        body = re.sub(r"\b(AGREE|DISAGREE)\b\s*$", "", body, flags=re.I)
        body = re.sub(r"^\s*[0-4]\s*$", "", body)
        body = " ".join(body.split()).strip(" .|") + "."
        if len(body.split()) < 4:
            continue
        if len(out) >= max_claims:
            break
        counter[0] += 1
        out.append({
            "node_id": f"c{counter[0]}", "text": body, "grade": grade,
            "grade_explicit": grade is not None, "expert_id": agent,
            "round_idx": round_idx, "cited_features": feat,
            "addressed_ids": addressed, "stance_verdict": verdict,
            "is_repeat": False,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--model-b", default=None, help="second lineage; defaults to --model")
    p.add_argument("--split", default="test")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    p.add_argument("--pack", default=str(P.PACK))
    p.add_argument("--npz", default=None)
    p.add_argument("--no-kg", action="store_true",
                   help="KG-free control: no finding kit, agents use their own words")
    p.add_argument("--out-dir", default=str(P.DEBATES))
    args = p.parse_args()

    pack = Path(args.pack)
    findings = [(f[0], f[1]) for f in
                json.loads((pack / "probe_findings.json").read_text())]
    npz = Path(args.npz) if args.npz else pack / f"images_224/{args.split}.npz"
    z = np.load(npz)
    imgs, labels = z["imgs"], z["labels"]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards == args.shard]
    done = done_indices(args.out_dir, args.split)
    todo = [i for i in idxs if i not in done]

    model_b = args.model_b or args.model
    cl_a, cl_b = Client(args.url, args.model), Client(args.url, model_b)
    logger.info("retina debate | %s shard %d/%d | %d findings | %d samples (%d done)",
                args.split, args.shard, args.num_shards, len(findings),
                len(todo), len(done))

    t0 = time.time()
    for k, idx in enumerate(todo, 1):
        sid, gold, ts = f"{idx:04d}", int(labels[idx][0]), time.time()
        b64 = _b64(imgs[idx], args.image_size)
        kit_a, map_a = shuffled([(f, d, "") for f, d in findings], idx, 0)
        kit_b, map_b = shuffled([(f, d, "") for f, d in findings], idx, 1)
        counter, claims = [0], []
        # Seed `seen` with the finding descriptions and the worked examples, as
        # v5 does: without it the agents parrot the kit text back and every such
        # line is scored as a fresh observation.
        seen = ({_norm(d) for _, _, d in kit_a} | {_norm(d) for _, _, d in kit_b}
                | {_norm(x) for x in _EXAMPLES})
        by = {"agent_1": [], "agent_2": []}

        for r in range(args.rounds):
            for agent, cl, kit, tmap in (("agent_1", cl_a, kit_a, map_a),
                                         ("agent_2", cl_b, kit_b, map_b)):
                other = "agent_2" if agent == "agent_1" else "agent_1"
                prompt = (opening_prompt(kit, args.no_kg) if r == 0 else
                          rebuttal_prompt(kit, by[other], by[agent], args.no_kg))
                try:
                    out = cl.call(prompt, b64, num_predict=360)
                    text = out["message"]["content"]
                except Exception as e:
                    logger.warning("  %s r%d %s failed: %s", sid, r, agent, e)
                    continue
                got = parse_claims(text, agent, r, counter,
                                   {t.lower(): f for t, f, _ in kit},
                                   no_kg=args.no_kg)
                # _mark_repeat mutates the claim, maintains `seen` itself, and
                # returns True to DROP -- the same filter contract as v5:342.
                for c in [x for x in got if not _mark_repeat(x, seen)]:
                    claims.append(c); by[agent].append(c)

        append(args.out_dir, args.split, args.shard,
               {"sample_id": int(sid), "gold_grade": gold,
                "rounds_used": args.rounds,
                "models": {"agent_1": args.model, "agent_2": model_b},
                "claims": claims})
        gs = [c["grade"] for c in claims if c["grade"] is not None]
        rate = (time.time() - t0) / k
        logger.info("  [%3d/%3d] %s gold=%d | %2d claims mean-grade %s | %.0fs | ETA %.1fh",
                    k, len(todo), sid, gold, len(claims),
                    f"{np.mean(gs):.2f}" if gs else "n/a",
                    time.time() - ts, rate * (len(todo) - k) / 3600)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, args.out_dir)


if __name__ == "__main__":
    main()
