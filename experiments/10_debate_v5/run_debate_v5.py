"""Open-stance debate: no assigned side, shared findings, disagreement expected.

WHY v5
------
v4 locked each agent to a side and gave it only that side's findings. The result
was confabulation: on a benign image the malignant advocate still asserted
spiculation, irregular shape, microcalcifications and architectural distortion —
it recited its toolkit because it had been told to argue that position and had
nothing else to work with. Not one claim in 1064 from the benign advocate ever
conceded.

v5 removes the lock. Neither agent is told which side to argue. Both see the
same findings, malignant and benign together, and each forms its own view from
the image. What makes it a debate is the instruction rather than the
assignment: an agent is expected to challenge the other's claims, and may only
agree when it genuinely cannot find grounds to disagree.

The hypothesis is that disagreement grounded in the image beats disagreement
grounded in a role. If v4's claims were confabulated because the position came
first, v5's should track the image more closely. If instead both agents simply
converge on the same reading, that says the adversarial framing was carrying
the debate and the graph will collapse — which is worth knowing either way.

Agreement is stated explicitly per claim rather than inferred from labels
matching, so AGREE/DISAGREE edges mean what they say.

"Most relevant triples" here means the KG's stance triples: of 370 triples, the
rest are taxonomy, lesion-type appearances and category metadata, none of which
assert anything about a visible finding. All 16 are shared and shuffled per
sample, since with no assigned side there is no principled way to give one agent
a subset without reintroducing the bias v5 exists to remove.

Usage:
  python run_debate_v5.py --split test --limit 4 --out-dir .../debates_v5
"""
from __future__ import annotations

import argparse, base64, io, json, logging, random, re, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

_LABEL_MAP = {0: "MALIGNANT", 1: "BENIGN"}
_EXAMPLES = [
    "the upper border is angular where it meets the tissue",
    "the whole rim is smooth with no interruption",
    "a dark band sits directly behind the mass",
]


def all_findings(triples, schema):
    """Every stance finding with its visual description, both directions."""
    from run_kg_featureprobe import build_probes
    seen, out = set(), []
    for p in sorted(build_probes(triples, schema), key=lambda x: -abs(x["weight"])):
        d = p["question"].splitlines()[1]
        d = d.replace("Does the mass in this image show ", "").rstrip("?")
        if d in seen:
            continue
        seen.add(d)
        out.append((p["feature"], d, "MALIGNANT" if p["weight"] > 0 else "BENIGN"))
    return out


def shuffled(findings, sample_id, agent_offset=0):
    """Re-tag f1..fN in a per-sample random order.

    v4 showed the model cites whatever sits at the top of a fixed list: 90% of
    its citations went to the first three entries. Shuffling makes position
    uninformative. Seeded by sample id so runs stay reproducible.
    """
    items = list(findings)
    random.Random(int(sample_id) * 7919 + agent_offset).shuffle(items)
    kit = [(f"f{i+1}", feat, desc) for i, (feat, desc, _) in enumerate(items)]
    return kit, {t: f for t, f, _ in kit}


def _block(kit):
    return "\n".join(f"[{tag}] {desc}" for tag, _, desc in kit)


def opening_prompt(kit, no_kg=False):
    if no_kg:
        # The KG-free control. Everything else about the debate is held fixed:
        # open stance, same model, same rounds, same claim caps, same parser.
        # Only the finding list and its tag field are gone, so the agents argue
        # from the image in their own vocabulary.
        return f"""\
You are a radiologist examining a breast ultrasound image, taking part in a \
debate about whether this mass is malignant or benign.

You have not been assigned a side. Decide for yourself from the image.

Report ONLY what you can actually see in this image. Reply with between 2 and \
4 lines and nothing else, each with two fields separated by | :

  MALIGNANT or BENIGN | what you actually see in this image

Worked examples of the format only — write your own observations:

  MALIGNANT | {_EXAMPLES[0]}
  BENIGN | {_EXAMPLES[1]}
  MALIGNANT | {_EXAMPLES[2]}

Rules:
- Describe a different observation on each line.
- The first field is what that one observation implies on its own, not your
  overall verdict. Mixed evidence is normal and expected.
- One sentence per line."""
    return f"""\
You are a radiologist examining a breast ultrasound image, taking part in a \
debate about whether this mass is malignant or benign.

You have not been assigned a side. Decide for yourself from the image.

These ACR BI-RADS findings are the ones that matter for this decision. Some \
point to malignancy, some to a benign cause. Each is described by how it looks \
on ultrasound:
{_block(kit)}

Check them against the image and report ONLY the ones you can actually see. \
Reply with between 2 and 4 lines and nothing else, each with three fields \
separated by | :

  finding tag | MALIGNANT or BENIGN | what you actually see in this image

Worked examples of the format only — write your own observations:

  f1 | MALIGNANT | {_EXAMPLES[0]}
  f2 | BENIGN | {_EXAMPLES[1]}
  f3 | MALIGNANT | {_EXAMPLES[2]}

Rules:
- Never copy a finding's wording. Say what THIS image looks like.
- The middle field is what that one observation implies on its own, not your
  overall verdict. Mixed evidence is normal and expected.
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
    pairs = list(zip((ids + ids)[:2], ("DISAGREE", "AGREE"),
                     ("MALIGNANT", "BENIGN"), reasons))
    if no_kg:
        ex = "\n".join(f"  {cid} | {v} | {lab} | {t}" for cid, v, lab, t in pairs)
        fields = ("four fields", "their claim id | AGREE or DISAGREE | "
                                 "MALIGNANT or BENIGN | your reason")
        kg_block = ""
    else:
        ex = "\n".join(f"  {cid} | {v} | f{i+1} | {lab} | {t}"
                       for i, (cid, v, lab, t) in enumerate(pairs))
        fields = ("five fields", "their claim id | AGREE or DISAGREE | "
                                 "finding tag | MALIGNANT or BENIGN | your reason")
        kg_block = f"\nFindings that matter for this decision:\n{_block(kit)}\n"
    return f"""\
You are a radiologist examining a breast ultrasound image, taking part in a \
debate about whether this mass is malignant or benign.

This is a DEBATE. You are expected to CHALLENGE what the other radiologist \
claims. Look at the image again and find what they got wrong, overstated, or \
missed. Only AGREE with a claim when you genuinely cannot find any grounds to \
dispute it.
{kg_block}
THE OTHER RADIOLOGIST JUST CLAIMED:
{opp}

Sentences you have already used, do not repeat any:
{mine}

Answer their claims. Reply with between 2 and 4 lines and nothing else, each \
with {fields[0]} separated by | :

  {fields[1]}

Worked examples of the format only:

{ex}

Rules:
- Disagree wherever you honestly can; agree only when you cannot.
- Point at the image, not at their wording. Say what you see that they do not.
- Never copy their sentence or repeat one of your own. One sentence per line."""


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image=None, num_predict=360, temperature=0.4, seed=None):
        msg = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [image]
        opts = {"temperature": temperature, "num_ctx": 4096, "num_predict": num_predict}
        if seed is not None:
            opts["seed"] = seed
        body = {"model": self.model, "messages": [msg], "stream": False, "options": opts}
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
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


def _norm(t):
    return re.sub(r"[^a-z ]", " ", t.lower()).strip()


def parse_claims(text, agent, round_idx, counter, tag2feat, max_claims=4,
                 no_kg=False):
    """Parse 3-field opening lines or 5-field rebuttal lines.

    With --no-kg there is no finding tag, so the same lines carry 2 and 4
    fields instead.
    """
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
            # A stray tag turns up occasionally even with none in the prompt.
            if len(parts) >= 3 and not re.search(r"(MALIGNANT|BENIGN)", parts[0], re.I):
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
        lm = re.search(r"(MALIGNANT|BENIGN)", lab, re.I)
        label = lm.group(1).upper() if lm else None

        body = re.sub(r"\b(MALIGNANT|BENIGN|AGREE|DISAGREE)\b\s*$", "", body, flags=re.I)
        body = " ".join(body.split()).strip(" .|") + "."
        if len(body.split()) < 4:
            continue
        if len(out) >= max_claims:
            break
        counter[0] += 1
        out.append({
            "node_id": f"c{counter[0]}", "text": body, "label": label,
            "label_explicit": label is not None, "expert_id": agent,
            "round_idx": round_idx, "cited_features": feat,
            "addressed_ids": addressed, "stance_verdict": verdict,
            "is_repeat": False,
        })
    return out


def _mark_repeat(claim, seen):
    """Flag rather than drop; only reject when nothing usable is left.

    v4 measured what dropping costs: 55% of rebuttal turns lost every line, and
    with them the ADDRESSED edge each line carried, leaving the graph 80%
    opening statements. A familiar sentence pointed at a new opponent claim is
    still structurally new.
    """
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", claim["text"]) if x.strip()]
    kept = [x for x in parts if _norm(x) and _norm(x) not in seen]
    original = claim["text"]
    if kept:
        text = " ".join(kept)
        key = _norm(text)
        near = any(pt and len(set(key.split()) & pt) / len(set(key.split()) | pt) >= 0.9
                   for pt in (set(p.split()) for p in seen))
        if len(key.split()) >= 3 and not near:
            claim["text"] = text
            for x in kept:
                seen.add(_norm(x))
            seen.add(key)
            return False
    claim["text"] = original
    claim["is_repeat"] = True
    return len(_norm(original).split()) < 3


def debate(b64, client, kits, tag2feats, rounds, temperature, no_kg=False, run_id=0):
    counter = [0]
    by_agent = {"agent_1": [], "agent_2": []}
    seen = {_norm(d) for k in kits.values() for _, _, d in k}
    seen |= {_norm(x) for x in _EXAMPLES}
    out = []
    for r in range(rounds):
        for agent in ("agent_1", "agent_2"):
            other = "agent_2" if agent == "agent_1" else "agent_1"
            kit, tag2feat = kits[agent], tag2feats[agent]
            # Only the three most recent opponent claims are answerable; showing
            # fourteen made the model reply to every one and copy each verbatim.
            recent = by_agent[other][-3:]
            prompt = (opening_prompt(kit, no_kg) if r == 0 else
                      rebuttal_prompt(kit, recent, by_agent[agent], no_kg))
            try:
                res = client.call(prompt, image=b64, temperature=temperature,
                                  seed=100000 * run_id + 1000 * r
                                       + (0 if agent == "agent_1" else 1))
                txt = res["message"]["content"]
            except Exception as exc:
                logger.warning("turn %s r%d failed: %s", agent, r, exc)
                continue
            fresh = [c for c in parse_claims(txt, agent, r, counter, tag2feat,
                                             max_claims=4 if r == 0 else 3,
                                             no_kg=no_kg)
                     if not _mark_repeat(c, seen)]
            by_agent[agent] += fresh
            out += fresh
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="medgemma:4b")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--rounds",     type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--shard",      type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--reverse",    action="store_true")
    p.add_argument("--no-kg",      action="store_true",
                   help="KG-free control: identical debate, no findings in the prompt")
    p.add_argument("--run-id",     type=int, default=0,
                   help="independent repeat: reseeds sampling and finding order")
    p.add_argument("--url",        default="http://localhost:11434/api/chat")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--kg-root",    default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    findings = [] if args.no_kg else all_findings(triples, schema)
    if args.no_kg:
        logger.info("KG-FREE control: no findings in the prompt")
    else:
        logger.info("shared finding set: %d (%d malignant, %d benign)", len(findings),
                    sum(1 for _, _, s in findings if s == "MALIGNANT"),
                    sum(1 for _, _, s in findings if s == "BENIGN"))

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    n = len(imgs) if args.limit is None else min(args.limit, len(imgs))
    idxs = [i for i in range(n) if i % args.num_shards == args.shard]
    if args.reverse:
        idxs.reverse()

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, args.model)
    logger.info("debate v5 (open stance) | split=%s shard=%d/%d | %d samples",
                args.split, args.shard, args.num_shards, len(idxs))

    t0 = time.time()
    for k, idx in enumerate(idxs, 1):
        sid = f"{idx:03d}"
        dest = out_dir / f"debate_{sid}.json"
        if dest.exists():
            continue
        gold = _LABEL_MAP[int(labels[idx][0])]
        ts = time.time()
        k1, t1 = shuffled(findings, sid, 2 * args.run_id)
        k2, t2 = shuffled(findings, sid, 2 * args.run_id + 1)
        claims = debate(_b64(imgs[idx], args.image_size), client,
                        {"agent_1": k1, "agent_2": k2},
                        {"agent_1": t1, "agent_2": t2},
                        args.rounds, args.temperature, no_kg=args.no_kg,
                        run_id=args.run_id)
        tmp = dest.with_suffix(f".{args.shard}.tmp")
        tmp.write_text(json.dumps({
            "sample_id": sid, "gold_label": gold, "rounds_used": args.rounds,
            "claims": claims,
        }, indent=1))
        tmp.replace(dest)

        dis = sum(1 for c in claims if c.get("stance_verdict") == "DISAGREE")
        agr = sum(1 for c in claims if c.get("stance_verdict") == "AGREE")
        mal = sum(1 for c in claims if c["label"] == "MALIGNANT")
        rate = (time.time() - t0) / k
        logger.info("  [%3d/%3d] %s gold=%-9s | %2d claims %2d mal | %2d disagree %2d agree | %.0fs | ETA %.1fh",
                    k, len(idxs), sid, gold, len(claims), mal, dis, agr,
                    time.time() - ts, rate * (len(idxs) - k) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
