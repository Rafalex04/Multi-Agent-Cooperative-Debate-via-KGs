"""Stance-matched debate: each agent sees only the triples supporting its side.

WHY v4
------
v2 gave both agents the same measured evidence table. v3 gave them no knowledge
graph at all. Both produced graphs a GNN could not learn from:

  v2 GraphSAGE   AUC 0.6037    claim uniqueness 0.640
  v3 GraphSAGE   AUC 0.5543    claim uniqueness 0.181

v3 failed because free-text description with nothing to anchor on collapsed onto
boilerplate — "The mass has irregular margins." accounted for 743 of 6249
claims. v2 failed the opposite way: the shared table let both agents cite the
same saturated features, so the graph became a lossy copy of measurements the
probe vector already held in full.

v4 gives each agent its own toolkit. The malignant advocate sees only the
findings that indicate malignancy; the benign advocate sees only the benign
ones. Neither can borrow the other's vocabulary, so the two sides must describe
different things about the same image, and the resulting claims should separate.

THREE PROPERTIES THE EARLIER VERSIONS LACKED
--------------------------------------------
1. Claims cite a specific finding by tag, so claim-to-KG edges are grounded in
   what the agent actually invoked rather than recovered afterwards by
   embedding similarity, as v3 had to do.
2. A claim may be labelled against its own side. An agent that looks for
   spiculation and does not find it is asked to say so and label that claim for
   the opposition. Absent findings are evidence too, and neither v1 nor v2 could
   express them — which is a large part of why `mal_share` sat at 0.49 in both.
3. Findings carry their visual description, not their snake_case name. The
   bare-name variant cost 0.08 AUC in the probe experiments.

Usage:
  python run_debate_v4.py --split test --limit 4 --out-dir .../debates_v4
"""
from __future__ import annotations

import argparse, base64, io, json, logging, random, re, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "03_kg_grounded_vlm"))

_LABEL_MAP = {0: "MALIGNANT", 1: "BENIGN"}

_CLAIM_SPLIT_RE = re.compile(r"\[CLAIM\]")
_CITES_RE       = re.compile(r"\[CITES:\s*(f\d+)\s*\]", re.I)
_ADDR_RE        = re.compile(r"\[ADDRESSED:\s*(c\d+)\s*\]", re.I)
_LABEL_RE       = re.compile(r"\[LABEL:\s*(BENIGN|MALIGNANT)\s*\]", re.I)
_TAG_RE         = re.compile(
    r"\[(?:CITES:[^\]]*|ADDRESSED:[^\]]*|LABEL:[^\]]*)\]", re.I)

_SIDE = {"expert_a": "MALIGNANT", "expert_b": "BENIGN"}

# The format needs concrete example lines -- abstract placeholders made the 4B
# model lose the field structure. But concrete examples get copied verbatim, so
# these are seeded into the dedup memory and stripped if they come back.
_EXAMPLE_SENTENCES = [
    "the upper border is angular and the lower edge fades into tissue",
    "the margin is smooth the whole way round, no spicules at all",
    "a dark band sits directly behind the mass, obscuring the tissue",
]


def stance_toolkits(triples, schema):
    """Findings for each side, with visual descriptions and stable [fN] tags.

    Built from KG stance polarity only, so which findings a side receives is a
    property of the graph rather than of the labels.
    """
    from run_kg_featureprobe import build_probes
    seen, mal, ben = set(), [], []
    for p in sorted(build_probes(triples, schema), key=lambda x: -abs(x["weight"])):
        d = p["question"].splitlines()[1]
        d = d.replace("Does the mass in this image show ", "").rstrip("?")
        if d in seen:
            continue
        seen.add(d)
        (mal if p["weight"] > 0 else ben).append((p["feature"], d))
    tag = lambda lst, pre: [(f"{pre}{i+1}", feat, desc)
                            for i, (feat, desc) in enumerate(lst)]
    return tag(mal, "f"), tag(ben, "f")


def _toolkit_block(kit):
    return "\n".join(f"[{tag}] {desc}" for tag, _, desc in kit)


def opening_prompt(role, kit):
    side = _SIDE[role]
    other = "BENIGN" if side == "MALIGNANT" else "MALIGNANT"
    return f"""\
You are a radiologist examining a breast ultrasound image.

You are arguing that this mass is {side}.

These ACR BI-RADS findings indicate a {side} mass. Each is described by how it \
looks on ultrasound:
{_toolkit_block(kit)}

Check each finding against the image, then report ONLY the ones you can \
actually see. Reply with between 1 and 3 lines and nothing else. Each line has \
three fields separated by | :

  finding tag | {side} or {other} | what you actually see in this image

Worked examples of the format:

  f1 | {side} | {_EXAMPLE_SENTENCES[0]}
  f2 | {other} | {_EXAMPLE_SENTENCES[1]}
  f3 | {side} | {_EXAMPLE_SENTENCES[2]}

Those three are only to show the format. Write your own observations.

Rules:
- Report only findings genuinely visible in THIS image. If you can see just one,
  write one line. Three weak claims are worse than one solid one.
- Use a different finding tag on each line.
- Never copy a finding's wording. Say what THIS image looks like.
- If a finding is absent, say so and put {other} in the middle field, as in the \
second example. An honest absence is stronger evidence than a claimed presence.
- One sentence per line. No extra text, no numbering, no bullet points."""


def rebuttal_prompt(role, kit, opponent, own):
    side = _SIDE[role]
    other = "BENIGN" if side == "MALIGNANT" else "MALIGNANT"
    last = max((c["round_idx"] for c in opponent), default=0)
    recent = [c for c in opponent if c["round_idx"] == last]
    opp = "\n".join(f"  {c['node_id']} | {c['text']}" for c in recent) or "  (nothing yet)"
    mine = "\n".join(f"  - {c['text']}" for c in own) or "  (nothing yet)"
    ids = [c["node_id"] for c in recent] or ["c1", "c2", "c3"]
    picks = (ids + ids + ids)[:3]
    ex = "\n".join(
        f"  {cid} | f{i+1} | {side} | your answer to {cid}, describing the image"
        for i, cid in enumerate(picks))
    return f"""\
You are a radiologist examining a breast ultrasound image.

You are arguing that this mass is {side}.

Findings that indicate a {side} mass:
{_toolkit_block(kit)}

YOUR OPPONENT JUST SAID:
{opp}

Sentences you have already used, do not repeat any:
{mine}

Reply with between 1 and 3 lines and nothing else. Each line has four fields \
separated by | :

  their claim id | your finding tag | {side} or {other} | your answer

Answer their claims, one line each, using only findings you can actually see:

{ex}

Rules:
- Answer what they said about the image: point to something they missed, or \
explain why what they saw does not mean what they concluded.
- Never copy their wording or repeat a sentence you already used.
- If they are right, concede it and put {other} in the third field.
- One sentence per line. No extra text."""


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image=None, num_predict=340, temperature=0.4, seed=None):
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


def _dedupe(claim, seen):
    """Flag repetition instead of discarding it; return True only if empty.

    Dropping repeats starved the debate. Measured over 311 debates, round 0
    produced 11.1 claims per graph while rounds 1 and 2 together produced 2.7,
    because 55% of rebuttal turns had every line removed as a repeat. The model
    rebuts by re-reading its finding list and copying from it, so the filter was
    correctly identifying parroting -- but deleting the turn also deleted the
    ADDRESSED edge that came with it, leaving 2.7 debate edges per graph.

    A sentence aimed at a different opponent claim is structurally new even when
    its wording is not, so the claim is kept and marked. Echoed sentences are
    still stripped when something original remains alongside them; only when
    nothing original is left does the claim keep its original text and get
    is_repeat, which the graph builder exposes as a node feature.
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
            claim["is_repeat"] = False
            return False

    # Nothing original left: keep the turn for its edge, but mark it.
    claim["text"] = original
    claim["is_repeat"] = True
    return len(_norm(original).split()) < 3


def parse_claims(text, role, round_idx, counter, tag2feat):
    """Parse pipe-delimited lines, falling back to the old bracket tags.

    Bracket tags turned out to be fragile: once the prompt grew, the 4B model
    started dropping [CITES:] and [LABEL:] and spilling the label into the claim
    text. Pipe fields survive much better, but the bracket path is kept so
    earlier transcripts remain readable.
    """
    out = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        if "|" not in line:
            continue
        parts = [x.strip() for x in line.split("|")]
        addressed, feat, label, body = [], [], None, None

        if len(parts) >= 4 and re.fullmatch(r"c\d+", parts[0], re.I):
            addressed = [parts[0].lower()]
            parts = parts[1:]
        if len(parts) >= 3:
            tag, lab, body = parts[0], parts[1], " | ".join(parts[2:])
        elif len(parts) == 2:
            tag, lab, body = parts[0], "", parts[1]
        else:
            continue

        m = re.fullmatch(r"f\d+", tag.strip(), re.I)
        if m and tag.strip().lower() in tag2feat:
            feat = [tag2feat[tag.strip().lower()]]
        lm = re.search(r"(MALIGNANT|BENIGN)", lab, re.I)
        if lm:
            label = lm.group(1).upper()

        body = _TAG_RE.sub(" ", body)
        body = re.sub(r"\b(MALIGNANT|BENIGN)\b\s*$", "", body, flags=re.I)
        body = " ".join(body.split()).strip(" .|") + "."
        if len(body.split()) < 4:
            continue
        counter[0] += 1
        out.append({
            "node_id": f"c{counter[0]}", "text": body,
            "label": label, "label_explicit": label is not None,
            "expert_id": role, "round_idx": round_idx,
            "cited_features": feat, "addressed_ids": addressed,
            "is_repeat": False,
        })

    if out:
        return out
    # fallback: original bracket-tag format
    for part in _CLAIM_SPLIT_RE.split(text)[1:]:
        body = " ".join(_TAG_RE.sub(" ", part).split()).strip()
        if len(body.split()) < 3:
            continue
        lm = _LABEL_RE.search(part)
        counter[0] += 1
        out.append({
            "node_id": f"c{counter[0]}", "text": body,
            "label": lm.group(1).upper() if lm else None,
            "label_explicit": lm is not None,
            "expert_id": role, "round_idx": round_idx,
            "cited_features": [tag2feat[t.lower()] for t in _CITES_RE.findall(part)
                               if t.lower() in tag2feat],
            "addressed_ids": [a.lower() for a in _ADDR_RE.findall(part)],
            "is_repeat": False,
        })
    return out


def shuffled(kits, sample_id):
    """Re-order each toolkit per sample, re-tagging f1..fN in the new order.

    Measured on the first 227 debates, agents cite whatever sits at the top of
    the list: the malignant advocate spent 90% of its citations on f1-f3 and
    named spiculation, the strongest malignancy sign, in 2.6% of claims. Only 39
    distinct citation-sets appeared across 227 images. Shuffling per sample
    makes position uninformative, so a finding gets cited because it is visible
    rather than because it is first. Seeded by sample id to stay reproducible.
    """
    rng = random.Random(int(sample_id))
    out, t2f = {}, {}
    for role, kit in kits.items():
        items = [(feat, desc) for _, feat, desc in kit]
        rng.shuffle(items)
        out[role] = [(f"f{i+1}", feat, desc) for i, (feat, desc) in enumerate(items)]
        t2f[role] = {t: f for t, f, _ in out[role]}
    return out, t2f


def debate(b64, clients, kits, tag2feat, rounds, temperature):
    # `clients` maps role -> Client. Passing a bare Client keeps the old
    # homogeneous behaviour byte-for-byte, so pre-existing runs still reproduce.
    if not isinstance(clients, dict):
        clients = {"expert_a": clients, "expert_b": clients}
    counter = [0]
    by_expert = {"expert_a": [], "expert_b": []}
    # Pre-load the finding descriptions so a claim that just parrots one back is
    # removed by the same dedup that strips echoed sentences. Without this the
    # opening round comes back as verbatim copies of the toolkit.
    seen = {_norm(desc) for kit in kits.values() for _, _, desc in kit}
    seen |= {_norm(x) for x in _EXAMPLE_SENTENCES}
    out = []
    for r in range(rounds):
        for role in ("expert_a", "expert_b"):
            other = "expert_b" if role == "expert_a" else "expert_a"
            client = clients[role]
            kit = kits[role]
            prompt = (opening_prompt(role, kit) if r == 0 else
                      rebuttal_prompt(role, kit, by_expert[other], by_expert[role]))
            try:
                res = client.call(prompt, image=b64, temperature=temperature,
                                  seed=1000 * r + (0 if role == "expert_a" else 1))
                txt = res["message"]["content"]
            except Exception as exc:
                logger.warning("turn %s r%d failed: %s", role, r, exc)
                continue
            fresh = [c for c in parse_claims(txt, role, r, counter, tag2feat[role])
                     if not _dedupe(c, seen)]
            by_expert[role] += fresh
            out += fresh
    for _c in out:
        _c["model"] = clients[_c["expert_id"]].model
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="medgemma:4b")
    p.add_argument("--split",      default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--rounds",     type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--shard",      type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--reverse",    action="store_true")
    p.add_argument("--url",        default="http://localhost:11434/api/chat")
    p.add_argument("--model-b",    default=None,
                   help="model for expert_b. Default: same as --model.")
    p.add_argument("--url-b",      default=None,
                   help="endpoint for expert_b; defaults to --url.")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--kg-root",    default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema  = json.loads((root / "data/breast/schema.json").read_text())
    mal_kit, ben_kit = stance_toolkits(triples, schema)
    kits = {"expert_a": mal_kit, "expert_b": ben_kit}
    tag2feat = {"expert_a": {t: f for t, f, _ in mal_kit},
                "expert_b": {t: f for t, f, _ in ben_kit}}
    logger.info("toolkits: malignant=%d findings, benign=%d findings",
                len(mal_kit), len(ben_kit))

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    n = len(imgs) if args.limit is None else min(args.limit, len(imgs))
    idxs = [i for i in range(n) if i % args.num_shards == args.shard]
    if args.reverse:
        idxs.reverse()

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    model_b = args.model_b or args.model
    clients = {"expert_a": Client(args.url, args.model),
               "expert_b": Client(args.url_b or args.url, model_b)}
    logger.info("agents: expert_a=%s  expert_b=%s  [%s]", args.model, model_b,
                "HETEROGENEOUS" if (model_b != args.model
                                    or args.url_b not in (None, args.url))
                else "homogeneous")
    logger.info("debate v4 (stance-matched) | split=%s shard=%d/%d | %d samples",
                args.split, args.shard, args.num_shards, len(idxs))

    t0 = time.time()
    for k, idx in enumerate(idxs, 1):
        sid = f"{idx:03d}"
        dest = out_dir / f"debate_{sid}.json"
        if dest.exists():
            continue
        gold = _LABEL_MAP[int(labels[idx][0])]
        ts = time.time()
        kits_s, tag2feat_s = shuffled(kits, sid)
        claims = debate(_b64(imgs[idx], args.image_size), clients, kits_s,
                        tag2feat_s, args.rounds, args.temperature)
        tmp = dest.with_suffix(f".{args.shard}.tmp")
        tmp.write_text(json.dumps({
            "sample_id": sid, "gold_label": gold, "rounds_used": args.rounds,
            "claims": claims,
        }, indent=1))
        tmp.replace(dest)

        cited = sum(1 for c in claims if c["cited_features"])
        rep = sum(1 for c in claims if c.get("is_repeat"))
        against = sum(1 for c in claims
                      if c["label"] and c["label"] != _SIDE[c["expert_id"]])
        rate = (time.time() - t0) / k
        logger.info("  [%3d/%3d] %s gold=%-9s | %2d claims %2d cited %2d repeat %2d against | %.0fs | ETA %.1fh",
                    k, len(idxs), sid, gold, len(claims), cited, rep, against,
                    time.time() - ts, rate * (len(idxs) - k) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
