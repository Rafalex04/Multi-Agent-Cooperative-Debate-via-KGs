"""KG-free adversarial debate; knowledge graph attached afterwards by retrieval.

WHY v3 EXISTS
-------------
v2 gave both agents a measured evidence table before they argued. That made
every claim well grounded, but it also collapsed the debate onto a fixed
17-feature vocabulary, and the consequences showed up in training:

  probe-mlp (probe vector, no graph)   test AUC 0.6985
  GraphSAGE (v2 graph)                 test AUC 0.6083

The graph lost to not using the graph. The reason is that claims only mention
features the agents chose to cite, and they preferentially cited the saturated
ones, so the graph is a lossy and biased compression of the measurements it was
built from. Global claim uniqueness fell to 0.640 for the same reason.

v3 removes the constraint. The agents see only the image and argue from what
they observe, in their own words. The knowledge graph is applied *afterwards*:
each finished claim is linked to its most similar triples by dense retrieval.

The graph therefore encodes what the debate actually said, and the KG supplies
structure over it rather than dictating vocabulary. Whether that trains better
is exactly the open question — the claims may be richer, or they may be vaguer
and less label-correlated than measurements. This script produces the debates;
`link_kg.py` attaches the triples and builds the graphs.

Node features come from claim text embeddings rather than probe values, since
there are no measurements here by construction.

Usage:
  python run_debate_v3.py --split test --limit 8 --out-dir .../debates_v3
"""
from __future__ import annotations

import argparse, base64, io, json, logging, re, sys, time, urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_LABEL_MAP = {0: "MALIGNANT", 1: "BENIGN"}

_CLAIM_SPLIT_RE = re.compile(r"\[CLAIM\]")
_ADDR_ONLY_RE   = re.compile(r"\[ADDRESSED:\s*(c\d+)\s*\]", re.I)
_LABEL_RE       = re.compile(r"\[LABEL:\s*(BENIGN|MALIGNANT)\s*\]", re.I)
_TAG_RE         = re.compile(
    r"\[(?:ADDRESSED:[^\]]*|AGREE|DISAGREE|LABEL:[^\]]*|CITED:[^\]]*)\]", re.I)

_ROLE = {
    "expert_a": ("MALIGNANT",
                 "You argue this mass is MALIGNANT. Make the strongest honest "
                 "case for malignancy from what you can actually see."),
    "expert_b": ("BENIGN",
                 "You argue this mass is BENIGN. Make the strongest honest case "
                 "for a benign diagnosis from what you can actually see."),
}

# No feature list is supplied anywhere in these prompts. The whole point of v3
# is that the vocabulary comes from the model looking at the image.
_OPEN_FORMAT = """\
Reply with exactly 3 claims and nothing else:

[CLAIM] <one specific sentence about what you observe> [LABEL: MALIGNANT or BENIGN]

Rules:
- Describe something you can actually see: shape, margin, echo pattern,
  orientation, posterior features, internal content, surrounding tissue.
- Each claim must be about a different observation.
- Replace 'MALIGNANT or BENIGN' with whichever one that observation implies on
  its own; do not copy the template wording, and do not just echo your side. If what you see argues against your position, say so and label it
  honestly."""

def _rebut_format(ids: list[str]) -> str:
    """Build the template with the opponent's real claim ids substituted in.

    A generic "[ADDRESSED: c2]" example was being ignored by the 4B model and
    every rebuttal came back untagged, which left the graphs with no
    claim-claim edges at all. Showing three concrete lines pre-filled with the
    ids it is supposed to answer is a much stronger cue than a rule.
    """
    picks = (ids + ids + ids)[:3] if ids else ["c1", "c2", "c3"]
    lines = "\n".join(
        f"[CLAIM] [ADDRESSED: {i}] <one new sentence answering {i}> "
        f"[LABEL: MALIGNANT or BENIGN]" for i in picks)
    return (
        "Reply with exactly these 3 claims, filled in, and nothing else:\n\n"
        f"{lines}\n\n"
        "Rules:\n"
        "- Keep the [ADDRESSED: cN] tags exactly as shown above.\n"
        "- ONE sentence per claim. Never join two sentences with a full stop.\n"
        "- Do not copy your opponent's wording and do not repeat a sentence you\n"
        "  have already used. Say what in the image undermines their conclusion.\n"
        "- Replace 'MALIGNANT or BENIGN' with whichever the evidence implies; do\n"
        "  not copy the template wording, and do not just echo your assigned side.")


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url, self.model, self.timeout = url, model, timeout

    def call(self, prompt, image=None, num_predict=320, temperature=0.0, seed=None):
        msg = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [image]
        opts = {"temperature": temperature, "num_ctx": 4096,
                "num_predict": num_predict}
        if seed is not None:
            opts["seed"] = seed
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": opts}
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


def opening_prompt(role: str) -> str:
    _, instruction = _ROLE[role]
    return ("You are a radiologist examining a breast ultrasound image.\n\n"
            f"{instruction}\n\nLook carefully at the image.\n\n{_OPEN_FORMAT}")


def rebuttal_prompt(role: str, opponent: list[dict], own: list[dict]) -> str:
    _, instruction = _ROLE[role]
    last = max((c["round_idx"] for c in opponent), default=0)
    recent = [c for c in opponent if c["round_idx"] == last]
    opp = "\n".join(f"[{c['node_id']}] {c['text']} -> they labelled this {c['label']}"
                    for c in recent) or "(nothing yet)"
    mine = "\n".join(f"- {c['text']}" for c in own) or "(nothing yet)"
    return ("You are a radiologist examining a breast ultrasound image.\n\n"
            f"{instruction}\n\n"
            f"YOUR OPPONENT'S LATEST CLAIMS:\n{opp}\n\n"
            f"Sentences you have ALREADY used (do not repeat any):\n{mine}\n\n"
            f"{_rebut_format([c['node_id'] for c in recent])}")


def parse_claims(text: str, expert_id: str, round_idx: int,
                 counter: list[int]) -> list[dict]:
    claims = []
    for part in _CLAIM_SPLIT_RE.split(text)[1:]:
        body = " ".join(_TAG_RE.sub(" ", part).split()).strip()
        if len(body.split()) < 3:
            continue
        lm = _LABEL_RE.search(part)
        counter[0] += 1
        claims.append({
            "node_id": f"c{counter[0]}",
            "text": body,
            "label": lm.group(1).upper() if lm else None,
            "label_explicit": lm is not None,
            "expert_id": expert_id,
            "round_idx": round_idx,
            "addressed_ids": [a.strip().lower() for a in _ADDR_ONLY_RE.findall(part)],
        })
    return claims


def _norm(t: str) -> str:
    return re.sub(r"[^a-z ]", " ", t.lower()).strip()


def _dedupe(claim: dict, seen: set[str]) -> bool:
    """Strip echoed sentences in place; return True if nothing new is left.

    The model likes to open a "rebuttal" by restating a sentence verbatim and
    only then adding its argument. Dropping the whole claim loses the argument,
    and keeping it lets the echo inflate the graph — so remove just the repeated
    sentences and keep the remainder.
    """
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", claim["text"]) if x.strip()]
    kept = [x for x in parts if _norm(x) and _norm(x) not in seen]
    if not kept:
        return True

    text = " ".join(kept)
    key = _norm(text)
    if not key or len(key.split()) < 3 or key in seen:
        return True
    # Near-identical token sets are restatements even when the wording differs.
    kt = set(key.split())
    for prev in seen:
        pt = set(prev.split())
        if pt and len(kt & pt) / len(kt | pt) >= 0.9:
            return True

    claim["text"] = text
    for x in kept:
        seen.add(_norm(x))
    seen.add(key)
    return False


def debate(b64: str, client: Client, rounds: int, temperature: float) -> list[dict]:
    counter = [0]
    by_expert: dict[str, list[dict]] = {"expert_a": [], "expert_b": []}
    seen: set[str] = set()
    out: list[dict] = []
    for r in range(rounds):
        for role in ("expert_a", "expert_b"):
            other = "expert_b" if role == "expert_a" else "expert_a"
            prompt = (opening_prompt(role) if r == 0 else
                      rebuttal_prompt(role, by_expert[other], by_expert[role]))
            try:
                res = client.call(prompt, image=b64, num_predict=320,
                                  temperature=temperature,
                                  seed=1000 * r + (0 if role == "expert_a" else 1))
                txt = res["message"]["content"]
            except Exception as exc:
                logger.warning("turn %s r%d failed: %s", role, r, exc)
                continue
            fresh = []
            for c in parse_claims(txt, role, r, counter):
                if _dedupe(c, seen):
                    continue
                fresh.append(c)
            by_expert[role] += fresh
            out += fresh
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
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--kg-root",
        default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    root = Path(args.kg_root)
    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    n = len(imgs) if args.limit is None else min(args.limit, len(imgs))
    idxs = [i for i in range(n) if i % args.num_shards == args.shard]
    if args.reverse:
        idxs.reverse()

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, args.model)
    logger.info("debate v3 (no KG) | split=%s shard=%d/%d | %d samples | %d rounds",
                args.split, args.shard, args.num_shards, len(idxs), args.rounds)

    t0 = time.time()
    for k, idx in enumerate(idxs, 1):
        sid  = f"{idx:03d}"
        dest = out_dir / f"debate_{sid}.json"
        if dest.exists():
            continue
        gold = _LABEL_MAP[int(labels[idx][0])]
        ts = time.time()
        claims = debate(_b64(imgs[idx], args.image_size), client,
                        args.rounds, args.temperature)
        tmp = dest.with_suffix(f".{args.shard}.tmp")
        tmp.write_text(json.dumps({
            "sample_id": sid, "gold_label": gold, "rounds_used": args.rounds,
            "claims": claims,
        }, indent=1))
        tmp.replace(dest)

        rate = (time.time() - t0) / k
        logger.info("  [%3d/%3d] %s gold=%-9s | %d claims | %.0fs | ETA %.1fh",
                    k, len(idxs), sid, gold, len(claims), time.time() - ts,
                    rate * (len(idxs) - k) / 3600)

    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
