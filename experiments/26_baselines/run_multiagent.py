"""6-agent debate corpus. Feeds B2 (GraphGeo) and the 6-agent control of ours.

The spec requires 6 agents because a 2-agent graph has 2 nodes and 1 edge, which
is a degenerate and therefore unfair implementation of GraphGeo.

  3 x qwen3-vl:8b-instruct   different sampling seeds and finding orders
  3 x gemma3:12b             the second lineage

InternVL3.5-8B was the spec's preferred second lineage and is NOT obtainable on
this cluster (absent from the ollama library under six tags; the HuggingFace
route fails on a host-redirect bug or reports the repo is not llama.cpp
compatible; a text-only GGUF carries no vision projector). gemma3:12b is the
spec's named acceptable fallback. Recorded in the pre-registration before
running.

Each agent ALSO emits an overall verdict with its first-token logprob, because
GraphGeo's edge rule needs (stance_i, conf_i) per agent - that is what the
agree/conflict/transfer relations are built from, not the per-claim labels.

  python run_multiagent.py --split test --out-dir .../multi6 --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse, json, logging, math, sys, time
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

# The verdict must be AGENT-SPECIFIC or the graph degenerates. A bare question
# at temperature 0 returns the identical distribution for every agent sharing a
# model, so the three qwen slots (and the three gemma slots) collapse to one
# node each, r_conflict never fires, and B2 is a null by construction rather
# than by measurement. Conditioning on the agent's OWN claims makes the verdict
# what the spec says it is - "agent i's overall verdict" - and keeps conf_i a
# genuine first-token logprob.
VERDICT_PROMPT = """\
You are a radiologist examining a breast ultrasound image.

These are the observations YOU reported during the case discussion:

{own}

Taking your own observations and the image together, is this mass malignant?

Answer with one word, yes or no."""


def p_yes(out):
    lp = out.get("logprobs") or out["message"].get("logprobs")
    if not lp:
        return None
    y = n = 0.0
    for c in lp[0].get("top_logprobs") or []:
        t = c.get("token", "").strip().lower()
        p = math.exp(c.get("logprob", -100.0))
        if t.startswith("yes"):
            y += p
        elif t.startswith("no"):
            n += p
    return y / (y + n) if (y + n) > 0 else None


def verdict(cl, b64, own_claims):
    """Overall stance + confidence, from the first-token distribution.

    GraphGeo thresholds the distance between agents' PREDICTIONS to build edges;
    it never parses AGREE/DISAGREE text. This is the port of that quantity.
    """
    own = "\n".join(f"  - {c['text']}" for c in own_claims) or "  (none)"
    body_out = cl.call_raw(VERDICT_PROMPT.format(own=own), b64)
    v = p_yes(body_out)
    if v is None:
        return None, None
    return ("MALIGNANT" if v >= 0.5 else "BENIGN"), float(max(v, 1.0 - v))


class VClient(Client):
    def call_raw(self, prompt, image):
        import urllib.request
        msg = {"role": "user", "content": prompt, "images": [image]}
        body = {"model": self.model, "messages": [msg], "stream": False,
                "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 3},
                "logprobs": True, "top_logprobs": 20}
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-url", default="http://localhost:11434/api/chat")
    p.add_argument("--gemma-url", required=True)
    p.add_argument("--qwen-model", default="qwen3-vl:8b-instruct")
    p.add_argument("--gemma-model", default="gemma3:12b")
    p.add_argument("--split", default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--kg-root", default=str(_HERE.parents[2] / "breastMnist"))
    args = p.parse_args()

    import numpy as np
    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema = json.loads((root / "data/breast/schema.json").read_text())
    findings = all_findings(triples, schema)

    qc = VClient(args.qwen_url, args.qwen_model)
    gc = VClient(args.gemma_url, args.gemma_model)
    # 3 of each lineage; agent slot index is stable and is what B2 embeds
    AGENTS = [("a0", qc), ("a1", qc), ("a2", qc),
              ("a3", gc), ("a4", gc), ("a5", gc)]

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards == args.shard]

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("6-agent | split=%s shard=%d/%d | %d samples | qwen=%s gemma=%s",
                args.split, args.shard, args.num_shards, len(idxs),
                args.qwen_url.split("//")[-1].split(":")[0],
                args.gemma_url.split("//")[-1].split(":")[0])

    t0 = time.time()
    width = max(3, len(str(len(imgs) - 1)))
    for k, idx in enumerate(idxs, 1):
        sid = f"{idx:0{width}d}"
        dest = out_dir / f"debate_{sid}.json"
        if dest.exists():
            continue
        b64 = _b64(imgs[idx], args.image_size)
        kits, t2f = {}, {}
        for j, (a, _) in enumerate(AGENTS):
            kits[a], t2f[a] = shuffled(findings, sid, j)

        counter = [0]
        by_agent = {a: [] for a, _ in AGENTS}
        seen = {_norm(d) for kk in kits.values() for _, _, d in kk}
        claims = []
        for r in range(args.rounds):
            for j, (a, cl) in enumerate(AGENTS):
                # rebut the most recent claims of every OTHER agent
                others = [c for b, _ in AGENTS if b != a for c in by_agent[b][-1:]]
                prompt = (opening_prompt(kits[a]) if r == 0 else
                          rebuttal_prompt(kits[a], others[-3:], by_agent[a]))
                try:
                    txt = cl.call(prompt, image=b64, temperature=args.temperature,
                                  seed=1000 * r + j)["message"]["content"]
                except Exception as exc:
                    logger.warning("  %s r%d failed: %s", a, r, exc)
                    continue
                fresh = [c for c in parse_claims(txt, a, r, counter, t2f[a],
                                                 max_claims=4 if r == 0 else 3)
                         if not _mark_repeat(c, seen)]
                for c in fresh:
                    c["model"] = cl.model
                by_agent[a] += fresh
                claims += fresh

        # per-agent overall stance + confidence: GraphGeo's edge inputs
        stances = {}
        for a, cl in AGENTS:
            try:
                s, cf = verdict(cl, b64, by_agent[a])
            except Exception as exc:
                logger.warning("  verdict %s failed: %s", a, exc)
                s, cf = None, None
            stances[a] = {"stance": s, "conf": cf, "model": cl.model}

        tmp = dest.with_suffix(f".{args.shard}.tmp")
        tmp.write_text(json.dumps({
            "sample_id": sid, "gold_label": _LABEL_MAP[int(labels[idx][0])],
            "agents": {a: cl.model for a, cl in AGENTS},
            "stances": stances, "claims": claims,
        }, indent=1))
        tmp.replace(dest)
        ok = sum(1 for v in stances.values() if v["stance"])
        logger.info("  [%3d/%3d] %s %-9s | %2d claims | stances %d/6 | %.0fs",
                    k, len(idxs), sid, _LABEL_MAP[int(labels[idx][0])],
                    len(claims), ok, (time.time() - t0) / k)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
