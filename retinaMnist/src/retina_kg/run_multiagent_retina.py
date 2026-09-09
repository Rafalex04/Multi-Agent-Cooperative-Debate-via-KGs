"""6-agent debate corpus on RetinaMNIST/ICDR. Feeds B2 (GraphGeo).

Ported from `26_baselines/run_multiagent.py`. Six agents because a 2-agent graph
has 2 nodes and 1 edge, which is a degenerate and therefore unfair implementation
of GraphGeo.

TWO RECORDED DEVIATIONS from the breast version:

1. SINGLE LINEAGE. Breast used 3x qwen3-vl + 3x gemma3:12b. Here all six slots are
   qwen3-vl:8b-instruct with different sampling seeds and finding orders. gemma3:12b
   costs ~5x per call (5.8 s vs 0.78 s measured in the Round 5 pre-registration),
   which turns a 6-10 h corpus into 20-30 h. The agent-slot embedding B2 learns is
   over slots, not lineages, so the graph is unchanged in shape; what is lost is
   the lineage diversity that made r_conflict fire more often on breast. State this
   wherever the retina B2 row appears.

2. VERDICT num_ctx RAISED 2048 -> 4096 to match the debate calls. A different
   num_ctx is a different ollama runner, so the breast layout swaps runners on
   every verdict -- 18 debate calls at 4096 then 6 verdicts at 2048, per image.
   With both lineages on separate nodes that was survivable; single-lineage it
   would be 24 swaps per image. num_ctx only sizes the KV cache and these prompts
   are short, so the returned distribution is unaffected.

The verdict stays BINARY (referable DR, grade >= 2) with a first-token logprob
confidence, because that pair (stance_i, conf_i) is exactly what GraphGeo's
agree/conflict/transfer rule is built from -- it never parses claim text. The
agent's mean claim grade is recorded alongside so an ordinal edge rule can be
tried without regenerating.

  python run_multiagent_retina.py --split test --num-shards 8 --shard 0
"""
from __future__ import annotations

import argparse, json, logging, math, sys, time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("10_debate_v5", "26_baselines")

from run_debate_v5 import Client, _b64, _mark_repeat, _norm, shuffled  # noqa: E402
from corpus_io import append, done_indices                            # noqa: E402
from run_debate_retina import (opening_prompt, parse_claims,          # noqa: E402
                               rebuttal_prompt, _SCALE)

VERDICT_PROMPT = """\
You are an ophthalmologist examining a colour fundus photograph.

These are the observations YOU reported during the case discussion:

{own}

{scale}

Taking your own observations and the image together, does this eye have \
REFERABLE diabetic retinopathy, meaning ICDR grade 2 (moderate) or worse?

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


class VClient(Client):
    def call_raw(self, prompt, image):
        import urllib.request
        msg = {"role": "user", "content": prompt, "images": [image]}
        body = {"model": self.model, "messages": [msg], "stream": False,
                # 4096, NOT 2048: same runner as the debate calls. See docstring.
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 3},
                "logprobs": True, "top_logprobs": 20}
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


def verdict(cl, b64, own_claims):
    """(referable stance, confidence, mean claim grade) -- GraphGeo's edge inputs."""
    own = "\n".join(f"  - {c['text']}" for c in own_claims) or "  (none)"
    v = p_yes(cl.call_raw(VERDICT_PROMPT.format(own=own, scale=_SCALE), b64))
    g = [c["grade"] for c in own_claims if c.get("grade") is not None]
    mg = float(np.mean(g)) if g else None
    if v is None:
        return None, None, mg
    return ("REFERABLE" if v >= 0.5 else "NON_REFERABLE"), float(max(v, 1.0 - v)), mg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    p.add_argument("--split", default="test")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--pack", default=str(P.PACK))
    p.add_argument("--npz", default=None)
    p.add_argument("--out-dir", default=str(P.PACK / "multi6"))
    args = p.parse_args()

    pack = Path(args.pack)
    findings = [tuple(x) for x in
                json.loads((pack / "probe_findings.json").read_text())]
    cl = VClient(args.url, args.model)
    AGENTS = [(f"a{j}", cl) for j in range(6)]     # single lineage, 6 slots

    z = np.load(Path(args.npz) if args.npz else pack / f"images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards == args.shard]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    done = done_indices(out_dir, args.split)
    todo = [i for i in idxs if i not in done]
    logger.info("6-agent retina | %s shard %d/%d | %d samples (%d done) | single-lineage %s",
                args.split, args.shard, args.num_shards, len(todo), len(done), args.model)

    t0 = time.time()
    for k, idx in enumerate(todo, 1):
        gold, ts = int(labels[idx][0]), time.time()
        b64 = _b64(imgs[idx], args.image_size)
        kits, t2f = {}, {}
        for j, (a, _) in enumerate(AGENTS):
            kits[a], t2f[a] = shuffled(findings, idx, j)

        counter = [0]
        by_agent = {a: [] for a, _ in AGENTS}
        seen = {_norm(d) for kk in kits.values() for _, _, d in kk}
        claims = []
        for r in range(args.rounds):
            for j, (a, c_) in enumerate(AGENTS):
                others = [c for b, _ in AGENTS if b != a for c in by_agent[b][-1:]]
                prompt = (opening_prompt(kits[a]) if r == 0 else
                          rebuttal_prompt(kits[a], others[-3:], by_agent[a]))
                try:
                    txt = c_.call(prompt, image=b64, temperature=args.temperature,
                                  seed=1000 * r + j)["message"]["content"]
                except Exception as exc:
                    logger.warning("  %s r%d failed: %s", a, r, exc)
                    continue
                fresh = [x for x in parse_claims(txt, a, r, counter, t2f[a],
                                                 max_claims=4 if r == 0 else 3)
                         if not _mark_repeat(x, seen)]
                for x in fresh:
                    x["model"] = c_.model
                by_agent[a] += fresh
                claims += fresh

        stances = {}
        for a, c_ in AGENTS:
            try:
                s, cf, mg = verdict(c_, b64, by_agent[a])
            except Exception as exc:
                logger.warning("  verdict %s failed: %s", a, exc)
                s, cf, mg = None, None, None
            stances[a] = {"stance": s, "conf": cf, "mean_grade": mg, "model": c_.model}

        append(out_dir, args.split, args.shard, {
            "sample_id": idx, "gold_grade": gold,
            "agents": {a: c_.model for a, c_ in AGENTS},
            "lineages": 1, "stances": stances, "claims": claims,
        })
        ok = sum(1 for v in stances.values() if v["stance"])
        rate = (time.time() - t0) / k
        logger.info("  [%3d/%3d] %04d gold=%d | %2d claims | stances %d/6 | %.0fs | ETA %.1fh",
                    k, len(todo), idx, gold, len(claims), ok,
                    time.time() - ts, rate * (len(todo) - k) / 3600)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
