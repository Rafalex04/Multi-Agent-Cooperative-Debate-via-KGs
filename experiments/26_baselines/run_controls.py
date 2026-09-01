"""Controls for B1: self-consistency and independent ensemble.

Neither appears anywhere in our ledger, and between them they decide whether
ANY of the debate work - ours or the Catfish Agent's - buys something that
repeated independent sampling does not.

One generation pass serves both. Each image gets N independent OPENING turns,
no interaction of any kind: different finding order per draw (the same
`shuffled` the debate uses) and a different sampling seed. Then

  self-consistency N=5   majority vote over the 5 draws' per-claim labels
  independent ensemble   the first 2 draws averaged, matching our 2 agents

  python run_controls.py --split test --out-dir .../controls_b1 --draws 5
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
    Client, _LABEL_MAP, _b64, all_findings, opening_prompt,
    parse_claims, shuffled,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    p.add_argument("--split", default="test")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--draws", type=int, default=5)
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

    z = np.load(root / f"data/breast/images_224/{args.split}.npz")
    imgs, labels = z["imgs"], z["labels"]
    idxs = [i for i in range(len(imgs)) if i % args.num_shards == args.shard]

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    cl = Client(args.url, args.model)
    logger.info("controls | split=%s shard=%d/%d | %d samples | %d draws",
                args.split, args.shard, args.num_shards, len(idxs), args.draws)

    t0 = time.time()
    width = max(3, len(str(len(imgs) - 1)))
    for k, idx in enumerate(idxs, 1):
        sid = f"{idx:0{width}d}"
        dest = out_dir / f"draws_{sid}.json"
        if dest.exists():
            continue
        b64 = _b64(imgs[idx], args.image_size)
        draws = []
        for j in range(args.draws):
            kit, tag2feat = shuffled(findings, sid, j)
            counter = [0]
            try:
                txt = cl.call(opening_prompt(kit), image=b64,
                              temperature=args.temperature,
                              seed=500000 + 1000 * j)["message"]["content"]
            except Exception as exc:
                logger.warning("draw %d failed: %s", j, exc)
                draws.append([]); continue
            draws.append(parse_claims(txt, f"draw_{j}", 0, counter, tag2feat,
                                      max_claims=4))
        tmp = dest.with_suffix(f".{args.shard}.tmp")
        tmp.write_text(json.dumps({
            "sample_id": sid, "gold_label": _LABEL_MAP[int(labels[idx][0])],
            "model": args.model, "draws": draws,
        }, indent=1))
        tmp.replace(dest)
        logger.info("  [%3d/%3d] %s %-9s | claims %s | %.0fs",
                    k, len(idxs), sid, _LABEL_MAP[int(labels[idx][0])],
                    [len(d) for d in draws], (time.time() - t0) / k)
    logger.info("Done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
