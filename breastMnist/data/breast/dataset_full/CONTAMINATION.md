# Mixed-resolution contamination in `dataset_full/test/`

**Status: this dataset is NOT resolution-homogeneous. Do not use `test/` as a
single population without accounting for the 12 samples listed below.**

## What happened

All 780 graphs were originally built from debates in which the VLM was fed
28x28 thumbnails upscaled to 224x224 — the loader bug fixed in commit
`b47893f` (`load_samples` never passed `size=` to `BreastMNIST`, so medmnist
defaulted to 28px).

A 12-sample probe was later re-run at genuine 224x224 native resolution to
measure the effect of that fix. While that probe ran, the `monitor_daemon.py`
watchdog was still looping and rebuilding `dataset_full` from every debate
directory carrying a `.split_*` marker. It swept the new 224px debates into
`dataset_full/test/graphs/`, overwriting the original 28px graphs for those
12 samples.

## Affected samples

These 12 test graphs were rebuilt from **224x224** debates. The other 144 test
graphs, and all 546 train and 78 val graphs, remain **28x28**:

```
007  024  033  038  050  052  107  108  114  115  124  150
```

Verified by MD5: each of these files is byte-identical to its counterpart in
`data/breast/dataset_probe12/test/graphs/`, and differs from the pre-probe
version preserved in `data/breast/dataset_probe12_BEFORE/test/graphs/`.

`test/index.json` and `test/stats.json` were recomputed over the mixed set and
are therefore also affected. The top-level `index.json` and `stats.json`
aggregate over all splits and inherit the same issue.

## Consequences

- Any test-split metric computed over all 156 samples mixes two input
  distributions. The 224px debates carry measurably more image detail
  (high-frequency content 13.9 vs 1.2 on sample 052).
- Train (546) and val (78) are unaffected and internally consistent.
- Per-split accuracy in `test/stats.json` should not be quoted without this
  caveat.

## Recovery

The original 28px graphs for all 12 samples are preserved two ways:

1. `data/breast/dataset_probe12_BEFORE/test/graphs/sample_{id}.json`
2. Git history prior to this commit (`git show <commit>:<path>`)

To restore a homogeneous 28px test split:

```bash
cp data/breast/dataset_probe12_BEFORE/test/graphs/*.json \
   data/breast/dataset_full/test/graphs/
# then rebuild index.json / stats.json
python -m debate_kg.dataset.build_graph_dataset --auto-discover \
    --out-dir data/breast/dataset_full --k 3 --force
```

To instead move the whole dataset to 224px, every split must be regenerated —
roughly 6 days of CPU inference at the measured ~10.5 min/sample.

## Prevention

`monitor_daemon.py` rebuilds `dataset_full` from all marked debate dirs on every
loop. Stop the daemon before running any experimental arm whose output should
not enter the main dataset, or write that arm to a separate `--out-dir` **and**
withhold the `.split_*` marker until the build is done.
