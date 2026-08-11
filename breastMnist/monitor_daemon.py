"""Autonomous pipeline monitor — runs all splits sequentially, restarts on crash."""
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

WORKDIR = Path(__file__).parent
VENV = str(WORKDIR / ".venv/bin/python")
PYTHONPATH = str(WORKDIR.parent)
LOG = WORKDIR / "outputs/monitor.log"
DATASET_DIR = WORKDIR / "data/breast/dataset_full"
CHECK_INTERVAL = 300    # 5 min
STALL_LIMIT = 46800     # 13 h without progress → restart (> OOM retry patience of 12 h)

os.chdir(WORKDIR)
LOG.parent.mkdir(parents=True, exist_ok=True)

# Each run: (label, split, indices_path, num_samples, resume_name)
RUNS = [
    ("remaining-test", "test",  "data/breast/remaining_indices.json",  106, "debate_kg_adversarial_106_42.jsonl"),
    ("val",            "val",   "data/breast/val_indices.json",          78, "debate_kg_adversarial_78_42.jsonl"),
    ("train",          "train", "data/breast/train_indices.json",        546, "debate_kg_adversarial_546_42.jsonl"),
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def resume_count(resume_name: str) -> int:
    path = WORKDIR / "outputs/resume" / resume_name
    try:
        return sum(1 for ln in path.read_text().splitlines() if ln.strip())
    except FileNotFoundError:
        return 0


def mark_split_dir(split: str, output_dir: Path) -> None:
    """Write a split marker into the pipeline output dir so we can identify it later."""
    try:
        (output_dir / f".split_{split}").touch()
    except Exception:
        pass


def debate_json_count(split: str, indices_path: str) -> int:
    """Count debate JSONs only from output dirs tagged for this split."""
    import json as _json
    indices = _json.loads((WORKDIR / indices_path).read_text())["indices"]
    expected_ids = {f"{i:03d}" for i in indices}
    found = set()
    for d in WORKDIR.glob("outputs/????-??-??/??-??-??"):
        if not (d / f".split_{split}").exists():
            continue
        for p in d.glob("debate_sample-*.json"):
            sid = p.stem.replace("debate_sample-", "")
            if sid in expected_ids:
                found.add(sid)
    return len(found)


def find_latest_log() -> Path | None:
    logs = sorted(WORKDIR.glob("outputs/????-??-??/??-??-??/breast_main.log"))
    return logs[-1] if logs else None


def log_tail(n: int = 4) -> None:
    p = find_latest_log()
    if not p:
        return
    try:
        for line in p.read_text().splitlines()[-n:]:
            log(f"  | {line}")
    except Exception:
        pass


def proc_alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def start_pipeline(split: str, indices_path: str, num_samples: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    log(f"Starting pipeline: split={split} n={num_samples} indices={indices_path}")
    proc = subprocess.Popen(
        [VENV, "-m", "debate_kg.breast_main",
         "run=breast_adversarial",
         f"data.split={split}",
         f"data.indices_path={indices_path}",
         f"data.num_samples={num_samples}",
         "run.skip_judge=true"],
        env=env, cwd=str(WORKDIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log(f"Pipeline PID: {proc.pid}")
    # Wait briefly for Hydra to create the output dir, then tag it with the split
    time.sleep(8)
    latest = find_latest_log()
    if latest:
        mark_split_dir(split, latest.parent)
        log(f"Marked {latest.parent.name} as split={split}")
    return proc


def build_full_dataset() -> None:
    log("Building full merged dataset from all debate outputs...")
    # Collect all debate dirs that have debate_sample-*.json files
    debate_dirs = sorted({
        str(p.parent)
        for p in WORKDIR.glob("outputs/????-??-??/??-??-??/debate_sample-*.json")
    })
    log(f"  Found {len(debate_dirs)} debate output dir(s)")
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    args = [VENV, "-m", "debate_kg.dataset.build_graph_dataset",
            "--out-dir", str(DATASET_DIR),
            "--k", "3", "--force"]
    for d in debate_dirs:
        args += ["--debate-dir", d]
    result = subprocess.run(args, env=env, cwd=str(WORKDIR), capture_output=True, text=True)
    log(result.stdout.strip() or "(no stdout)")
    if result.stderr.strip():
        log(f"STDERR: {result.stderr.strip()[:500]}")
    log(f"Dataset builder exit: {result.returncode}")


def run_split(label: str, split: str, indices_path: str, num_samples: int, resume_name: str) -> None:
    log(f"=== Starting split: {label} ({num_samples} samples) ===")

    # Check completion via debate JSON count (resume file is deleted on pipeline success)
    done = debate_json_count(split, indices_path)
    if done >= num_samples:
        log(f"  Already complete ({done}/{num_samples} debate JSONs found), skipping.")
        return

    done = resume_count(resume_name)
    proc = start_pipeline(split, indices_path, num_samples)
    last_done = done
    last_progress_t = time.time()

    while True:
        time.sleep(CHECK_INTERVAL)
        # Prefer resume file count (live progress); fall back to JSON count after success
        resume = resume_count(resume_name)
        json_done = debate_json_count(split, indices_path)
        done = resume if resume > 0 else json_done
        latest = find_latest_log()
        log(f"Check [{label}]: {done}/{num_samples} | alive={proc_alive(proc)} | dir={latest.parent.name if latest else '?'}")
        log_tail(4)

        if done > last_done:
            last_done = done
            last_progress_t = time.time()

        if done >= num_samples or json_done >= num_samples:
            log(f"Split {label} complete! ({done}/{num_samples})")
            if proc_alive(proc):
                proc.wait()
            return

        stalled = (time.time() - last_progress_t) > STALL_LIMIT
        if not proc_alive(proc) or stalled:
            # Only restart if genuinely not done
            if json_done >= num_samples:
                log(f"Split {label} already complete via JSON count ({json_done}/{num_samples}), not restarting.")
                return
            reason = "stalled" if stalled else f"exited({proc.returncode})"
            log(f"Pipeline {reason} at {done}/{num_samples} — restarting...")
            if not proc_alive(proc):
                proc.wait()
            else:
                proc.kill()
                proc.wait()
            proc = start_pipeline(split, indices_path, num_samples)
            last_progress_t = time.time()


def main() -> None:
    log("=== Monitor daemon started (all splits) ===")
    for label, split, indices_path, num_samples, resume_name in RUNS:
        run_split(label, split, indices_path, num_samples, resume_name)
    build_full_dataset()
    log("=== All splits done. Full dataset built. ===")


if __name__ == "__main__":
    main()
