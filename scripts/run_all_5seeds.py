#!/usr/bin/env python3
"""
Run FMGAD on multiple datasets and seeds in parallel; write one JSON per run under results/main_runs/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("books", "disney", "enron", "reddit", "weibo")


def _one(args: Tuple[str, int, str, str, Path, bool]) -> Dict[str, Any]:
    dataset, seed, gpu, py_exe, out_dir, deterministic = args
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{dataset}_seed{seed}.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_RUN_TAG_SUFFIX"] = f"seed{seed}"
    if not env.get("FMGAD_MODEL_ROOT"):
        default_models = (REPO / "outputs" / "models").resolve()
        try:
            default_models.mkdir(parents=True, exist_ok=True)
            env["FMGAD_MODEL_ROOT"] = str(default_models)
        except OSError:
            pass
    cmd = [
        py_exe,
        str(REPO / "scripts" / "run_single.py"),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--device",
        "0",
        "--result-file",
        str(result_path),
    ]
    if deterministic:
        cmd.append("--deterministic")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    row: Dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "result_file": str(result_path),
    }
    if proc.returncode == 0 and result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row["auc_mean"] = payload.get("auc_mean")
        row["ap_mean"] = payload.get("ap_mean")
    else:
        row["auc_mean"] = None
        row["ap_mean"] = None
        row["stderr_tail"] = (proc.stderr or "")[-2000:]
    log_path = out_dir / f"{dataset}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        if proc.stderr:
            f.write("\n--- stderr ---\n")
            f.write(proc.stderr)
    row["log_file"] = str(log_path)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default="42,0,1,2,3")
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--output-dir", type=str, default="results/main_runs")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--model-root", type=str, default=None, help="Checkpoint root shared by child runs")
    args = ap.parse_args()

    dsets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model_root:
        os.environ["FMGAD_MODEL_ROOT"] = str(Path(args.model_root).expanduser().resolve())
    py_exe = sys.executable

    jobs: List[Tuple[str, int, str, str, Path, bool]] = []
    idx = 0
    for d in dsets:
        for s in seeds:
            g = gpu_list[idx % len(gpu_list)]
            jobs.append((d, s, g, py_exe, out_dir, bool(args.deterministic)))
            idx += 1

    mw = min(args.max_workers, len(jobs), max(len(gpu_list), 1))
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            r = rows[-1]
            print(f"[{k}/{len(jobs)}] {r['dataset']} seed={r['seed']} rc={r['returncode']} auc={r.get('auc_mean')}", flush=True)

    by_ds: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("auc_mean") is not None:
            by_ds[r["dataset"]].append({"seed": r["seed"], "auc_mean": float(r["auc_mean"])})
    summary_by_dataset: Dict[str, Any] = {}
    for d in sorted(by_ds.keys()):
        aucs = [x["auc_mean"] for x in by_ds[d]]
        summary_by_dataset[d] = {
            "mean_auc": sum(aucs) / len(aucs),
            "n_seeds": len(aucs),
            "per_seed": by_ds[d],
        }

    meta_path = out_dir / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_jobs": len(jobs),
                "config_note": "Each job uses repo configs/<dataset>.yaml via scripts/run_single.py",
                "summary_by_dataset": summary_by_dataset,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("Wrote", meta_path, flush=True)
    return 0 if all(r.get("returncode") == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
