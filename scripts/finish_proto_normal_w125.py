#!/usr/bin/env python3
"""Finish remaining proto_normal_weight_w125 jobs with weibo-exclusive GPU."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUT = REPO / "results" / "proto_normal_weight_w125"
SEEDS = (0, 1, 2, 3, 42)
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
VARIANTS = {
    "control_w125": {
        "weight": 1.25,
        "use_proto": True,
        "use_proto_normal_weight": False,
        "num_trial": 1,
    },
    "normal_w125": {
        "weight": 1.25,
        "use_proto": True,
        "use_proto_normal_weight": True,
        "proto_normal_temp": 50.0,
        "num_trial": 1,
    },
}


def missing_jobs() -> List[Tuple[str, str, int]]:
    jobs = []
    for variant in VARIANTS:
        for ds in DATASETS:
            for seed in SEEDS:
                p = OUT / "results" / variant / f"{ds}_seed{seed}.json"
                if not p.exists():
                    jobs.append((variant, ds, seed))
    return jobs


def run_one(variant: str, dataset: str, seed: int, gpu: int) -> Tuple[str, str, int, int, float]:
    cfg = yaml.safe_load(open(REPO / "configs" / f"{dataset}.yaml")) or {}
    cfg.update(VARIANTS[variant])
    cfg["dataset"] = dataset
    cfg["sample_steps"] = 1
    cfg["exp_tag"] = f"proto_nw_{variant}_{dataset}"
    cfg_path = OUT / "configs" / variant / f"{dataset}.yaml"
    result_path = OUT / "results" / variant / f"{dataset}_seed{seed}.json"
    log_path = OUT / "logs" / variant / f"{dataset}_seed{seed}.log"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((OUT / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"{variant}_s{seed}"
    env.pop("FMGAD_REUSE_CHECKPOINTS", None)

    cmd = [
        PYTHON,
        str(REPO / "main_train.py"),
        "--config",
        str(cfg_path),
        "--seed",
        str(seed),
        "--device",
        "0",
        "--result-file",
        str(result_path),
        "--deterministic",
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    auc = float("nan")
    if rc == 0 and result_path.exists():
        payload = json.load(open(result_path))
        auc = float(payload.get("auc_mean", payload.get("auc", float("nan"))))
    print(f"[{variant}] {dataset} s{seed} gpu={gpu} rc={rc} auc={auc:.4f} {time.time()-t0:.0f}s", flush=True)
    return variant, dataset, seed, rc, auc


def aggregate() -> Path:
    lines = [
        "# Proto normality-weighted CFM (weight=1.25)",
        "",
        "| Variant | Books | Disney | Enron | Reddit | Weibo | Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary: Dict[str, Any] = {}
    avgs = {}
    for variant, label in [
        ("control_w125", "uniform proto CFM (w=1.25)"),
        ("normal_w125", "normality-weighted proto CFM (w=1.25)"),
    ]:
        cells = []
        means = []
        per = {}
        for ds in DATASETS:
            aucs = []
            for seed in SEEDS:
                p = OUT / "results" / variant / f"{ds}_seed{seed}.json"
                if p.exists():
                    d = json.load(open(p))
                    aucs.append(float(d.get("auc_mean", d.get("auc"))))
            if len(aucs) == len(SEEDS):
                m = float(np.mean(aucs))
                s = float(np.std(aucs, ddof=1))
                cells.append(f"{m:.4f}±{s:.4f}")
                means.append(m)
                per[ds] = {"auc_mean": m, "auc_std": s, "n": len(aucs)}
            else:
                cells.append(f"nan({len(aucs)}/5)")
                per[ds] = {"n": len(aucs)}
        avg = float(np.mean(means)) if means else float("nan")
        avgs[variant] = avg
        lines.append(f"| {label} | " + " | ".join(cells) + f" | **{avg:.4f}** |")
        summary[variant] = {"per_dataset": per, "avg_auc": avg}
    if all(k in avgs and avgs[k] == avgs[k] for k in ("control_w125", "normal_w125")):
        delta = avgs["normal_w125"] - avgs["control_w125"]
        lines += ["", f"**ΔAvg (normal − control) = {delta:+.4f}**"]
        summary["delta_avg"] = delta
    summary_dir = OUT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    md = summary_dir / "summary.md"
    md.write_text("\n".join(lines) + "\n")
    json.dump(summary, open(summary_dir / "summary.json", "w"), indent=2)
    print("\n".join(lines), flush=True)
    return md


def main() -> int:
    # Weibo exclusive GPU; others share the remaining free cards.
    weibo_gpu = int(os.environ.get("WEIBO_GPU", "5"))
    other_gpus = [int(x) for x in os.environ.get("OTHER_GPUS", "2,4,7").split(",") if x.strip()]

    jobs = missing_jobs()
    print(f"Missing {len(jobs)} jobs; weibo_gpu={weibo_gpu} other_gpus={other_gpus}", flush=True)
    if not jobs:
        aggregate()
        return 0

    weibo_jobs = [j for j in jobs if j[1] == "weibo"]
    other_jobs = [j for j in jobs if j[1] != "weibo"]

    fails = 0

    # Others in parallel
    if other_jobs:
        q: Queue = Queue()
        for g in other_gpus:
            q.put(g)

        def _wrap(job):
            g = q.get()
            try:
                return run_one(job[0], job[1], job[2], g)
            finally:
                q.put(g)

        with ThreadPoolExecutor(max_workers=len(other_gpus)) as ex:
            futs = [ex.submit(_wrap, j) for j in other_jobs]
            for fut in as_completed(futs):
                *_, rc, _ = fut.result()
                if rc != 0:
                    fails += 1

    # Weibo strictly serial on its dedicated GPU
    for variant, ds, seed in weibo_jobs:
        *_, rc, _ = run_one(variant, ds, seed, weibo_gpu)
        if rc != 0:
            fails += 1

    aggregate()
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
