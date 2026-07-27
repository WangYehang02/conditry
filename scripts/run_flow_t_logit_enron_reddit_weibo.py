#!/usr/bin/env python3
"""Switch Enron/Reddit/Weibo flow_t_sampling uniform → logit_normal; compare to Full."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATASETS = ("enron", "reddit", "weibo")
SEEDS = (0, 1, 2, 3, 42)
FULL_ROOT = REPO / "results" / "ablation_best" / "results" / "full"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _pick_auc(payload: Dict[str, Any]) -> float:
    for k in ("auc", "auc_mean", "AUC"):
        if k in payload and payload[k] is not None:
            return float(payload[k])
    raise KeyError(f"no auc in keys={list(payload.keys())}")


def _run_one(dataset: str, seed: int, gpu: int, out_root: Path, deterministic: bool) -> Tuple[str, int, int, float]:
    cfg = _load_yaml(REPO / "configs" / f"{dataset}.yaml")
    assert str(cfg.get("flow_t_sampling")) == "uniform", (dataset, cfg.get("flow_t_sampling"))
    assert float(cfg.get("weight", 0.0)) == 0.0, (dataset, cfg.get("weight"))
    cfg = deepcopy(cfg)
    cfg["flow_t_sampling"] = "logit_normal"
    cfg["dataset"] = dataset
    cfg["exp_tag"] = f"ablation_flow_t_logit_{dataset}"

    cfg_path = out_root / "configs" / f"{dataset}.yaml"
    result_path = out_root / "results" / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / f"{dataset}_seed{seed}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return dataset, seed, 0, _pick_auc(json.load(f))
        except Exception:
            pass

    _save_yaml(cfg_path, cfg)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"s{seed}"
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
    ]
    if deterministic:
        cmd.append("--deterministic")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    auc = float("nan")
    if rc == 0 and result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            auc = _pick_auc(json.load(f))
    print(f"[logit-t] {dataset} s{seed} gpu={gpu} rc={rc} auc={auc:.4f} {time.time()-t0:.0f}s", flush=True)
    return dataset, seed, rc, auc


def _aggregate(out_root: Path) -> None:
    lines = [
        "# flow_t_sampling: uniform (current) → logit_normal",
        "",
        "Enron/Reddit/Weibo have `weight=0` (proto velocity unused at inference),",
        "but proto training still updates the residual gate.",
        "",
        "| Dataset | uniform (Full) | logit_normal | Δ |",
        "|---|---:|---:|---:|",
    ]
    summary = {}
    for ds in DATASETS:
        base_vals = [_pick_auc(json.load(open(FULL_ROOT / f"{ds}_seed{s}.json"))) for s in SEEDS]
        new_vals = []
        for s in SEEDS:
            p = out_root / "results" / f"{ds}_seed{s}.json"
            if p.exists():
                new_vals.append(_pick_auc(json.load(open(p))))
        b = round(float(np.mean(base_vals)), 3)
        if len(new_vals) != len(SEEDS):
            lines.append(f"| {ds} | {b:.3f} | -- | -- |")
            continue
        n = round(float(np.mean(new_vals)), 3)
        lines.append(f"| {ds} | {b:.3f} | {n:.3f} | {n-b:+.3f} |")
        summary[ds] = {"uniform": b, "logit_normal": n, "delta": round(n - b, 3)}
    md = out_root / "summary" / "flow_t_logit.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "summary" / "flow_t_logit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(md.read_text(encoding="utf-8"), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=str, default=str(REPO / "results" / "flow_t_logit_enron_reddit_weibo"))
    ap.add_argument("--gpus", type=str, default="2,3,4,5")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(out_root)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs = [(d, s) for d in DATASETS for s in SEEDS]
    # Prefer finishing smaller sets first is automatic via as_completed; schedule weibo last in list.
    print(f"Scheduling {len(jobs)} jobs on {gpus}", flush=True)
    gpu_q: Queue = Queue()
    for i in range(args.max_workers):
        gpu_q.put(gpus[i % len(gpus)])

    def _wrapped(job):
        d, s = job
        gpu = gpu_q.get()
        try:
            return _run_one(d, s, gpu, out_root, args.deterministic)
        finally:
            gpu_q.put(gpu)

    fails = 0
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as ex:
        futs = [ex.submit(_wrapped, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            _, _, rc, _ = fut.result()
            if rc != 0:
                fails += 1
            if i % 5 == 0 or i == len(futs):
                print(f"Progress {i}/{len(futs)} fails={fails}", flush=True)
    _aggregate(out_root)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
