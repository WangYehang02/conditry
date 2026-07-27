#!/usr/bin/env python3
"""Ablation: replace Flow Matching with Diffusion (EDM).

Pipeline kept: GAE + residual-augmented latent z + generative model + AE recon error.
Disabled (per ablation protocol): polarity / score orientation, prototype guidance.
"""
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
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
SEEDS = (0, 1, 2, 3, 42)

# DiffGAD-style EDM sampling steps (FM path uses 1; EDM needs multi-step).
DIFFUSION_SAMPLE_STEPS = 50

OVERRIDES: Dict[str, Any] = {
    "generative_backend": "diffusion",
    "use_proto": False,
    "polarity_enabled": False,
    "sample_steps": DIFFUSION_SAMPLE_STEPS,
}


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


def _pick_ap(payload: Dict[str, Any]) -> float:
    for k in ("ap", "ap_mean", "AP"):
        if k in payload and payload[k] is not None:
            return float(payload[k])
    return float("nan")


def _run_one(
    dataset: str,
    seed: int,
    gpu: int,
    out_root: Path,
    deterministic: bool,
) -> Tuple[str, int, int, float]:
    cfg = _load_yaml(REPO / "configs" / f"{dataset}.yaml")
    cfg.update(deepcopy(OVERRIDES))
    cfg["dataset"] = dataset
    cfg["exp_tag"] = f"ablation_diffusion_{dataset}"

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
    print(
        f"[diffusion] {dataset} s{seed} gpu={gpu} rc={rc} auc={auc:.4f} {time.time()-t0:.0f}s",
        flush=True,
    )
    return dataset, seed, rc, auc


def _aggregate(out_root: Path) -> Path:
    rows: Dict[str, List[float]] = {d: [] for d in DATASETS}
    ap_rows: Dict[str, List[float]] = {d: [] for d in DATASETS}
    for dataset in DATASETS:
        for seed in SEEDS:
            p = out_root / "results" / f"{dataset}_seed{seed}.json"
            if not p.exists():
                continue
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows[dataset].append(_pick_auc(payload))
            ap_rows[dataset].append(_pick_ap(payload))

    means: Dict[str, float] = {}
    disp = []
    for dataset in DATASETS:
        vals = rows[dataset]
        if len(vals) != len(SEEDS):
            means[dataset] = float("nan")
        else:
            means[dataset] = round(float(np.mean(vals)), 3)
            disp.append(means[dataset])
    means["avg"] = round(sum(disp) / len(disp), 3) if disp else float("nan")

    ap_means: Dict[str, float] = {}
    ap_disp = []
    for dataset in DATASETS:
        vals = ap_rows[dataset]
        if len(vals) != len(SEEDS) or any(np.isnan(v) for v in vals):
            ap_means[dataset] = float("nan")
        else:
            ap_means[dataset] = round(float(np.mean(vals)), 4)
            ap_disp.append(ap_means[dataset])
    ap_means["avg"] = round(sum(ap_disp) / len(ap_disp), 4) if ap_disp else float("nan")

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Diffusion Ablation (GAE + Residual + EDM + AE recon)",
        "",
        "Replaces Flow Matching with free-only EDM diffusion.",
        "Disabled: `use_proto=false`, `polarity_enabled=false`.",
        f"Sampling steps: `{DIFFUSION_SAMPLE_STEPS}`. Seeds: `{list(SEEDS)}`.",
        "",
        "| Variant | Books | Disney | Enron | Reddit | Weibo | Avg. |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Diffusion (no polarity / no guidance) | "
        + " | ".join(f"{means[d]:.3f}" if not np.isnan(means[d]) else "--" for d in DATASETS)
        + f" | {means['avg']:.3f} |"
        if not np.isnan(means["avg"])
        else "| Diffusion (no polarity / no guidance) | -- | -- | -- | -- | -- | -- |",
        "",
        "## AP (mean over seeds)",
        "",
        "| Dataset | AP |",
        "|---|---:|",
    ]
    for d in DATASETS:
        lines.append(f"| {d} | {ap_means[d]:.4f} |" if not np.isnan(ap_means[d]) else f"| {d} | -- |")
    lines.append(f"| Avg. | {ap_means['avg']:.4f} |" if not np.isnan(ap_means["avg"]) else "| Avg. | -- |")
    lines.append("")

    md_path = summary_dir / "diffusion_ablation_table.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with open(summary_dir / "diffusion_ablation_means.json", "w", encoding="utf-8") as f:
        json.dump({"auc": means, "ap": ap_means, "overrides": OVERRIDES}, f, indent=2)
    print(md_path.read_text(encoding="utf-8"), flush=True)
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=str, default=str(REPO / "results" / "diffusion_ablation"))
    ap.add_argument("--gpus", type=str, default="0,3,4,7")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(out_root)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    jobs = [(d, s) for d in datasets for s in SEEDS]
    print(f"Scheduling {len(jobs)} diffusion ablation jobs on GPUs {gpus}", flush=True)

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
    print(f"Done. fails={fails}. Summary: {out_root / 'summary' / 'diffusion_ablation_table.md'}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
