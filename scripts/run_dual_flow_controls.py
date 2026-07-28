#!/usr/bin/env python3
"""Residual-contrastive dual-flow control experiments.

Variants (paper table):
  single_full            — only full latent flow (base)
  single_ref             — only residual-suppressed flow
  duplicate_dual         — two full-latent flows (init diversity)
  residual_contrastive   — full + residual-suppressed (new method)
  wider_single           — param-matched wider single full flow
  shared_two_heads       — shared trunk, two heads

Weibo runs exclusively on WEIBO_GPU; other datasets share OTHER_GPUS.
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
GUIDANCE_WEIGHT = 1.25

VARIANTS: Dict[str, Dict[str, Any]] = {
    "single_full": {
        "label": "Single Full Flow",
        "overrides": {
            "dual_flow_mode": "single_full",
            "use_proto": False,
            "weight": 0.0,
        },
    },
    "single_ref": {
        "label": "Single Reference Flow",
        "overrides": {
            "dual_flow_mode": "single_ref",
            "use_proto": False,
            "weight": 0.0,
        },
    },
    "duplicate_dual": {
        "label": "Duplicate Dual Flow",
        "overrides": {
            "dual_flow_mode": "duplicate_dual",
            "use_proto": False,
            "weight": GUIDANCE_WEIGHT,
        },
    },
    "residual_contrastive": {
        "label": "Residual-Contrastive Dual Flow",
        "overrides": {
            "dual_flow_mode": "residual_contrastive",
            "use_proto": False,
            "weight": GUIDANCE_WEIGHT,
        },
    },
    "wider_single": {
        "label": "Wider Single Flow",
        "overrides": {
            "dual_flow_mode": "wider_single",
            "use_proto": False,
            "weight": 0.0,
        },
    },
    "shared_two_heads": {
        "label": "Shared Backbone, Two Heads",
        "overrides": {
            "dual_flow_mode": "shared_two_heads",
            "use_proto": False,
            "weight": GUIDANCE_WEIGHT,
        },
    },
}
VARIANT_ORDER = list(VARIANTS.keys())


def _run_one(variant: str, dataset: str, seed: int, gpu: int, out_root: Path) -> Tuple[str, str, int, int, float]:
    result_path = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / variant / f"{dataset}_seed{seed}.log"
    cfg_path = out_root / "configs" / variant / f"{dataset}.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        try:
            payload = json.load(open(result_path))
            auc = float(payload.get("auc_mean", payload.get("auc")))
            print(f"[{variant}] {dataset} s{seed} cached auc={auc:.4f}", flush=True)
            return variant, dataset, seed, 0, auc
        except Exception:
            pass

    cfg = yaml.safe_load(open(REPO / "configs" / f"{dataset}.yaml")) or {}
    cfg.update(deepcopy(VARIANTS[variant]["overrides"]))
    cfg["dataset"] = dataset
    cfg["sample_steps"] = 1
    cfg["num_trial"] = 1
    cfg["use_proto_normal_weight"] = False
    cfg["exp_tag"] = f"dualflow_{variant}_{dataset}"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
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
    print(
        f"[{variant}] {dataset} s{seed} gpu={gpu} rc={rc} auc={auc:.4f} {time.time()-t0:.0f}s",
        flush=True,
    )
    return variant, dataset, seed, rc, auc


def _aggregate(out_root: Path) -> None:
    lines = [
        "# Residual-contrastive dual-flow controls",
        "",
        f"Guidance weight for dual variants: `{GUIDANCE_WEIGHT}`. Seeds `{list(SEEDS)}`.",
        "",
        "| Variant | Books | Disney | Enron | Reddit | Weibo | Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary: Dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        cells = []
        means = []
        per = {}
        for ds in DATASETS:
            aucs = []
            for seed in SEEDS:
                p = out_root / "results" / variant / f"{ds}_seed{seed}.json"
                if p.exists():
                    d = json.load(open(p))
                    aucs.append(float(d.get("auc_mean", d.get("auc"))))
            if len(aucs) == len(SEEDS):
                m = float(np.mean(aucs))
                s = float(np.std(aucs, ddof=1))
                cells.append(f"{m:.4f}±{s:.4f}")
                means.append(m)
                per[ds] = {"auc_mean": m, "auc_std": s}
            else:
                cells.append(f"nan({len(aucs)}/5)")
                per[ds] = {"n": len(aucs)}
        avg = float(np.mean(means)) if means else float("nan")
        label = VARIANTS[variant]["label"]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | **{avg:.4f}** |")
        summary[variant] = {"label": label, "per_dataset": per, "avg_auc": avg}

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(summary_dir / "summary.json", "w"), indent=2)
    print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weibo-gpu", type=int, default=6)
    ap.add_argument("--other-gpus", type=str, default="0,2,5")
    ap.add_argument("--output-dir", type=str, default="results/dual_flow_controls")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER))
    args = ap.parse_args()

    out_root = (REPO / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    other_gpus = [int(x) for x in args.other_gpus.split(",") if x.strip()]
    weibo_gpu = int(args.weibo_gpu)

    jobs: List[Tuple[str, str, int]] = []
    for variant in variants:
        for ds in datasets:
            for seed in seeds:
                jobs.append((variant, ds, seed))

    weibo_jobs = [j for j in jobs if j[1] == "weibo"]
    other_jobs = [j for j in jobs if j[1] != "weibo"]
    print(
        f"Jobs={len(jobs)} (other={len(other_jobs)} on {other_gpus}, "
        f"weibo={len(weibo_jobs)} exclusive on {weibo_gpu})",
        flush=True,
    )

    fails = 0
    if other_jobs:
        q: Queue = Queue()
        for g in other_gpus:
            q.put(g)

        def _wrap(job):
            g = q.get()
            try:
                return _run_one(job[0], job[1], job[2], g, out_root)
            finally:
                q.put(g)

        with ThreadPoolExecutor(max_workers=len(other_gpus)) as ex:
            futs = [ex.submit(_wrap, j) for j in other_jobs]
            for fut in as_completed(futs):
                *_, rc, _ = fut.result()
                if rc != 0:
                    fails += 1

    for variant, ds, seed in weibo_jobs:
        *_, rc, _ = _run_one(variant, ds, seed, weibo_gpu, out_root)
        if rc != 0:
            fails += 1

    _aggregate(out_root)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
