#!/usr/bin/env python3
"""Schedule / proto minimal experiment (books / disney only).

Variants:
  single_uniform  — main: uniform; aux: none; cond: none
  dual_uniform    — main: uniform; aux: uniform; cond: none
  dual_schedule   — main: uniform; aux: logit-normal; cond: none
  original_proto  — main: uniform; aux: logit-normal; cond: prototype
  wider_single    — optional param-matched single (uniform)

5 seeds. Keep yaml polarity/smoothing (performance protocol).
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
DATASETS = ("books", "disney")
SEEDS = (0, 1, 2, 3, 42)
GUIDANCE_WEIGHT = 1.25

VARIANTS: Dict[str, Dict[str, Any]] = {
    "single_uniform": {
        "label": "Single Uniform",
        "main": "uniform",
        "aux": "无",
        "cond": "无",
        "overrides": {
            "dual_flow_mode": "single_full",
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": 0.0,
        },
    },
    "dual_uniform": {
        "label": "Dual Uniform",
        "main": "uniform",
        "aux": "uniform",
        "cond": "无",
        "overrides": {
            "dual_flow_mode": "dual_uniform",
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": GUIDANCE_WEIGHT,
        },
    },
    "dual_schedule": {
        "label": "Dual Schedule",
        "main": "uniform",
        "aux": "logit-normal",
        "cond": "无",
        "overrides": {
            "dual_flow_mode": "dual_schedule",
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": GUIDANCE_WEIGHT,
        },
    },
    "original_proto": {
        "label": "Original Proto",
        "main": "uniform",
        "aux": "logit-normal",
        "cond": "prototype",
        "overrides": {
            "dual_flow_mode": None,
            "use_proto": True,
            "use_proto_normal_weight": False,
            # weight from yaml
        },
    },
    "wider_single": {
        "label": "Wider Single",
        "main": "uniform (wider)",
        "aux": "无",
        "cond": "无",
        "overrides": {
            "dual_flow_mode": "wider_single",
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": 0.0,
        },
    },
}
VARIANT_ORDER = list(VARIANTS.keys())


def _run_one(variant: str, dataset: str, seed: int, gpu: int, out_root: Path) -> Dict[str, Any]:
    result_path = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / variant / f"{dataset}_seed{seed}.log"
    cfg_path = out_root / "configs" / variant / f"{dataset}_seed{seed}.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, Any] = {"variant": variant, "dataset": dataset, "seed": seed, "gpu": gpu}
    if result_path.exists():
        try:
            payload = json.load(open(result_path))
            row.update(
                {
                    "returncode": 0,
                    "auc": float(payload.get("auc_mean", payload.get("auc", float("nan")))),
                    "ap": float(payload.get("ap_mean", payload.get("ap", float("nan")))),
                    "elapsed_sec": 0.0,
                    "cached": True,
                }
            )
            print(f"[{variant}] {dataset} s{seed} cached auc={row['auc']:.4f}", flush=True)
            return row
        except Exception:
            pass

    cfg = yaml.safe_load(open(REPO / "configs" / f"{dataset}.yaml")) or {}
    cfg.update(deepcopy(VARIANTS[variant]["overrides"]))
    cfg["dataset"] = dataset
    cfg["sample_steps"] = 1
    cfg["num_trial"] = 1
    cfg["exp_tag"] = f"sched_{variant}_{dataset}"
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
    elapsed = time.time() - t0
    row["returncode"] = rc
    row["elapsed_sec"] = elapsed
    row["cached"] = False
    if rc == 0 and result_path.exists():
        payload = json.load(open(result_path))
        row["auc"] = float(payload.get("auc_mean", payload.get("auc", float("nan"))))
        row["ap"] = float(payload.get("ap_mean", payload.get("ap", float("nan"))))
    else:
        row["auc"] = float("nan")
        row["ap"] = float("nan")
    print(
        f"[{variant}] {dataset} s{seed} gpu={gpu} rc={rc} auc={row['auc']:.4f} {elapsed:.0f}s",
        flush=True,
    )
    return row


def _aggregate(out_root: Path, variants: List[str]) -> None:
    lines = [
        "# Schedule / proto minimal experiment (books / disney)",
        "",
        f"Guidance for dual variants: `w={GUIDANCE_WEIGHT}`. Seeds `{list(SEEDS)}`.",
        "Polarity / smoothing: keep dataset yaml.",
        "",
        "| Variant | Main | Aux | Cond | Books | Disney | Avg |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    summary: Dict[str, Any] = {}
    for variant in variants:
        meta = VARIANTS[variant]
        cells, means, per = [], [], {}
        for ds in DATASETS:
            aucs = []
            for seed in SEEDS:
                p = out_root / "results" / variant / f"{ds}_seed{seed}.json"
                if p.exists():
                    d = json.load(open(p))
                    aucs.append(float(d.get("auc_mean", d.get("auc"))))
            if len(aucs) == len(SEEDS):
                m, s = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
                cells.append(f"{m:.4f}±{s:.4f}")
                means.append(m)
                per[ds] = {"auc_mean": m, "auc_std": s}
            else:
                cells.append(f"nan({len(aucs)}/5)")
                per[ds] = {"n": len(aucs)}
        avg = float(np.mean(means)) if means else float("nan")
        lines.append(
            f"| {meta['label']} | {meta['main']} | {meta['aux']} | {meta['cond']} | "
            f"{cells[0]} | {cells[1]} | **{avg:.4f}** |"
        )
        summary[variant] = {**meta, "per_dataset": per, "avg_auc": avg}

    # Key contrasts
    lines += ["", "## Key contrasts", ""]
    def _avg(v):
        return summary.get(v, {}).get("avg_auc")

    pairs = [
        ("dual_uniform - single_uniform", "dual_uniform", "single_uniform", "第二分支（同 schedule）收益"),
        ("dual_schedule - dual_uniform", "dual_schedule", "dual_uniform", "仅改 aux 为 logit-normal 的收益"),
        ("original_proto - dual_schedule", "original_proto", "dual_schedule", "在 schedule 之上加 prototype 的收益"),
        ("original_proto - single_uniform", "original_proto", "single_uniform", "相对单 flow 总收益"),
    ]
    for name, a, b, desc in pairs:
        va, vb = _avg(a), _avg(b)
        if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb):
            lines.append(f"- {desc} (`{name}`): incomplete")
        else:
            lines.append(f"- {desc} (`{name}`): ΔAvg={va - vb:+.4f}")

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(summary_dir / "summary.json", "w"), indent=2)
    print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=str, default="0,2,3,5,6,7")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--output-dir", type=str, default="results/schedule_proto_miniex")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument(
        "--variants",
        type=str,
        default="single_uniform,dual_uniform,dual_schedule,original_proto,wider_single",
    )
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = (REPO / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    if args.aggregate_only:
        _aggregate(out_root, variants)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs: List[Tuple[str, str, int]] = []
    for v in variants:
        for ds in datasets:
            for seed in seeds:
                jobs.append((v, ds, seed))
    print(f"Jobs={len(jobs)} on GPUs {gpus} variants={variants}", flush=True)

    q: Queue = Queue()
    for i in range(args.max_workers):
        q.put(gpus[i % len(gpus)])

    def _wrap(job):
        v, ds, seed = job
        gpu = q.get()
        try:
            return _run_one(v, ds, seed, gpu, out_root)
        finally:
            q.put(gpu)

    fails = 0
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as ex:
        futs = [ex.submit(_wrap, j) for j in jobs]
        for fut in as_completed(futs):
            if fut.result().get("returncode", 1) != 0:
                fails += 1

    _aggregate(out_root, variants)
    print(f"Done. fails={fails}. Summary: {out_root / 'summary' / 'summary.md'}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
