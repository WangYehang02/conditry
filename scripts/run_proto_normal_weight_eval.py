#!/usr/bin/env python3
"""Evaluate normality-weighted proto CFM with guidance weight forced on.

Enron/Reddit/Weibo defaults use weight=0 (proto unused at inference). For a fair
test of L_normal, force weight=1.25 on all datasets and compare:
  - control: use_proto_normal_weight=false (uniform proto CFM)
  - normal:  use_proto_normal_weight=true  (q_i-weighted proto CFM)
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
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
SEEDS = (0, 1, 2, 3, 42)
GUIDANCE_WEIGHT = 1.25

VARIANTS: Dict[str, Dict[str, Any]] = {
    "control_w125": {
        "label": "uniform proto CFM (w=1.25)",
        "overrides": {
            "weight": GUIDANCE_WEIGHT,
            "use_proto": True,
            "use_proto_normal_weight": False,
        },
    },
    "normal_w125": {
        "label": "normality-weighted proto CFM (w=1.25)",
        "overrides": {
            "weight": GUIDANCE_WEIGHT,
            "use_proto": True,
            "use_proto_normal_weight": True,
            "proto_normal_temp": 50.0,
        },
    },
}
VARIANT_ORDER = list(VARIANTS.keys())


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _pick(payload: Dict[str, Any], *keys: str) -> float:
    for k in keys:
        if k in payload and payload[k] is not None:
            return float(payload[k])
    raise KeyError(f"missing keys={keys} in {list(payload.keys())}")


def _run_one(
    variant: str,
    dataset: str,
    seed: int,
    gpu: int,
    out_root: Path,
    deterministic: bool,
) -> Tuple[str, str, int, int, float, float]:
    cfg = _load_yaml(REPO / "configs" / f"{dataset}.yaml")
    cfg.update(deepcopy(VARIANTS[variant]["overrides"]))
    cfg["dataset"] = dataset
    cfg["sample_steps"] = 1
    cfg["exp_tag"] = f"proto_nw_{variant}_{dataset}"

    cfg_path = out_root / "configs" / variant / f"{dataset}.yaml"
    result_path = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / variant / f"{dataset}_seed{seed}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return (
                variant,
                dataset,
                seed,
                0,
                _pick(payload, "auc", "auc_mean", "AUC"),
                _pick(payload, "ap", "ap_mean", "AP"),
            )
        except Exception:
            pass

    _save_yaml(cfg_path, cfg)
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
    ]
    if deterministic:
        cmd.append("--deterministic")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    auc = float("nan")
    ap = float("nan")
    if rc == 0 and result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        auc = _pick(payload, "auc", "auc_mean", "AUC")
        ap = _pick(payload, "ap", "ap_mean", "AP")
    print(
        f"[{variant}] {dataset} s{seed} gpu={gpu} rc={rc} "
        f"auc={auc:.4f} ap={ap:.4f} {time.time()-t0:.0f}s",
        flush=True,
    )
    return variant, dataset, seed, rc, auc, ap


def _aggregate(out_root: Path) -> Path:
    rows: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        v: {d: {"auc": [], "ap": []} for d in DATASETS} for v in VARIANT_ORDER
    }
    for variant in VARIANT_ORDER:
        for dataset in DATASETS:
            for seed in SEEDS:
                p = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
                if not p.exists():
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                rows[variant][dataset]["auc"].append(_pick(payload, "auc", "auc_mean", "AUC"))
                rows[variant][dataset]["ap"].append(_pick(payload, "ap", "ap_mean", "AP"))

    summary: Dict[str, Any] = {"guidance_weight": GUIDANCE_WEIGHT, "variants": {}}
    lines = [
        "# Proto normality-weighted CFM (guidance weight forced to 1.25)",
        "",
        "All datasets use `weight=1.25` so proto guidance is active at inference "
        "(including Enron/Reddit/Weibo, whose default yaml has `weight=0`).",
        "",
        "| Variant | Books | Disney | Enron | Reddit | Weibo | Avg AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        cells = []
        means = []
        per = {}
        for dataset in DATASETS:
            aucs = rows[variant][dataset]["auc"]
            aps = rows[variant][dataset]["ap"]
            if len(aucs) != len(SEEDS):
                cells.append("nan")
                per[dataset] = None
            else:
                m = float(np.mean(aucs))
                s = float(np.std(aucs, ddof=1))
                cells.append(f"{m:.4f}±{s:.4f}")
                means.append(m)
                per[dataset] = {
                    "auc_mean": m,
                    "auc_std": s,
                    "ap_mean": float(np.mean(aps)),
                    "ap_std": float(np.std(aps, ddof=1)),
                }
        avg = float(np.mean(means)) if means else float("nan")
        lines.append(
            f"| {VARIANTS[variant]['label']} | "
            + " | ".join(cells)
            + f" | **{avg:.4f}** |"
        )
        summary["variants"][variant] = {"per_dataset": per, "avg_auc": avg}

    if all(summary["variants"][v]["avg_auc"] == summary["variants"][v]["avg_auc"] for v in VARIANT_ORDER):
        delta = (
            summary["variants"]["normal_w125"]["avg_auc"]
            - summary["variants"]["control_w125"]["avg_auc"]
        )
        lines.extend(["", f"**ΔAvg (normal − control) AUROC = {delta:+.4f}**"])

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    md_path = summary_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(lines), flush=True)
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=str, default="2,3,4,7")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--output-dir", type=str, default="results/proto_normal_weight_w125")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER))
    args = ap.parse_args()

    out_root = (REPO / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    jobs: List[Tuple[str, str, int, int]] = []
    idx = 0
    for variant in variants:
        for dataset in datasets:
            for seed in SEEDS:
                jobs.append((variant, dataset, seed, int(gpus[idx % len(gpus)])))
                idx += 1

    print(
        f"Scheduling {len(jobs)} jobs on GPUs {gpus} "
        f"(forced weight={GUIDANCE_WEIGHT})",
        flush=True,
    )
    fails = 0
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs), len(gpus))) as ex:
        futs = [
            ex.submit(_run_one, v, d, s, g, out_root, bool(args.deterministic))
            for (v, d, s, g) in jobs
        ]
        for fut in as_completed(futs):
            *_, rc, _, _ = fut.result()
            if rc != 0:
                fails += 1

    _aggregate(out_root)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
