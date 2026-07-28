#!/usr/bin/env python3
"""CST (AutoGAD-style) HP search for normality-weighted proto CFM.

Scope: books + disney only (user request after fair w=1.25 eval underperformed).

Selection: maximize mean CST over SELECT_SEEDS — label-free.
AUROC/AP logged for reference only (never used for selection).

Search (one-factor-at-a-time on yaml base + use_proto=True):
  - control_uniform: normality weight OFF
  - proto_normal_temp τ grid with weight=1.25, normality ON
  - guidance weight grid with τ=50, normality ON
  - proto_alpha grid with τ=50, weight=1.25, normality ON
"""
from __future__ import annotations

import argparse
import csv
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
SELECT_SEEDS = (0, 42)
BASE_NORMAL = {
    "use_proto": True,
    "use_proto_normal_weight": True,
    "weight": 1.25,
    "proto_normal_temp": 50.0,
}


def compute_cst(scores: np.ndarray, ratio: float = 0.05) -> float:
    data = np.asarray(scores, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size < 4:
        return float("nan")
    sorted_data = np.sort(data)[::-1]
    k = max(1, int(len(sorted_data) * ratio))
    if k >= len(sorted_data):
        k = max(1, len(sorted_data) // 5)
    A, B = sorted_data[:k], sorted_data[k:]
    if B.size < 2 or A.size < 2:
        return float("nan")
    mean_a, mean_b = float(np.mean(A)), float(np.mean(B))
    var_a, var_b = float(np.var(A, ddof=1)), float(np.var(B, ddof=1))
    denom = np.sqrt(max(var_a + var_b, 1e-12))
    return float((mean_a - mean_b) / denom)


def build_trials() -> Dict[str, Dict[str, Any]]:
    trials: Dict[str, Dict[str, Any]] = {
        "control_uniform": {
            "desc": "uniform proto CFM (normality OFF, w=1.25)",
            "overrides": {
                "use_proto": True,
                "use_proto_normal_weight": False,
                "weight": 1.25,
            },
        },
        "normal_default": {
            "desc": "normality ON τ=50 w=1.25 (default)",
            "overrides": dict(BASE_NORMAL),
        },
    }
    for tau in (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        tag = str(int(tau)) if tau == int(tau) else str(tau).replace(".", "")
        ov = dict(BASE_NORMAL)
        ov["proto_normal_temp"] = float(tau)
        trials[f"tau_{tag}"] = {
            "desc": f"normality ON proto_normal_temp={tau}",
            "overrides": ov,
        }
    for w in (0.0, 0.5, 1.0, 1.25, 2.5):
        tag = str(w).replace(".", "")
        ov = dict(BASE_NORMAL)
        ov["weight"] = float(w)
        trials[f"w_{tag}"] = {
            "desc": f"normality ON weight={w} τ=50",
            "overrides": ov,
        }
    for pa in (0.001, 0.003, 0.005, 0.01):
        tag = str(pa).replace(".", "")
        ov = dict(BASE_NORMAL)
        ov["proto_alpha"] = float(pa)
        trials[f"pa_{tag}"] = {
            "desc": f"normality ON proto_alpha={pa} τ=50 w=1.25",
            "overrides": ov,
        }
    return trials


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _run_one(
    trial_id: str,
    dataset: str,
    seed: int,
    overrides: Dict[str, Any],
    gpu: int,
    out_root: Path,
) -> Dict[str, Any]:
    result_path = out_root / "results" / trial_id / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / trial_id / f"{dataset}_seed{seed}.log"
    cfg_path = out_root / "configs" / trial_id / f"{dataset}.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, Any] = {
        "trial_id": trial_id,
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
    }
    if result_path.exists():
        try:
            payload = json.load(open(result_path))
            scores = payload.get("scores")
            if scores is not None:
                row.update(
                    {
                        "returncode": 0,
                        "auc": float(payload.get("auc_mean", payload.get("auc", float("nan")))),
                        "ap": float(payload.get("ap_mean", payload.get("ap", float("nan")))),
                        "cst": compute_cst(np.asarray(scores, dtype=np.float64)),
                        "elapsed_sec": 0.0,
                        "cached": True,
                    }
                )
                print(
                    f"[{trial_id}] {dataset} s{seed} cached "
                    f"cst={row['cst']:.4f} auc={row['auc']:.4f}",
                    flush=True,
                )
                return row
        except Exception:
            pass

    cfg = _load_yaml(REPO / "configs" / f"{dataset}.yaml")
    cfg.update(deepcopy(overrides))
    cfg["dataset"] = dataset
    cfg["exp_tag"] = f"proto_normal_cst_{trial_id}_{dataset}"
    cfg["ensemble_score"] = True
    cfg["num_trial"] = 1
    cfg["sample_steps"] = 1
    _save_yaml(cfg_path, cfg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"{trial_id}_s{seed}"
    env["FMGAD_SAVE_SCORES"] = "1"
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
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    row["returncode"] = rc
    row["elapsed_sec"] = elapsed
    row["cached"] = False
    if rc == 0 and result_path.exists():
        payload = json.load(open(result_path))
        scores = payload.get("scores")
        row["auc"] = float(payload.get("auc_mean", payload.get("auc", float("nan"))))
        row["ap"] = float(payload.get("ap_mean", payload.get("ap", float("nan"))))
        row["cst"] = (
            compute_cst(np.asarray(scores, dtype=np.float64)) if scores is not None else float("nan")
        )
    else:
        row["auc"] = float("nan")
        row["ap"] = float("nan")
        row["cst"] = float("nan")
    print(
        f"[{trial_id}] {dataset} s{seed} gpu={gpu} rc={rc} "
        f"cst={row['cst']:.4f} auc={row['auc']:.4f} {elapsed:.0f}s",
        flush=True,
    )
    return row


def _aggregate(out_root: Path, trials: Dict[str, Dict[str, Any]], datasets: List[str]) -> Path:
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for trial_id, meta in trials.items():
            for seed in SELECT_SEEDS:
                p = out_root / "results" / trial_id / f"{dataset}_seed{seed}.json"
                if not p.exists():
                    continue
                payload = json.load(open(p))
                scores = payload.get("scores")
                all_rows.append(
                    {
                        "trial_id": trial_id,
                        "dataset": dataset,
                        "seed": seed,
                        "auc": float(payload.get("auc_mean", payload.get("auc", float("nan")))),
                        "ap": float(payload.get("ap_mean", payload.get("ap", float("nan")))),
                        "cst": compute_cst(np.asarray(scores, dtype=np.float64)) if scores else float("nan"),
                        "desc": meta["desc"],
                        "overrides": meta["overrides"],
                    }
                )

    with open(summary_dir / "all_runs.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "dataset", "seed", "cst", "auc", "ap", "desc"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    best_by_ds: Dict[str, Any] = {}
    lines = [
        "# Proto-normality CST tune (books / disney)",
        "",
        "Selection: **maximize mean CST** over seeds "
        f"`{list(SELECT_SEEDS)}` (label-free). AUROC† for reference only.",
        "",
        "Base: `use_proto=True`, search `proto_normal_temp` / `weight` / `proto_alpha`.",
        "",
    ]
    for dataset in datasets:
        rows = [r for r in all_rows if r["dataset"] == dataset]
        by_trial: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_trial.setdefault(r["trial_id"], []).append(r)
        ranked = []
        for tid, rs in by_trial.items():
            if len(rs) < len(SELECT_SEEDS):
                continue
            if any(not np.isfinite(r["cst"]) for r in rs):
                continue
            ranked.append(
                {
                    "trial_id": tid,
                    "mean_cst": float(np.mean([r["cst"] for r in rs])),
                    "mean_auc": float(np.mean([r["auc"] for r in rs])),
                    "mean_ap": float(np.mean([r["ap"] for r in rs])),
                    "desc": rs[0]["desc"],
                    "overrides": rs[0]["overrides"],
                }
            )
        ranked.sort(key=lambda x: x["mean_cst"], reverse=True)
        lines += [
            f"## {dataset}",
            "",
            "| rank | trial | mean_CST | mean_AUC† | mean_AP† | desc |",
            "|---:|---|---:|---:|---:|---|",
        ]
        for i, info in enumerate(ranked[:20], 1):
            lines.append(
                f"| {i} | `{info['trial_id']}` | {info['mean_cst']:.4f} | "
                f"{info['mean_auc']:.4f} | {info['mean_ap']:.4f} | {info['desc']} |"
            )
        lines.append("")
        ctrl = next((x for x in ranked if x["trial_id"] == "control_uniform"), None)
        best = ranked[0] if ranked else None
        if best is not None:
            best_by_ds[dataset] = {
                **best,
                "control_cst": None if ctrl is None else ctrl["mean_cst"],
                "control_auc": None if ctrl is None else ctrl["mean_auc"],
                "delta_cst_vs_control": None
                if ctrl is None
                else best["mean_cst"] - ctrl["mean_cst"],
            }
            dlt = best_by_ds[dataset]["delta_cst_vs_control"]
            lines.append(
                f"- **CST-best**: `{best['trial_id']}` CST={best['mean_cst']:.4f} "
                f"(AUC†={best['mean_auc']:.4f})"
                + (f", ΔCST vs control={dlt:+.4f}" if dlt is not None else "")
            )
            if ctrl is not None:
                lines.append(
                    f"- **control_uniform**: CST={ctrl['mean_cst']:.4f} "
                    f"(AUC†={ctrl['mean_auc']:.4f})"
                )
            lines.append("")

    summary = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "select_seeds": list(SELECT_SEEDS),
        "selection_metric": "mean_CST",
        "best_by_dataset": best_by_ds,
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = summary_dir / "summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md.read_text(encoding="utf-8"), flush=True)
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=str,
        default=str(REPO / "results" / "proto_normal_cst_tune"),
    )
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--gpus", type=str, default="0,2,3,6")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    trials = build_trials()

    if args.aggregate_only:
        _aggregate(out_root, trials, datasets)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs: List[Tuple[str, str, int, Dict[str, Any]]] = []
    for ds in datasets:
        for tid, meta in trials.items():
            for seed in SELECT_SEEDS:
                jobs.append((tid, ds, seed, meta["overrides"]))
    jobs.sort(key=lambda j: (0 if j[1] == "disney" else 1, j[0], j[2]))
    print(
        f"Trials={len(trials)} datasets={datasets} seeds={list(SELECT_SEEDS)} "
        f"→ {len(jobs)} jobs on GPUs {gpus}",
        flush=True,
    )

    gpu_q: Queue = Queue()
    for i in range(args.max_workers):
        gpu_q.put(gpus[i % len(gpus)])

    def _wrap(job):
        tid, ds, seed, ov = job
        gpu = gpu_q.get()
        try:
            return _run_one(tid, ds, seed, ov, gpu, out_root)
        finally:
            gpu_q.put(gpu)

    fails = 0
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as ex:
        futs = [ex.submit(_wrap, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if row.get("returncode", 1) != 0:
                fails += 1
            if i % 10 == 0 or i == len(futs):
                print(f"Progress {i}/{len(futs)} fails={fails}", flush=True)

    _aggregate(out_root, trials, datasets)
    print(f"Done. fails={fails}. Summary: {out_root / 'summary' / 'summary.md'}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
