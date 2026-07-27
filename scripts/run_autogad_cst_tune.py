#!/usr/bin/env python3
"""Truly-unsupervised HP search for RALFlow-GAD (AutoGAD-style CST).

Inspired by AutoGAD2024 (Li et al., DAMI 2025): select hyperparameters by an
internal score statistic (CST) computed from anomaly scores only — never by AUROC.

Parameters searched (reasons as provided by user):
  weight, proto_alpha, residual_scale,
  use_virtual_neighbors, virtual_degree_threshold, virtual_k

Note: score smoothing is permanently enabled; flow_t_sampling is fixed to logit_normal.
`score_smoothing_alpha` remains a per-dataset config (not searched here unless listed).

Selection: maximize mean CST over selection seeds.
AUROC/AP are logged only for post-hoc analysis (not used for selection).
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
# Fewer seeds for CST selection to keep wall-clock tractable; winners can be re-checked later.
# Two seeds balance CST stability vs wall-clock (weibo is expensive).
SELECT_SEEDS = (0, 42)


def compute_cst(scores: np.ndarray, ratio: float = 0.05) -> float:
    """AutoGAD custom statistic (CST): separation of top-ratio vs the rest."""
    data = np.asarray(scores, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size < 4:
        return float("nan")
    sorted_data = np.sort(data)[::-1]
    k = max(1, int(len(sorted_data) * ratio))
    if k >= len(sorted_data):
        k = max(1, len(sorted_data) // 5)
    A = sorted_data[:k]
    B = sorted_data[k:]
    if B.size < 2 or A.size < 2:
        return float("nan")
    mean_a, mean_b = float(np.mean(A)), float(np.mean(B))
    var_a, var_b = float(np.var(A, ddof=1)), float(np.var(B, ddof=1))
    denom = np.sqrt(max(var_a + var_b, 1e-12))
    return float((mean_a - mean_b) / denom)


def build_trials(dataset: str) -> Dict[str, Dict[str, Any]]:
    """One-factor-at-a-time candidates on top of current best yaml."""
    trials: Dict[str, Dict[str, Any]] = {
        "baseline": {"desc": "current best yaml", "overrides": {}},
    }

    # residual_scale — changes residual magnitude in z
    for v in (5.0, 10.0, 15.0, 20.0, 25.0):
        trials[f"res_{int(v)}"] = {
            "desc": f"residual_scale={v}",
            "overrides": {"residual_scale": float(v)},
        }

    # weight / proto_alpha — only meaningful when guidance is used; still search all sets
    for v in (0.0, 0.5, 1.0, 1.25, 2.5):
        trials[f"w_{str(v).replace('.', '')}"] = {
            "desc": f"weight={v}",
            "overrides": {"weight": float(v)},
        }
    for v in (0.001, 0.003, 0.005, 0.01):
        tag = str(v).replace(".", "")
        trials[f"pa_{tag}"] = {
            "desc": f"proto_alpha={v}",
            "overrides": {"proto_alpha": float(v)},
        }

    # virtual neighbors
    trials["virt_off"] = {
        "desc": "use_virtual_neighbors=false",
        "overrides": {"use_virtual_neighbors": False},
    }
    for thr, k in ((3, 5), (3, 6), (4, 6), (5, 5), (5, 8)):
        trials[f"virt_t{thr}_k{k}"] = {
            "desc": f"virtual thr={thr} k={k}",
            "overrides": {
                "use_virtual_neighbors": True,
                "virtual_degree_threshold": int(thr),
                "virtual_k": int(k),
            },
        }

    # Dataset-specific pruning: drop trials identical to baseline defaults to reduce load a bit
    # (still keep baseline). No further pruning — user asked to tune all listed knobs.
    _ = dataset
    return trials


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _pick_gpus(min_free_mb: int = 12000) -> List[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return [0]
    cands = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, free, util = int(parts[0]), int(float(parts[1])), int(float(parts[2]))
        if free >= min_free_mb and util <= 30:
            cands.append((free, idx))
    cands.sort(reverse=True)
    return [idx for _, idx in cands] or [0]


def _run_one(
    trial_id: str,
    dataset: str,
    seed: int,
    overrides: Dict[str, Any],
    gpu: int,
    out_root: Path,
    deterministic: bool,
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
        "result_file": str(result_path),
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
                return row
        except Exception:
            pass

    cfg = _load_yaml(REPO / "configs" / f"{dataset}.yaml")
    cfg.update(deepcopy(overrides))
    cfg["dataset"] = dataset
    cfg["exp_tag"] = f"autogad_tune_{trial_id}_{dataset}"
    cfg["ensemble_score"] = True
    cfg["num_trial"] = 1
    _save_yaml(cfg_path, cfg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"{trial_id}_s{seed}"
    env["FMGAD_SAVE_SCORES"] = "1"
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
    elapsed = time.time() - t0
    row["returncode"] = rc
    row["elapsed_sec"] = elapsed
    row["cached"] = False
    if rc == 0 and result_path.exists():
        payload = json.load(open(result_path))
        scores = payload.get("scores")
        row["auc"] = float(payload.get("auc_mean", payload.get("auc", float("nan"))))
        row["ap"] = float(payload.get("ap_mean", payload.get("ap", float("nan"))))
        row["cst"] = compute_cst(np.asarray(scores, dtype=np.float64)) if scores is not None else float("nan")
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


def _aggregate(out_root: Path, trials_meta: Dict[str, Dict[str, Dict[str, Any]]]) -> Path:
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    # Collect
    for dataset, trials in trials_meta.items():
        for trial_id in trials:
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
                        "desc": trials[trial_id]["desc"],
                        "overrides": trials[trial_id]["overrides"],
                    }
                )

    with open(summary_dir / "all_runs.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "dataset", "seed", "cst", "auc", "ap", "desc"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    best_by_ds: Dict[str, Any] = {}
    lines = [
        "# AutoGAD-style CST Hyperparameter Search",
        "",
        "Selection criterion: **maximize mean CST** over seeds "
        f"`{list(SELECT_SEEDS)}` (label-free). AUROC is reported only for reference.",
        "",
    ]
    for dataset in DATASETS:
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
            mean_cst = float(np.mean([r["cst"] for r in rs]))
            mean_auc = float(np.mean([r["auc"] for r in rs]))
            mean_ap = float(np.mean([r["ap"] for r in rs]))
            ranked.append(
                {
                    "trial_id": tid,
                    "mean_cst": mean_cst,
                    "mean_auc": mean_auc,
                    "mean_ap": mean_ap,
                    "desc": rs[0]["desc"],
                    "overrides": rs[0]["overrides"],
                    "n": len(rs),
                }
            )
        ranked.sort(key=lambda x: x["mean_cst"], reverse=True)
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| rank | trial | mean_CST | mean_AUC† | mean_AP† | desc |")
        lines.append("|---:|---|---:|---:|---:|---|")
        for i, info in enumerate(ranked[:15], 1):
            lines.append(
                f"| {i} | `{info['trial_id']}` | {info['mean_cst']:.4f} | "
                f"{info['mean_auc']:.4f} | {info['mean_ap']:.4f} | {info['desc']} |"
            )
        lines.append("")
        lines.append("† AUROC/AP are **not** used for selection (AutoGAD protocol).")
        lines.append("")
        if ranked:
            best = ranked[0]
            base = next((x for x in ranked if x["trial_id"] == "baseline"), None)
            best_by_ds[dataset] = {
                **best,
                "baseline_cst": None if base is None else base["mean_cst"],
                "baseline_auc": None if base is None else base["mean_auc"],
                "delta_cst": None if base is None else best["mean_cst"] - base["mean_cst"],
            }
            lines.append(
                f"- **CST-best**: `{best['trial_id']}` CST={best['mean_cst']:.4f} "
                f"(AUC†={best['mean_auc']:.4f})"
                + (
                    f", ΔCST vs baseline={best_by_ds[dataset]['delta_cst']:+.4f}"
                    if best_by_ds[dataset]["delta_cst"] is not None
                    else ""
                )
            )
            lines.append("")

    summary = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "select_seeds": list(SELECT_SEEDS),
        "selection_metric": "mean_CST",
        "best_by_dataset": best_by_ds,
    }
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = summary_dir / "summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md.read_text(encoding="utf-8"), flush=True)
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=str, default=str(REPO / "results" / "autogad_cst_tune"))
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--gpus", type=str, default="auto")
    ap.add_argument("--min-free-mb", type=int, default=12000)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--trials", type=str, default=None, help="Comma-separated trial ids (optional filter)")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    trials_meta = {ds: build_trials(ds) for ds in datasets}
    if args.trials:
        keep = {x.strip() for x in args.trials.split(",") if x.strip()}
        trials_meta = {ds: {k: v for k, v in trials.items() if k in keep} for ds, trials in trials_meta.items()}

    if args.aggregate_only:
        _aggregate(out_root, trials_meta)
        return 0

    if args.gpus.strip().lower() == "auto":
        gpus = _pick_gpus(args.min_free_mb)
    else:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    print(f"Idle/usable GPUs: {gpus}", flush=True)

    jobs: List[Tuple[str, str, int, Dict[str, Any]]] = []
    for ds in datasets:
        for tid, meta in trials_meta[ds].items():
            for seed in SELECT_SEEDS:
                jobs.append((tid, ds, seed, meta["overrides"]))
    # Prefer small datasets first so GPUs fill quickly; weibo last within each trial batch
    order = {d: i for i, d in enumerate(("disney", "books", "enron", "reddit", "weibo"))}
    jobs.sort(key=lambda j: (order.get(j[1], 99), j[0], j[2]))
    print(f"Scheduling {len(jobs)} jobs on {gpus} (max_workers={args.max_workers})", flush=True)

    gpu_q: Queue = Queue()
    for i in range(args.max_workers):
        gpu_q.put(gpus[i % len(gpus)])

    def _wrapped(job):
        tid, ds, seed, ov = job
        gpu = gpu_q.get()
        try:
            return _run_one(tid, ds, seed, ov, gpu, out_root, args.deterministic)
        finally:
            gpu_q.put(gpu)

    fails = 0
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as ex:
        futs = [ex.submit(_wrapped, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if row.get("returncode", 1) != 0:
                fails += 1
            if i % 10 == 0 or i == len(futs):
                print(f"Progress {i}/{len(futs)} fails={fails}", flush=True)

    _aggregate(out_root, trials_meta)
    print(f"Done. fails={fails}. Summary: {out_root / 'summary' / 'summary.md'}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
