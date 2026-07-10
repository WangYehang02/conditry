#!/usr/bin/env python3
"""Hyperparameter tuning for FMGAD: 5 datasets x 5 seeds, multi-GPU parallel."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
SEEDS = (42, 0, 1, 2, 3)
PY = os.environ.get("FMGAD_PYTHON", "/home/yehang/miniconda3/envs/fmgad/bin/python")

# Each trial: global overrides applied to all datasets, plus optional per-dataset overrides.
TUNING_TRIALS: Dict[str, Dict[str, Any]] = {
    "t00_baseline": {
        "desc": "Current repo configs (reference)",
        "global": {},
        "per_dataset": {},
    },
    "t01_smooth045": {
        "desc": "Stronger graph score smoothing",
        "global": {"score_smoothing_alpha": 0.45},
        "per_dataset": {},
    },
    "t02_smooth015": {
        "desc": "Lighter graph score smoothing",
        "global": {"score_smoothing_alpha": 0.15},
        "per_dataset": {},
    },
    "t03_polar095": {
        "desc": "More weight on local_prior rank in polarity blend",
        "global": {"polarity_consensus_score_weight": 0.95},
        "per_dataset": {},
    },
    "t04_polar080": {
        "desc": "More weight on main score rank; lower flip threshold",
        "global": {
            "polarity_consensus_score_weight": 0.80,
            "polarity_consensus_threshold": 0.65,
        },
        "per_dataset": {},
    },
    "t05_virtual_k8_t3": {
        "desc": "More aggressive virtual neighbor supplementation",
        "global": {"virtual_degree_threshold": 3, "virtual_k": 8},
        "per_dataset": {},
    },
    "t06_disney_refined": {
        "desc": "Disney-specific overrides from historical best search",
        "global": {},
        "per_dataset": {
            "disney": {
                "ae_alpha": 0.55,
                "ae_dropout": 0.35,
                "ae_lr": 0.025,
                "proto_alpha": 0.005,
                "residual_scale": 10.0,
                "weight": 1.75,
                "score_smoothing_alpha": 0.25,
            },
        },
    },
    "t07_perdataset_v1": {
        "desc": "Per-dataset heuristic combo (residual + smoothing + polarity)",
        "global": {},
        "per_dataset": {
            "books": {"residual_scale": 12.0, "score_smoothing_alpha": 0.40, "polarity_consensus_score_weight": 0.95},
            "disney": {"residual_scale": 8.0, "weight": 1.5, "score_smoothing_alpha": 0.25, "proto_alpha": 0.004},
            "enron": {"score_smoothing_alpha": 0.15, "polarity_consensus_score_weight": 0.95, "residual_scale": 15.0},
            "reddit": {"residual_scale": 15.0, "score_smoothing_alpha": 0.40, "weight": 0.5},
            "weibo": {"score_smoothing_alpha": 0.20, "proto_alpha": 0.002},
        },
    },
    "t08_combo_global": {
        "desc": "Global combo: moderate smooth + polar095 + virtual k8",
        "global": {
            "score_smoothing_alpha": 0.35,
            "polarity_consensus_score_weight": 0.95,
            "virtual_degree_threshold": 4,
            "virtual_k": 6,
        },
        "per_dataset": {},
    },
}


def _merge_config(base: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg.update(overrides)
    return cfg


def _write_trial_configs(trial_id: str, trial: dict, config_root: Path) -> Path:
    trial_dir = config_root / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    meta = {"trial_id": trial_id, "desc": trial.get("desc", ""), "datasets": {}}
    global_ov = trial.get("global") or {}
    per_ds = trial.get("per_dataset") or {}
    for ds in DATASETS:
        with open(REPO / "configs" / f"{ds}.yaml", "r", encoding="utf-8") as f:
            base = yaml.load(f, Loader=yaml.Loader)
        ov = copy.deepcopy(global_ov)
        ov.update(per_ds.get(ds, {}))
        cfg = _merge_config(base, ov)
        cfg["exp_tag"] = f"tune_{trial_id}_{ds}"
        out_path = trial_dir / f"{ds}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        meta["datasets"][ds] = {"config": str(out_path), "overrides": ov}
    with open(trial_dir / "trial_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return trial_dir


def _run_one(job: Tuple[str, str, int, str, str, Path, Path, Path, bool]) -> Dict[str, Any]:
    trial_id, dataset, seed, gpu, py_exe, out_root, model_root, repo, deterministic = job
    result_path = out_root / trial_id / f"{dataset}_seed{seed}.json"
    log_path = out_root.parent / "logs" / trial_id / f"{dataset}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = out_root.parent / "configs" / trial_id / f"{dataset}.yaml"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str(model_root / trial_id)
    env["FMGAD_RUN_TAG_SUFFIX"] = f"seed{seed}"
    cmd = [
        py_exe,
        str(repo / "main_train.py"),
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
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    row: Dict[str, Any] = {
        "trial_id": trial_id,
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "result_file": str(result_path),
        "log_file": str(log_path),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        if proc.stderr:
            f.write("\n--- stderr ---\n")
            f.write(proc.stderr)
    if proc.returncode == 0 and result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        row.update(
            {
                "auc": payload.get("auc_mean"),
                "ap": payload.get("ap_mean"),
                "rec": payload.get("rec_mean"),
                "auprc": payload.get("auprc_mean"),
                "f1": payload.get("f1_mean"),
                "polarity_flipped": (payload.get("polarity_diagnostics") or {}).get("flipped"),
            }
        )
    else:
        row["stderr_tail"] = (proc.stderr or "")[-1500:]
    return row


def _aggregate(out_root: Path, summary_dir: Path, trial_ids: List[str]) -> Dict[str, Any]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    per_trial: Dict[str, Any] = {}

    for trial_id in trial_ids:
        trial_rows = []
        for p in sorted((out_root / trial_id).glob("*_seed*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "dataset": data["dataset"],
                    "seed": data["seed"],
                    "auc": float(data.get("auc_mean", data.get("auc", 0))),
                    "ap": float(data.get("ap_mean", 0)),
                    "rec": float(data.get("rec_mean", 0)),
                    "auprc": float(data.get("auprc_mean", 0)),
                    "f1": float(data.get("f1_mean", 0)),
                    "time_sec": float(data.get("time_sec", 0)),
                    "result_file": str(p),
                }
            )
        if not trial_rows:
            continue
        all_rows.extend(trial_rows)
        by_ds: Dict[str, List[dict]] = {ds: [] for ds in DATASETS}
        for r in trial_rows:
            by_ds[r["dataset"]].append(r)
        ds_summary = {}
        aucs, aps = [], []
        for ds in DATASETS:
            rs = by_ds[ds]
            if not rs:
                continue
            m_auc = sum(x["auc"] for x in rs) / len(rs)
            m_ap = sum(x["ap"] for x in rs) / len(rs)
            ds_summary[ds] = {
                "n_seeds": len(rs),
                "mean_auc": m_auc,
                "mean_ap": m_ap,
                "per_seed": rs,
            }
            aucs.extend(x["auc"] for x in rs)
            aps.extend(x["ap"] for x in rs)
        per_trial[trial_id] = {
            "n_runs": len(trial_rows),
            "mean_auc_all": sum(aucs) / len(aucs) if aucs else None,
            "mean_ap_all": sum(aps) / len(aps) if aps else None,
            "by_dataset": ds_summary,
        }

    csv_path = summary_dir / "all_runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "dataset",
                "seed",
                "auc",
                "ap",
                "rec",
                "auprc",
                "f1",
                "time_sec",
                "result_file",
            ],
        )
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    ranking = sorted(
        per_trial.items(),
        key=lambda kv: (kv[1]["mean_auc_all"] or 0.0, kv[1]["mean_ap_all"] or 0.0),
        reverse=True,
    )
    summary = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trials": per_trial,
        "ranking_by_mean_auc": [
            {"trial_id": tid, "mean_auc_all": info["mean_auc_all"], "mean_ap_all": info["mean_ap_all"]}
            for tid, info in ranking
        ],
    }
    json_path = summary_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# FMGAD Tuning Summary",
        "",
        f"Updated: {summary['updated_at']}",
        "",
        "## Trial Ranking (by mean AUC over 5 datasets x seeds)",
        "",
        "| trial | mean_auc | mean_ap |",
        "|-------|----------|---------|",
    ]
    for item in summary["ranking_by_mean_auc"]:
        md_lines.append(
            f"| {item['trial_id']} | {item['mean_auc_all']:.4f} | {item['mean_ap_all']:.4f} |"
        )
    md_lines.extend(["", "## Per-dataset best trial", ""])
    best_per_ds: Dict[str, Tuple[str, float, float]] = {}
    for ds in DATASETS:
        best = ("", 0.0, 0.0)
        for tid, info in per_trial.items():
            ds_info = info["by_dataset"].get(ds)
            if ds_info and ds_info["mean_auc"] > best[1]:
                best = (tid, ds_info["mean_auc"], ds_info["mean_ap"])
        best_per_ds[ds] = best
        md_lines.append(f"- **{ds}**: {best[0]} (AUC={best[1]:.4f}, AP={best[2]:.4f})")
    (summary_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=str, default="/mnt/yehang/调参")
    ap.add_argument("--trials", type=str, default=None, help="Comma-separated trial ids (default: all)")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,7")
    ap.add_argument("--max-workers", type=int, default=7)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root).expanduser().resolve()
    config_root = out_root / "configs"
    result_root = out_root / "results"
    model_root = out_root / "models"
    summary_dir = out_root / "summary"
    for p in (config_root, result_root, model_root, out_root / "logs", summary_dir):
        p.mkdir(parents=True, exist_ok=True)

    trial_ids = [t.strip() for t in (args.trials.split(",") if args.trials else TUNING_TRIALS.keys()) if t.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if args.aggregate_only:
        _aggregate(result_root, summary_dir, trial_ids)
        print("Wrote summary to", summary_dir)
        return 0

    jobs: List[Tuple[str, str, int, str, str, Path, Path, Path, bool]] = []
    for trial_id in trial_ids:
        if trial_id not in TUNING_TRIALS:
            print("Unknown trial:", trial_id, file=sys.stderr)
            return 1
        _write_trial_configs(trial_id, TUNING_TRIALS[trial_id], config_root)
        (result_root / trial_id).mkdir(parents=True, exist_ok=True)
        for i, ds in enumerate(datasets):
            for j, seed in enumerate(seeds):
                idx = i * len(seeds) + j
                gpu = gpus[idx % len(gpus)]
                jobs.append(
                    (trial_id, ds, seed, gpu, PY, result_root, model_root, REPO, bool(args.deterministic))
                )

    print(f"Scheduling {len(jobs)} jobs across GPUs {gpus}", flush=True)
    rows: List[Dict[str, Any]] = []
    mw = min(args.max_workers, len(jobs), max(len(gpus), 1))
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_run_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print(
                f"[{k}/{len(jobs)}] {row['trial_id']} {row['dataset']} seed={row['seed']} "
                f"rc={row['returncode']} auc={row.get('auc')} ap={row.get('ap')}",
                flush=True,
            )

    run_log = summary_dir / "run_log.json"
    run_log.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _aggregate(result_root, summary_dir, trial_ids)
    print("\n=== Top trials ===")
    for item in summary["ranking_by_mean_auc"][:5]:
        print(f"  {item['trial_id']}: auc={item['mean_auc_all']:.4f} ap={item['mean_ap_all']:.4f}")
    print("Summary:", summary_dir / "summary.json")
    return 0 if all(r.get("returncode") == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
