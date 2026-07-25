#!/usr/bin/env python3
"""Focused books/reddit hyperparameter search for FMGAD (better).

Runs multi-GPU parallel jobs, picks best 5-seed mean AUC configs, and can
optionally write winners back into configs/*.yaml.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
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
PY = os.environ.get("FMGAD_PYTHON", "/home/yehang/miniconda3/envs/fmgad/bin/python")
SEEDS = (0, 1, 2, 3, 42)
DATASETS = ("books", "reddit")

# Overrides applied on top of configs/<dataset>.yaml
TRIALS: Dict[str, Dict[str, Any]] = {
    # ----- books -----
    "b0_baseline": {
        "dataset": "books",
        "desc": "Current books.yaml",
        "overrides": {},
    },
    "b1_virt3": {
        "dataset": "books",
        "desc": "virtual_degree_threshold=3, virtual_k=8 (1D best historically)",
        "overrides": {"virtual_degree_threshold": 3, "virtual_k": 8},
    },
    "b2_res25": {
        "dataset": "books",
        "desc": "residual_scale=25",
        "overrides": {"residual_scale": 25.0},
    },
    "b3_proto02": {
        "dataset": "books",
        "desc": "proto_alpha=0.02",
        "overrides": {"proto_alpha": 0.02},
    },
    "b4_res25_virt3": {
        "dataset": "books",
        "desc": "residual_scale=25 + virtual_t3_k8",
        "overrides": {
            "residual_scale": 25.0,
            "virtual_degree_threshold": 3,
            "virtual_k": 8,
        },
    },
    "b5_res25_proto02": {
        "dataset": "books",
        "desc": "residual_scale=25 + proto_alpha=0.02",
        "overrides": {"residual_scale": 25.0, "proto_alpha": 0.02},
    },
    "b6_combo": {
        "dataset": "books",
        "desc": "res25 + virt3 + proto0.02 + polar0.9",
        "overrides": {
            "residual_scale": 25.0,
            "virtual_degree_threshold": 3,
            "virtual_k": 8,
            "proto_alpha": 0.02,
            "polarity_consensus_score_weight": 0.9,
            "score_smoothing_alpha": 0.45,
        },
    },
    "b7_res20_virt3_smooth05": {
        "dataset": "books",
        "desc": "res20 + virt3 + stronger smoothing 0.5",
        "overrides": {
            "residual_scale": 20.0,
            "virtual_degree_threshold": 3,
            "virtual_k": 8,
            "score_smoothing_alpha": 0.5,
        },
    },
    # Wave-2: push books mean AUC past 0.62
    "b8_virt3_k6": {
        "dataset": "books",
        "desc": "Reproduce historical best: virt_t=3, virtual_k=6 (was 0.6365)",
        "overrides": {"virtual_degree_threshold": 3, "virtual_k": 6},
    },
    "b9_virt3_k5": {
        "dataset": "books",
        "desc": "virt_t=3, virtual_k=5",
        "overrides": {"virtual_degree_threshold": 3, "virtual_k": 5},
    },
    "b10_virt2_k6": {
        "dataset": "books",
        "desc": "virt_t=2, virtual_k=6",
        "overrides": {"virtual_degree_threshold": 2, "virtual_k": 6},
    },
    "b11_virt4_k6": {
        "dataset": "books",
        "desc": "virt_t=4, virtual_k=6",
        "overrides": {"virtual_degree_threshold": 4, "virtual_k": 6},
    },
    "b12_novirt": {
        "dataset": "books",
        "desc": "Disable virtual neighbors",
        "overrides": {"use_virtual_neighbors": False},
    },
    "b13_smooth050": {
        "dataset": "books",
        "desc": "score_smoothing_alpha=0.50",
        "overrides": {"score_smoothing_alpha": 0.50},
    },
    "b14_smooth040": {
        "dataset": "books",
        "desc": "score_smoothing_alpha=0.40",
        "overrides": {"score_smoothing_alpha": 0.40},
    },
    "b15_polar095": {
        "dataset": "books",
        "desc": "polarity score weight 0.95",
        "overrides": {"polarity_consensus_score_weight": 0.95},
    },
    "b16_polar085": {
        "dataset": "books",
        "desc": "polarity score weight 0.85",
        "overrides": {"polarity_consensus_score_weight": 0.85},
    },
    "b17_polar_th065": {
        "dataset": "books",
        "desc": "polarity threshold 0.65",
        "overrides": {"polarity_consensus_threshold": 0.65},
    },
    "b18_weight15": {
        "dataset": "books",
        "desc": "guidance weight 1.5",
        "overrides": {"weight": 1.5},
    },
    "b19_weight10": {
        "dataset": "books",
        "desc": "guidance weight 1.0",
        "overrides": {"weight": 1.0},
    },
    "b20_res12": {
        "dataset": "books",
        "desc": "residual_scale=12",
        "overrides": {"residual_scale": 12.0},
    },
    "b21_res18": {
        "dataset": "books",
        "desc": "residual_scale=18",
        "overrides": {"residual_scale": 18.0},
    },
    "b22_proto0005": {
        "dataset": "books",
        "desc": "proto_alpha=0.0005",
        "overrides": {"proto_alpha": 0.0005},
    },
    "b23_ae_lr02": {
        "dataset": "books",
        "desc": "ae_lr=0.02",
        "overrides": {"ae_lr": 0.02},
    },
    "b24_ae_alpha065": {
        "dataset": "books",
        "desc": "ae_alpha=0.65",
        "overrides": {"ae_alpha": 0.65},
    },
    "b25_numtrial3": {
        "dataset": "books",
        "desc": "num_trial=3 ensemble",
        "overrides": {"num_trial": 3},
    },
    "b26_virt3k6_smooth05": {
        "dataset": "books",
        "desc": "virt3_k6 + smooth 0.50",
        "overrides": {
            "virtual_degree_threshold": 3,
            "virtual_k": 6,
            "score_smoothing_alpha": 0.50,
        },
    },
    "b27_virt3k6_polar095": {
        "dataset": "books",
        "desc": "virt3_k6 + polarity weight 0.95",
        "overrides": {
            "virtual_degree_threshold": 3,
            "virtual_k": 6,
            "polarity_consensus_score_weight": 0.95,
        },
    },
    "b28_virt3k6_res12": {
        "dataset": "books",
        "desc": "virt3_k6 + residual_scale 12",
        "overrides": {
            "virtual_degree_threshold": 3,
            "virtual_k": 6,
            "residual_scale": 12.0,
        },
    },
    "b29_virt3k6_weight15": {
        "dataset": "books",
        "desc": "virt3_k6 + weight 1.5",
        "overrides": {
            "virtual_degree_threshold": 3,
            "virtual_k": 6,
            "weight": 1.5,
        },
    },
    "b30_virt3k6_combo": {
        "dataset": "books",
        "desc": "virt3_k6 + smooth0.5 + polar0.95 + res12",
        "overrides": {
            "virtual_degree_threshold": 3,
            "virtual_k": 6,
            "score_smoothing_alpha": 0.50,
            "polarity_consensus_score_weight": 0.95,
            "residual_scale": 12.0,
        },
    },
    "b31_novirt_smooth05_polar095": {
        "dataset": "books",
        "desc": "no virt + smooth0.5 + polar0.95",
        "overrides": {
            "use_virtual_neighbors": False,
            "score_smoothing_alpha": 0.50,
            "polarity_consensus_score_weight": 0.95,
        },
    },
    # ----- reddit -----
    "r0_baseline": {
        "dataset": "reddit",
        "desc": "Current reddit.yaml",
        "overrides": {},
    },
    "r1_smooth045": {
        "dataset": "reddit",
        "desc": "Add score_smoothing_alpha=0.45 (missing in current yaml)",
        "overrides": {"score_smoothing_alpha": 0.45},
    },
    "r2_res15_w05": {
        "dataset": "reddit",
        "desc": "residual=15, weight=0.5, smooth=0.45",
        "overrides": {
            "residual_scale": 15.0,
            "weight": 0.5,
            "score_smoothing_alpha": 0.45,
        },
    },
    "r3_virt6": {
        "dataset": "reddit",
        "desc": "virtual_t6_k8 + smooth045",
        "overrides": {
            "virtual_degree_threshold": 6,
            "virtual_k": 8,
            "score_smoothing_alpha": 0.45,
        },
    },
    "r4_polar07": {
        "dataset": "reddit",
        "desc": "polarity weight 0.7 + smooth045",
        "overrides": {
            "polarity_consensus_score_weight": 0.7,
            "score_smoothing_alpha": 0.45,
        },
    },
    "r5_combo": {
        "dataset": "reddit",
        "desc": "smooth045 + res15 + w0.5 + virt6 + polar07",
        "overrides": {
            "score_smoothing_alpha": 0.45,
            "residual_scale": 15.0,
            "weight": 0.5,
            "virtual_degree_threshold": 6,
            "virtual_k": 8,
            "polarity_consensus_score_weight": 0.7,
        },
    },
    "r6_polar_flip": {
        "dataset": "reddit",
        "desc": "easier flip (thresh 0.55) + polar07 + smooth045 + res15",
        "overrides": {
            "score_smoothing_alpha": 0.45,
            "residual_scale": 15.0,
            "weight": 0.5,
            "polarity_consensus_score_weight": 0.7,
            "polarity_consensus_threshold": 0.55,
        },
    },
    "r7_proto001_smooth": {
        "dataset": "reddit",
        "desc": "proto_alpha=0.001 + smooth045 + res15",
        "overrides": {
            "proto_alpha": 0.001,
            "score_smoothing_alpha": 0.45,
            "residual_scale": 15.0,
            "weight": 0.5,
        },
    },
}


def _gpu_status() -> List[Dict[str, int]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "mem_free": int(free), "util": int(util)})
    return rows


def _pick_gpus(preferred: Optional[List[int]], min_free: int = 5000) -> List[int]:
    status = {g["index"]: g for g in _gpu_status()}
    cands = []
    for idx, g in status.items():
        if preferred is not None and idx not in preferred:
            continue
        if g["mem_free"] >= min_free and g["util"] <= 20:
            cands.append((0, -g["mem_free"], idx))
        elif g["mem_free"] >= min_free + 3000 and g["util"] <= 50:
            cands.append((1, -g["mem_free"], idx))
        elif g["mem_free"] >= 18000:
            cands.append((2, -g["mem_free"], idx))
    cands.sort()
    return [idx for _, __, idx in cands]


def _write_cfg(trial_id: str, dataset: str, overrides: dict, cfg_root: Path) -> Path:
    with open(REPO / "configs" / f"{dataset}.yaml", "r", encoding="utf-8") as f:
        base = yaml.load(f, Loader=yaml.Loader)
    cfg = copy.deepcopy(base)
    cfg.update(overrides)
    cfg["exp_tag"] = f"tune_{trial_id}"
    out = cfg_root / trial_id / f"{dataset}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return out


def _run_one(job: Tuple) -> Dict[str, Any]:
    trial_id, dataset, seed, gpu, cfg_path, result_path, log_path, model_root, deterministic = job
    result_path = Path(result_path)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("auc_mean") is not None or payload.get("auc") is not None:
                return {
                    "trial_id": trial_id,
                    "dataset": dataset,
                    "seed": seed,
                    "gpu": gpu,
                    "returncode": 0,
                    "elapsed_sec": 0.0,
                    "auc": float(payload.get("auc_mean", payload.get("auc"))),
                    "ap": float(payload.get("ap_mean", payload.get("ap", 0.0) or 0.0)),
                    "skipped": True,
                    "result_file": str(result_path),
                }
        except Exception:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str(model_root)
    env["FMGAD_RUN_TAG_SUFFIX"] = f"{trial_id}_seed{seed}"
    cmd = [
        PY,
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
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    log_path.write_text((proc.stdout or "") + ("\n--- stderr ---\n" + (proc.stderr or "") if proc.stderr else ""), encoding="utf-8")
    row: Dict[str, Any] = {
        "trial_id": trial_id,
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "result_file": str(result_path),
        "log_file": str(log_path),
        "skipped": False,
    }
    if proc.returncode == 0 and result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        row["auc"] = float(payload.get("auc_mean", payload.get("auc")))
        row["ap"] = float(payload.get("ap_mean", payload.get("ap", 0.0) or 0.0))
        row["polarity_flipped"] = (payload.get("polarity_diagnostics") or {}).get("flipped")
    else:
        row["stderr_tail"] = (proc.stderr or "")[-1500:]
    return row


def _aggregate(result_root: Path, summary_dir: Path, trial_ids: List[str]) -> Dict[str, Any]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_trial: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    for tid in trial_ids:
        rows = []
        for p in sorted((result_root / tid).glob("*_seed*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            auc = float(data.get("auc_mean", data.get("auc", 0)))
            ap = float(data.get("ap_mean", data.get("ap", 0) or 0))
            rows.append({"trial_id": tid, "dataset": data["dataset"], "seed": data["seed"], "auc": auc, "ap": ap})
        if not rows:
            continue
        all_rows.extend(rows)
        ds = rows[0]["dataset"]
        mean_auc = sum(r["auc"] for r in rows) / len(rows)
        mean_ap = sum(r["ap"] for r in rows) / len(rows)
        per_trial[tid] = {
            "dataset": ds,
            "n_seeds": len(rows),
            "mean_auc": mean_auc,
            "mean_ap": mean_ap,
            "per_seed_auc": [r["auc"] for r in sorted(rows, key=lambda x: x["seed"])],
            "desc": TRIALS.get(tid, {}).get("desc", ""),
            "overrides": TRIALS.get(tid, {}).get("overrides", {}),
        }

    with open(summary_dir / "all_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "dataset", "seed", "auc", "ap"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    best_by_ds: Dict[str, Any] = {}
    lines = [
        "# Books / Reddit Tuning Summary",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for ds in DATASETS:
        items = [(tid, info) for tid, info in per_trial.items() if info["dataset"] == ds and info["n_seeds"] >= 5]
        items.sort(key=lambda kv: kv[1]["mean_auc"], reverse=True)
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| trial | mean_auc | mean_ap | n | desc |")
        lines.append("|-------|----------|---------|---|------|")
        for tid, info in items:
            lines.append(
                f"| {tid} | {info['mean_auc']:.4f} | {info['mean_ap']:.4f} | {info['n_seeds']} | {info['desc']} |"
            )
        lines.append("")
        if items:
            best_tid, best_info = items[0]
            base = per_trial.get(f"{'b' if ds=='books' else 'r'}0_baseline")
            delta = None
            if base and base.get("n_seeds", 0) >= 5:
                delta = best_info["mean_auc"] - base["mean_auc"]
            best_by_ds[ds] = {
                "trial_id": best_tid,
                "mean_auc": best_info["mean_auc"],
                "mean_ap": best_info["mean_ap"],
                "overrides": best_info["overrides"],
                "delta_vs_baseline": delta,
                "baseline_auc": base["mean_auc"] if base else None,
            }
            lines.append(
                f"- **best**: `{best_tid}` AUC={best_info['mean_auc']:.4f}"
                + (f" (Δ={delta:+.4f} vs baseline)" if delta is not None else "")
            )
            lines.append("")

    summary = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "trials": per_trial, "best_by_dataset": best_by_ds}
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _apply_winners(summary: Dict[str, Any], min_delta: float = 0.002) -> List[str]:
    applied = []
    for ds, info in (summary.get("best_by_dataset") or {}).items():
        delta = info.get("delta_vs_baseline")
        if delta is None or delta < min_delta:
            continue
        cfg_path = REPO / "configs" / f"{ds}.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        before = copy.deepcopy(cfg)
        cfg.update(info.get("overrides") or {})
        backup = cfg_path.with_suffix(".yaml.bak_before_tune")
        if not backup.exists():
            with open(backup, "w", encoding="utf-8") as f:
                yaml.dump(before, f, default_flow_style=False, allow_unicode=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        applied.append(f"{ds}:{info['trial_id']} Δauc={delta:+.4f}")
    return applied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=str, default=str(REPO / "results" / "tune_books_reddit"))
    ap.add_argument("--trials", type=str, default=None, help="Comma-separated trial ids")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--gpus", type=str, default="auto")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--min-free-mb", type=int, default=5000)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--apply-winners", action="store_true", help="Write best configs back if AUC improves")
    ap.add_argument("--min-delta", type=float, default=0.002)
    args = ap.parse_args()

    out_root = Path(args.output_root).expanduser().resolve()
    cfg_root = out_root / "configs"
    result_root = out_root / "results"
    model_root = out_root / "models"
    summary_dir = out_root / "summary"
    log_root = out_root / "logs"
    for p in (cfg_root, result_root, model_root, summary_dir, log_root):
        p.mkdir(parents=True, exist_ok=True)

    lock_fd = open(out_root / "tune.lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another tune_books_reddit holds the lock; exiting.", flush=True)
        return 0

    trial_ids = [t.strip() for t in (args.trials.split(",") if args.trials else TRIALS.keys()) if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.aggregate_only:
        summary = _aggregate(result_root, summary_dir, trial_ids)
        print(json.dumps(summary.get("best_by_dataset"), indent=2))
        if args.apply_winners:
            print("Applied:", _apply_winners(summary, args.min_delta))
        return 0

    if args.gpus == "auto":
        gpus = _pick_gpus(None, min_free=args.min_free_mb)
        if not gpus:
            # fallback: allow busy but roomy GPUs
            gpus = [g["index"] for g in sorted(_gpu_status(), key=lambda x: -x["mem_free"]) if g["mem_free"] >= args.min_free_mb][:2]
    else:
        gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        print("No suitable GPUs found", file=sys.stderr)
        return 1
    print(f"Using GPUs: {gpus}", flush=True)
    print("GPU snapshot:", _gpu_status(), flush=True)

    jobs = []
    for tid in trial_ids:
        if tid not in TRIALS:
            print("Unknown trial", tid, file=sys.stderr)
            return 1
        meta = TRIALS[tid]
        ds = meta["dataset"]
        cfg_path = _write_cfg(tid, ds, meta["overrides"], cfg_root)
        (result_root / tid).mkdir(parents=True, exist_ok=True)
        for i, seed in enumerate(seeds):
            gpu = gpus[(len(jobs)) % len(gpus)]
            jobs.append(
                (
                    tid,
                    ds,
                    seed,
                    gpu,
                    str(cfg_path),
                    str(result_root / tid / f"{ds}_seed{seed}.json"),
                    str(log_root / tid / f"{ds}_seed{seed}.log"),
                    str(model_root / tid),
                    bool(args.deterministic),
                )
            )

    print(f"Scheduling {len(jobs)} jobs", flush=True)
    rows: List[Dict[str, Any]] = []
    # Allow multiple workers per GPU (books is light; reddit still OK on 24GB).
    mw = min(args.max_workers, len(jobs))
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_run_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            tag = "SKIP" if row.get("skipped") else f"{float(row.get('elapsed_sec') or 0):.0f}s"
            print(
                f"[{k}/{len(jobs)}] {row['trial_id']} {row['dataset']} s{row['seed']} "
                f"gpu={row['gpu']} rc={row['returncode']} auc={row.get('auc')} {tag}",
                flush=True,
            )

    (summary_dir / "run_log.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _aggregate(result_root, summary_dir, trial_ids)
    print("\n=== Best by dataset ===", flush=True)
    for ds, info in (summary.get("best_by_dataset") or {}).items():
        print(
            f"  {ds}: {info['trial_id']} auc={info['mean_auc']:.4f} "
            f"baseline={info.get('baseline_auc')} delta={info.get('delta_vs_baseline')}",
            flush=True,
        )
    if args.apply_winners:
        applied = _apply_winners(summary, args.min_delta)
        print("Applied winners:", applied, flush=True)
    print("Summary:", summary_dir / "summary.md", flush=True)
    ok = all(r.get("returncode") == 0 for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
