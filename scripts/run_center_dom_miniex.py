#!/usr/bin/env python3
"""Minimal center-dom dual-flow experiment (books / disney only).

Four weight modes for the dominant flow:
  uniform  — q_i = 1
  learned  — center-consistent q_i = sg[σ((R(s)_i - κ)/τ)]
  shuffled — permute learned q
  reversed — 1 - q

Target: learned > uniform, shuffled, reversed.
Also reports N_eff = (Σq)^2 / Σq^2.
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
MODES = ("uniform", "learned", "shuffled", "reversed")
GUIDANCE_WEIGHT = 1.25
# Aim for ~0.3N–0.8N effective sample size
KAPPA_Q = 0.5
TAU_Q = 0.2


def _run_one(mode: str, dataset: str, seed: int, gpu: int, out_root: Path) -> Dict[str, Any]:
    result_path = out_root / "results" / mode / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / mode / f"{dataset}_seed{seed}.log"
    cfg_path = out_root / "configs" / mode / f"{dataset}.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, Any] = {"mode": mode, "dataset": dataset, "seed": seed, "gpu": gpu}
    if result_path.exists():
        try:
            payload = json.load(open(result_path))
            row.update(
                {
                    "returncode": 0,
                    "auc": float(payload.get("auc_mean", payload.get("auc", float("nan")))),
                    "ap": float(payload.get("ap_mean", payload.get("ap", float("nan")))),
                    "n_eff": float(payload.get("n_eff", float("nan"))),
                    "n_eff_ratio": float(payload.get("n_eff_ratio", float("nan"))),
                    "elapsed_sec": 0.0,
                    "cached": True,
                }
            )
            print(
                f"[{mode}] {dataset} s{seed} cached auc={row['auc']:.4f} "
                f"N_eff_ratio={row['n_eff_ratio']:.3f}",
                flush=True,
            )
            return row
        except Exception:
            pass

    cfg = yaml.safe_load(open(REPO / "configs" / f"{dataset}.yaml")) or {}
    cfg.update(
        {
            "dataset": dataset,
            "dual_flow_mode": "center_dom",
            "dom_weight_mode": mode,
            "dom_kappa_q": KAPPA_Q,
            "dom_tau_q": TAU_Q,
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": GUIDANCE_WEIGHT,
            "sample_steps": 1,
            "num_trial": 1,
            "exp_tag": f"center_dom_{mode}_{dataset}",
        }
    )
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"{mode}_s{seed}"
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
        row["n_eff"] = float(payload.get("n_eff", float("nan")))
        row["n_eff_ratio"] = float(payload.get("n_eff_ratio", float("nan")))
    else:
        row["auc"] = float("nan")
        row["ap"] = float("nan")
        row["n_eff"] = float("nan")
        row["n_eff_ratio"] = float("nan")
    print(
        f"[{mode}] {dataset} s{seed} gpu={gpu} rc={rc} "
        f"auc={row['auc']:.4f} N_eff_ratio={row['n_eff_ratio']:.3f} {elapsed:.0f}s",
        flush=True,
    )
    return row


def _aggregate(out_root: Path) -> None:
    lines = [
        "# Center-dom minimal experiment (books / disney)",
        "",
        f"Guidance `w={GUIDANCE_WEIGHT}`, `κ_q={KAPPA_Q}`, `τ_q={TAU_Q}`. "
        f"Seeds `{list(SEEDS)}`.",
        "",
        "Inference: `v = v_free + w(v_free - v_dom)`.",
        "",
        "| Mode | Books AUROC | Disney AUROC | Books N_eff/N | Disney N_eff/N |",
        "|---|---:|---:|---:|---:|",
    ]
    summary: Dict[str, Any] = {}
    for mode in MODES:
        cells = []
        neff_cells = []
        per = {}
        for ds in DATASETS:
            aucs, ratios = [], []
            for seed in SEEDS:
                p = out_root / "results" / mode / f"{ds}_seed{seed}.json"
                if not p.exists():
                    continue
                d = json.load(open(p))
                aucs.append(float(d.get("auc_mean", d.get("auc"))))
                ratios.append(float(d.get("n_eff_ratio", float("nan"))))
            if len(aucs) == len(SEEDS):
                m, s = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
                cells.append(f"{m:.4f}±{s:.4f}")
                rr = [x for x in ratios if np.isfinite(x)]
                neff_cells.append(f"{float(np.mean(rr)):.3f}" if rr else "nan")
                per[ds] = {"auc_mean": m, "auc_std": s, "n_eff_ratio_mean": float(np.mean(rr)) if rr else None}
            else:
                cells.append(f"nan({len(aucs)}/5)")
                neff_cells.append("nan")
                per[ds] = {"n": len(aucs)}
        lines.append(
            f"| {mode} | {cells[0]} | {cells[1]} | {neff_cells[0]} | {neff_cells[1]} |"
        )
        summary[mode] = per

    # Pass/fail checklist vs hypothesis
    lines += ["", "## Hypothesis check (learned > others)", ""]
    for ds in DATASETS:
        learned = summary.get("learned", {}).get(ds, {}).get("auc_mean")
        if learned is None:
            lines.append(f"- **{ds}**: incomplete")
            continue
        bits = []
        for other in ("uniform", "shuffled", "reversed"):
            o = summary.get(other, {}).get(ds, {}).get("auc_mean")
            if o is None:
                bits.append(f"vs {other}=incomplete")
            else:
                ok = learned > o
                bits.append(f"vs {other}: {'PASS' if ok else 'FAIL'} (Δ={learned-o:+.4f})")
        lines.append(f"- **{ds}**: " + "; ".join(bits))

    lines += [
        "",
        "## N_eff diagnostic",
        "",
        "- Target band: `0.3 ≲ N_eff/N ≲ 0.8`",
        "- `≈1`: weights too flat (both flows similar)",
        "- `≪0.3`: too sharp (overfit risk on small graphs)",
        "",
    ]

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(summary_dir / "summary.json", "w"), indent=2)
    print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=str, default="0,2")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--output-dir", type=str, default="results/center_dom_miniex")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--modes", type=str, default=",".join(MODES))
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = (REPO / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    if args.aggregate_only:
        _aggregate(out_root)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs: List[Tuple[str, str, int]] = []
    for mode in modes:
        for ds in datasets:
            for seed in seeds:
                jobs.append((mode, ds, seed))
    print(f"Jobs={len(jobs)} on GPUs {gpus} (modes={modes}, datasets={datasets})", flush=True)

    q: Queue = Queue()
    for i in range(args.max_workers):
        q.put(gpus[i % len(gpus)])

    def _wrap(job):
        mode, ds, seed = job
        gpu = q.get()
        try:
            return _run_one(mode, ds, seed, gpu, out_root)
        finally:
            q.put(gpu)

    fails = 0
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as ex:
        futs = [ex.submit(_wrap, j) for j in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            if row.get("returncode", 1) != 0:
                fails += 1

    _aggregate(out_root)
    print(f"Done. fails={fails}. Summary: {out_root / 'summary' / 'summary.md'}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
