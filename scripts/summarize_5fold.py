#!/usr/bin/env python3
"""Consolidate the 5-fold validation summaries for a trainer into one report.

Reads every ``fold_<F>/validation/summary.json`` under
``$nnUNet_results/<dataset>/<trainer>__<plans>__<config>/`` and prints:

- Per-case Dice table (all folds joined; one row per patient).
- Mean Dice per fold + overall.
- Cases with Dice == 0 (failures) and Dice < 0.5 (problematic).
- If ``--baseline`` is provided, compares per-case against that trainer
  and lists improvers (Δ >= +0.1) and regressors (Δ <= -0.1).

Usage:
    # One trainer / plans / config
    python scripts/summarize_5fold.py --dataset 501 \
        --trainer nnUNetTrainerALT_os033_250epochs \
        --plans nnUNetTSMRIPlans --config 3d_fullres

    # Compare against DA5 baseline
    python scripts/summarize_5fold.py --dataset 501 \
        --trainer nnUNetTrainerALT_os033_250epochs \
        --plans nnUNetTSMRIPlans --config 3d_fullres \
        --baseline nnUNetTrainerDA5_100epochs

    # All configs for a trainer (auto-discovers 2d + 3d_fullres if present)
    python scripts/summarize_5fold.py --dataset 501 \
        --trainer nnUNetTrainerALT_os033_250epochs --all-configs

No training is triggered. Pure read-only reporting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def env_path(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"env var {name} is not set")
    return Path(v)


def resolve_dataset_name(nnunet_results: Path, dataset_id: int) -> str:
    prefix = f"Dataset{dataset_id:03d}_"
    matches = sorted(
        p.name for p in nnunet_results.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    )
    if not matches:
        raise SystemExit(
            f"No dataset folder starting with '{prefix}' under {nnunet_results}"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Ambiguous: {len(matches)} folders match '{prefix}*': {matches}"
        )
    return matches[0]


def load_fold_summaries(trainer_dir: Path) -> dict[int, dict[str, float]]:
    """Return ``{fold: {case_id: dice}}`` for every fold with a summary.json."""
    out: dict[int, dict[str, float]] = {}
    for f in range(5):
        p = trainer_dir / f"fold_{f}" / "validation" / "summary.json"
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        per_case: dict[str, float] = {}
        for c in data.get("metric_per_case", []):
            case_id = os.path.basename(c["prediction_file"]).replace(".nii.gz", "")
            dice = c["metrics"]["1"]["Dice"]
            per_case[case_id] = float(dice)
        out[f] = per_case
    return out


def discover_trainer_dirs(
    dataset_dir: Path, trainer: str
) -> list[tuple[str, str, Path]]:
    """Return ``(plans, config, path)`` tuples for every trainer output dir."""
    results: list[tuple[str, str, Path]] = []
    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith(trainer + "__"):
            continue
        # format: <trainer>__<plans>__<config>
        rest = d.name[len(trainer) + 2 :]
        if "__" not in rest:
            continue
        plans, config = rest.rsplit("__", 1)
        results.append((plans, config, d))
    return results


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def render_report(
    label: str,
    trainer_dir: Path,
    per_fold: dict[int, dict[str, float]],
    baseline_per_case: dict[str, float] | None = None,
    baseline_label: str = "baseline",
) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f" {label}")
    lines.append(f" {trainer_dir}")
    lines.append("=" * 78)

    # Flatten into case -> (fold, dice)
    case_fold: dict[str, tuple[int, float]] = {}
    for f, cases in per_fold.items():
        for k, v in cases.items():
            case_fold[k] = (f, v)

    if not case_fold:
        lines.append("  (no fold validation summaries found yet)")
        return "\n".join(lines)

    # Per-fold mean
    lines.append("")
    lines.append("Per-fold mean Dice:")
    fold_means = {}
    for f in sorted(per_fold.keys()):
        vs = list(per_fold[f].values())
        fold_means[f] = _mean(vs)
        lines.append(f"  fold {f}: n={len(vs):>2}  mean={fold_means[f]:.4f}  "
                     f"min={min(vs):.3f}  max={max(vs):.3f}")
    all_vals = [v for (_, v) in case_fold.values()]
    overall = _mean(all_vals)
    lines.append(f"  5-fold agg:  n={len(all_vals):>2}  mean={overall:.4f}  "
                 f"median={sorted(all_vals)[len(all_vals)//2]:.3f}  "
                 f"min={min(all_vals):.3f}  max={max(all_vals):.3f}")

    # Per-case table
    lines.append("")
    header = f"{'case':8} {'fold':>4} {'Dice':>7}"
    if baseline_per_case is not None:
        header += f"  {baseline_label[:14]:>14} {'Δ':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    rows = sorted(case_fold.items(), key=lambda kv: kv[0])
    for case, (f, dice) in rows:
        row = f"{case:8} {f:>4} {dice:>7.3f}"
        if baseline_per_case is not None:
            b = baseline_per_case.get(case)
            if b is None:
                row += f"  {'-':>14} {'-':>7}"
            else:
                row += f"  {b:>14.3f} {dice-b:>+7.3f}"
        lines.append(row)

    # Failure / problematic counts
    n_zero = sum(1 for v in all_vals if v == 0.0)
    n_low = sum(1 for v in all_vals if 0.0 < v < 0.5)
    lines.append("")
    lines.append(f"Failures (Dice == 0.000): n={n_zero}")
    zeros = [k for k, (_, v) in case_fold.items() if v == 0.0]
    if zeros:
        lines.append(f"  cases: {sorted(zeros)}")
    lines.append(f"Problematic (0 < Dice < 0.5): n={n_low}")
    lows = sorted(
        [(k, v) for k, (_, v) in case_fold.items() if 0.0 < v < 0.5],
        key=lambda kv: kv[1],
    )
    if lows:
        lines.append(
            "  cases: " + ", ".join(f"{k}={v:.3f}" for k, v in lows)
        )

    # Regressors / improvers vs baseline
    if baseline_per_case is not None:
        deltas = []
        for case, (_, dice) in case_fold.items():
            b = baseline_per_case.get(case)
            if b is None:
                continue
            deltas.append((case, b, dice, dice - b))
        if deltas:
            shared = [d[3] for d in deltas]
            lines.append("")
            lines.append(
                f"Comparison vs {baseline_label} (shared n={len(deltas)}):"
            )
            lines.append(f"  mean {baseline_label}: {_mean(d[1] for d in deltas):.4f}")
            lines.append(f"  mean current:        {_mean(d[2] for d in deltas):.4f}")
            lines.append(f"  mean Δ:              {_mean(shared):+.4f}")
            improvers = sorted(
                [d for d in deltas if d[3] >= 0.1], key=lambda x: -x[3]
            )
            regressors = sorted(
                [d for d in deltas if d[3] <= -0.1], key=lambda x: x[3]
            )
            lines.append(f"  improvers (Δ ≥ +0.1): n={len(improvers)}")
            for case, b, d, dlt in improvers:
                lines.append(f"    {case:8} {b:.3f} -> {d:.3f}  ({dlt:+.3f})")
            lines.append(f"  regressors (Δ ≤ -0.1): n={len(regressors)}")
            for case, b, d, dlt in regressors:
                lines.append(f"    {case:8} {b:.3f} -> {d:.3f}  ({dlt:+.3f})")

    return "\n".join(lines)


def flatten(per_fold: dict[int, dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cases in per_fold.values():
        out.update(cases)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", type=int, help="Dataset id (e.g. 501).")
    g.add_argument("--dataset-name", type=str,
                   help="Full dataset folder name (e.g. Dataset501_ALT_T1).")
    parser.add_argument("--trainer", required=True,
                        help="Trainer class name, e.g. "
                             "nnUNetTrainerALT_os033_250epochs.")
    parser.add_argument("--plans",
                        help="Plans identifier (e.g. nnUNetTSMRIPlans). "
                             "Ignored when --all-configs is set.")
    parser.add_argument("--config", choices=["2d", "3d_fullres", "3d_lowres",
                                              "3d_cascade_fullres"],
                        help="nnU-Net config. Ignored when --all-configs is set.")
    parser.add_argument("--all-configs", action="store_true",
                        help="Auto-discover every (plans, config) combo "
                             "already trained for this trainer.")
    parser.add_argument("--baseline",
                        help="Trainer to compare against, same plans/config.")
    parser.add_argument("--output", type=Path,
                        help="If set, also writes the report to this path.")
    args = parser.parse_args()

    nnunet_results = env_path("nnUNet_results")
    if args.dataset_name:
        ds_name = args.dataset_name
    else:
        ds_name = resolve_dataset_name(nnunet_results, args.dataset)
    dataset_dir = nnunet_results / ds_name

    if args.all_configs:
        combos = discover_trainer_dirs(dataset_dir, args.trainer)
        if not combos:
            raise SystemExit(
                f"No trainer folders found for {args.trainer} under {dataset_dir}"
            )
    else:
        if not args.plans or not args.config:
            raise SystemExit("Provide --plans and --config, or use --all-configs.")
        trainer_dir = dataset_dir / f"{args.trainer}__{args.plans}__{args.config}"
        combos = [(args.plans, args.config, trainer_dir)]

    reports: list[str] = []
    for plans, config, trainer_dir in combos:
        if not trainer_dir.is_dir():
            print(f"[skip] {trainer_dir} does not exist", file=sys.stderr)
            continue

        per_fold = load_fold_summaries(trainer_dir)

        baseline_per_case: dict[str, float] | None = None
        baseline_label = "baseline"
        if args.baseline:
            baseline_dir = dataset_dir / f"{args.baseline}__{plans}__{config}"
            if baseline_dir.is_dir():
                baseline_per_case = flatten(load_fold_summaries(baseline_dir))
                baseline_label = args.baseline
            else:
                print(f"[warn] baseline dir not found: {baseline_dir}",
                      file=sys.stderr)

        label = f"{args.trainer}  |  plans={plans}  |  config={config}"
        report = render_report(
            label, trainer_dir, per_fold,
            baseline_per_case=baseline_per_case,
            baseline_label=baseline_label,
        )
        reports.append(report)

    full = "\n\n".join(reports)
    print(full)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(full + "\n")
        print(f"\n[wrote] {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
