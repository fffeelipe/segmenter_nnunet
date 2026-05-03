#!/usr/bin/env python3
"""
Rebuild reports/per_case_baseline_and_mixed.csv from nnU-Net summary.json artifacts.

This script is intentionally path-opinionated to match the repo layout used in ANALYSIS.md.
You can override any input via CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dice_from_metrics(metrics: Mapping[str, Any]) -> float:
    # nnU-Net uses label keys either as strings or ints depending on version.
    if "1" in metrics:
        return float(metrics["1"]["Dice"])
    if 1 in metrics:
        return float(metrics[1]["Dice"])
    raise KeyError("Could not find label 1 in metrics")


def _tp_fp_fn_from_metrics(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    if "1" in metrics:
        m = metrics["1"]
    elif 1 in metrics:
        m = metrics[1]
    else:
        raise KeyError("Could not find label 1 in metrics")
    return float(m["TP"]), float(m["FP"]), float(m["FN"])


def _case_from_pred_path(pred_path: str, *, strip_t_suffix: bool) -> str:
    base = os.path.basename(pred_path)
    if base.endswith(".nii.gz"):
        base = base[: -len(".nii.gz")]
    if strip_t_suffix:
        base = re.sub(r"(_T1|_T2)$", "", base)
    return base


def _global_dice_from_items(
    items: List[Mapping[str, Any]],
    *,
    aggregate_t_suffix: bool,
) -> float:
    """
    Compute voxel-global Dice by aggregating TP/FP/FN:
      Dice = 2*TP / (2*TP + FP + FN)

    If `aggregate_t_suffix=True`, sums TP/FP/FN within each patient first (stripping `_T1/_T2`)
    and then sums across patients. (This avoids double-counting a patient as separate rows.)
    """
    if not aggregate_t_suffix:
        tp = fp = fn = 0.0
        for it in items:
            tpi, fpi, fni = _tp_fp_fn_from_metrics(it["metrics"])
            tp += tpi
            fp += fpi
            fn += fni
        denom = 2 * tp + fp + fn
        return 0.0 if denom == 0 else (2 * tp / denom)

    by_patient: Dict[str, List[tuple[float, float, float]]] = defaultdict(list)
    for it in items:
        patient = _case_from_pred_path(it["prediction_file"], strip_t_suffix=True)
        by_patient[patient].append(_tp_fp_fn_from_metrics(it["metrics"]))

    tp = fp = fn = 0.0
    for triples in by_patient.values():
        tps = sum(t for t, _, _ in triples)
        fps = sum(f for _, f, _ in triples)
        fns = sum(n for _, _, n in triples)
        tp += tps
        fp += fps
        fn += fns

    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp / denom)


def _read_nnunet_summary_metric_per_case(
    path: Path,
    *,
    strip_t_suffix: bool = False,
    aggregate_t_suffix: bool = False,
) -> Dict[str, float]:
    """
    Reads nnU-Net `summary.json` with `metric_per_case`.

    If `aggregate_t_suffix=True`, we first map entries to patient-id by stripping `_T1/_T2`
    and average the Dice within each patient, then return patient -> mean(Dice).
    """
    s = _load_json(path)
    items = s.get("metric_per_case", [])

    if not aggregate_t_suffix:
        out: Dict[str, float] = {}
        for it in items:
            case = _case_from_pred_path(it["prediction_file"], strip_t_suffix=strip_t_suffix)
            out[case] = _dice_from_metrics(it["metrics"])
        return out

    by_patient: Dict[str, List[float]] = defaultdict(list)
    for it in items:
        patient = _case_from_pred_path(it["prediction_file"], strip_t_suffix=True)
        by_patient[patient].append(_dice_from_metrics(it["metrics"]))
    return {k: statistics.mean(v) for k, v in by_patient.items()}


def _read_gated_ensemble_summary(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Reads `scripts/ensemble_gated.py` summary.json format.
    Returns: case -> {dice_2d, dice_3d, dice_ens, dice_gated}.
    """
    s = _load_json(path)
    per_case = s.get("per_case", [])
    out: Dict[str, Dict[str, float]] = {}
    for it in per_case:
        case = str(it["case"])
        out[case] = {
            "dice_2d": float(it["dice_2d"]),
            "dice_3d": float(it["dice_3d"]),
            "dice_ens": float(it["dice_ens"]),
            "dice_gated": float(it["dice_gated"]),
        }
    return out


def _merge_fold_validation_summaries(summary_paths: Iterable[Path]) -> Dict[str, float]:
    """
    DA5 baseline results are stored as fold-specific validation summaries.
    We merge those (case is unique across folds in CV).
    """
    out: Dict[str, float] = {}
    for p in summary_paths:
        d = _read_nnunet_summary_metric_per_case(p)
        overlap = set(out) & set(d)
        if overlap:
            raise RuntimeError(f"Duplicate cases across folds in {p}: {sorted(overlap)[:10]}")
        out.update(d)
    return out


def _global_dice_from_fold_summaries(summary_paths: Iterable[Path]) -> float:
    tp = fp = fn = 0.0
    for p in summary_paths:
        s = _load_json(p)
        items = s.get("metric_per_case", [])
        for it in items:
            tpi, fpi, fni = _tp_fp_fn_from_metrics(it["metrics"])
            tp += tpi
            fp += fpi
            fn += fni
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp / denom)


@dataclass(frozen=True)
class Inputs:
    da5_2d_fold_summaries: List[Path]
    da5_3d_fold_summaries: List[Path]
    gated_v2c_summary: Path
    gated_v2b_summary: Path
    intersection_503_postprocessed_summary: Path
    union_v2_503_postprocessed_fold0_summary: Path
    union_v2_503_postprocessed_folds1_4_summary: Path


def _default_inputs(repo_root: Path) -> Inputs:
    return Inputs(
        da5_2d_fold_summaries=[
            repo_root
            / "nnunet_env_base/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerDA5_100epochs__nnUNetPlans__2d"
            / f"fold_{i}/validation/summary.json"
            for i in range(5)
        ],
        da5_3d_fold_summaries=[
            repo_root
            / "nnunet_env_base/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerDA5_100epochs__nnUNetTSMRIPlans__3d_fullres"
            / f"fold_{i}/validation/summary.json"
            for i in range(5)
        ],
        gated_v2c_summary=repo_root
        / "nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_v2c_sigmoid/summary.json",
        gated_v2b_summary=repo_root
        / "nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_v2b_hard/summary.json",
        intersection_503_postprocessed_summary=repo_root
        / "nnunet_env_intersection/nnUNet_results/Dataset503_ALT_T1T2/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres"
        / "crossval_results_folds_0_1_2_3_4/postprocessed/summary.json",
        union_v2_503_postprocessed_fold0_summary=repo_root
        / "nnUNet_results_union_V2/Dataset503_ALT_T1T2/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres"
        / "crossval_results_folds_0/postprocessed/summary.json",
        union_v2_503_postprocessed_folds1_4_summary=repo_root
        / "nnUNet_results_union_V2/Dataset503_ALT_T1T2/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres"
        / "crossval_results_folds_1_2_3_4/postprocessed/summary.json",
    )


def _write_csv(
    out_path: Path,
    *,
    cases: List[str],
    columns: List[str],
    values: Dict[str, Dict[str, Optional[float]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", *columns])
        for case in cases:
            row = [case]
            for col in columns:
                v = values[col].get(case)
                row.append("" if v is None else f"{v:.15g}")
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild reports/per_case_baseline_and_mixed.csv from summary.json results."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of reports/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <repo-root>/reports/per_case_baseline_and_mixed.csv).",
    )
    args = parser.parse_args()

    repo_root: Path = args.repo_root.resolve()
    out_path: Path = (
        args.out.resolve()
        if args.out is not None
        else repo_root / "reports/per_case_baseline_and_mixed.csv"
    )

    inp = _default_inputs(repo_root)

    # Load columns.
    da5_2d = _merge_fold_validation_summaries(inp.da5_2d_fold_summaries)
    da5_3d = _merge_fold_validation_summaries(inp.da5_3d_fold_summaries)
    da5_2d_global = _global_dice_from_fold_summaries(inp.da5_2d_fold_summaries)
    da5_3d_global = _global_dice_from_fold_summaries(inp.da5_3d_fold_summaries)

    v2c = _read_gated_ensemble_summary(inp.gated_v2c_summary)
    v2b = _read_gated_ensemble_summary(inp.gated_v2b_summary)

    softavg_2d3d_tta = {k: v["dice_ens"] for k, v in v2c.items()}
    gated_v2b = {k: v["dice_gated"] for k, v in v2b.items()}
    gated_v2c = {k: v["dice_gated"] for k, v in v2c.items()}

    intersection_503_summary = _load_json(inp.intersection_503_postprocessed_summary)
    intersection_503_items = intersection_503_summary.get("metric_per_case", [])
    intersection_503 = _read_nnunet_summary_metric_per_case(
        inp.intersection_503_postprocessed_summary, strip_t_suffix=False, aggregate_t_suffix=False
    )
    intersection_503_global = _global_dice_from_items(intersection_503_items, aggregate_t_suffix=False)

    # union_v2 comes as two summaries; we want patient-level Dice (average T1/T2 when present)
    union0_summary = _load_json(inp.union_v2_503_postprocessed_fold0_summary)
    union14_summary = _load_json(inp.union_v2_503_postprocessed_folds1_4_summary)
    union0 = _read_nnunet_summary_metric_per_case(inp.union_v2_503_postprocessed_fold0_summary, aggregate_t_suffix=True)
    union14 = _read_nnunet_summary_metric_per_case(inp.union_v2_503_postprocessed_folds1_4_summary, aggregate_t_suffix=True)
    union_v2_503 = {**union0, **union14}
    union_v2_503_items = union0_summary.get("metric_per_case", []) + union14_summary.get("metric_per_case", [])
    union_v2_503_global = _global_dice_from_items(union_v2_503_items, aggregate_t_suffix=True)

    columns = [
        "da5_2d",
        "da5_3d",
        "softavg_2d3d_tta",
        "gated_v2b",
        "gated_v2c",
        "fusion_intersection_503_3d",
        "fusion_union_v2_503_3d",
    ]

    # Union-of-cases across columns, but keep deterministic order (IOG<number> ascending).
    all_cases = set()
    for d in [
        da5_2d,
        da5_3d,
        softavg_2d3d_tta,
        gated_v2b,
        gated_v2c,
        intersection_503,
        union_v2_503,
    ]:
        all_cases |= set(d.keys())

    def sort_key(case: str):
        m = re.match(r"IOG(\\d+)$", case)
        return (0, int(m.group(1))) if m else (1, case)

    cases = sorted(all_cases, key=sort_key)

    values: Dict[str, Dict[str, Optional[float]]] = {
        "da5_2d": {k: da5_2d.get(k) for k in cases},
        "da5_3d": {k: da5_3d.get(k) for k in cases},
        "softavg_2d3d_tta": {k: softavg_2d3d_tta.get(k) for k in cases},
        "gated_v2b": {k: gated_v2b.get(k) for k in cases},
        "gated_v2c": {k: gated_v2c.get(k) for k in cases},
        "fusion_intersection_503_3d": {k: intersection_503.get(k) for k in cases},
        "fusion_union_v2_503_3d": {k: union_v2_503.get(k) for k in cases},
    }

    # Add summary rows at the end:
    # - __GLOBAL_DICE__: voxel-global Dice (aggregated TP/FP/FN when available)
    # - __MEAN_DICE__: mean across cases (simple average of per-case Dice)
    def _mean_of_column(col: str) -> Optional[float]:
        vs = [v for v in values[col].values() if v is not None]
        return None if not vs else float(statistics.mean(vs))

    mean_row = {col: _mean_of_column(col) for col in columns}
    global_row: Dict[str, Optional[float]] = {
        "da5_2d": da5_2d_global,
        "da5_3d": da5_3d_global,
        # For gated/soft-avg summaries we don't have TP/FP/FN, so we report mean-of-cases.
        "softavg_2d3d_tta": mean_row["softavg_2d3d_tta"],
        "gated_v2b": mean_row["gated_v2b"],
        "gated_v2c": mean_row["gated_v2c"],
        "fusion_intersection_503_3d": intersection_503_global,
        "fusion_union_v2_503_3d": union_v2_503_global,
    }

    cases.extend(["__GLOBAL_DICE__", "__MEAN_DICE__"])
    for col in columns:
        values[col]["__GLOBAL_DICE__"] = global_row.get(col)
        values[col]["__MEAN_DICE__"] = mean_row.get(col)

    # Existence checks with helpful errors.
    missing_inputs = []
    for p in (
        inp.da5_2d_fold_summaries
        + inp.da5_3d_fold_summaries
        + [
            inp.gated_v2c_summary,
            inp.gated_v2b_summary,
            inp.intersection_503_postprocessed_summary,
            inp.union_v2_503_postprocessed_fold0_summary,
            inp.union_v2_503_postprocessed_folds1_4_summary,
        ]
    ):
        if not p.exists():
            missing_inputs.append(str(p))
    if missing_inputs:
        raise FileNotFoundError(
            "Missing required input files:\n- " + "\n- ".join(missing_inputs)
        )

    _write_csv(out_path, cases=cases, columns=columns, values=values)
    print(f"Wrote {out_path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()

