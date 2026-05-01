#!/usr/bin/env python3
"""
Compute Dataset503_ALT_T1T2 dataset statistics from already-extracted artifacts.

This repo snapshot does not include the raw NIfTI volumes in nnUNet_raw, so this
script intentionally relies on:
- nnUNet_preprocessed/Dataset503_ALT_T1T2/dataset_fingerprint.json
- nnUNet_preprocessed/Dataset503_ALT_T1T2/nnUNetPlans.json
- nnUNet_preprocessed/Dataset503_ALT_T1T2/splits_final.json
- reports/per_case_baseline_and_mixed.csv (patient-level list)
- nnUNet_results_union_V2/.../postprocessed/summary.json (for n_ref distribution)

Output:
- reports/dataset503_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FINGERPRINT = Path("nnunet_env/nnUNet_preprocessed/Dataset503_ALT_T1T2/dataset_fingerprint.json")
DEFAULT_PLANS = Path("nnunet_env/nnUNet_preprocessed/Dataset503_ALT_T1T2/nnUNetPlans.json")
DEFAULT_SPLITS = Path("nnunet_env/nnUNet_preprocessed/Dataset503_ALT_T1T2/splits_final.json")
DEFAULT_PATIENT_CSV = Path("reports/per_case_baseline_and_mixed.csv")

DEFAULT_SUMMARY_FOLD0 = Path(
    "nnUNet_results_union_V2/Dataset503_ALT_T1T2/"
    "nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres/"
    "crossval_results_folds_0/postprocessed/summary.json"
)
DEFAULT_SUMMARY_FOLDS1_4 = Path(
    "nnUNet_results_union_V2/Dataset503_ALT_T1T2/"
    "nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres/"
    "crossval_results_folds_1_2_3_4/postprocessed/summary.json"
)

DEFAULT_OUT = Path("reports/dataset503_stats.json")


def _as_base_case(case_id: str) -> str:
    return re.sub(r"_(T1|T2)$", "", case_id)


def _safe_mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _safe_stdev(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def _percentile(xs_sorted: list[float], q: float) -> float | None:
    """
    Linear interpolation percentile. q in [0, 100].
    """
    if not xs_sorted:
        return None
    if q <= 0:
        return xs_sorted[0]
    if q >= 100:
        return xs_sorted[-1]
    pos = (len(xs_sorted) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs_sorted[lo]
    w = pos - lo
    return xs_sorted[lo] * (1.0 - w) + xs_sorted[hi] * w


def _summarize(xs: Iterable[float]) -> dict[str, Any]:
    vals = [float(x) for x in xs if x is not None]
    vals_sorted = sorted(vals)
    return {
        "n": len(vals_sorted),
        "mean": _safe_mean(vals_sorted),
        "std": _safe_stdev(vals_sorted),
        "min": vals_sorted[0] if vals_sorted else None,
        "p05": _percentile(vals_sorted, 5),
        "p25": _percentile(vals_sorted, 25),
        "median": _percentile(vals_sorted, 50),
        "p75": _percentile(vals_sorted, 75),
        "p95": _percentile(vals_sorted, 95),
        "max": vals_sorted[-1] if vals_sorted else None,
    }


def _read_patient_list_from_csv(path: Path) -> list[str]:
    cases: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            c = (row.get("case") or "").strip()
            if not c or c.startswith("__"):
                continue
            cases.append(c)
    return cases


def _read_split_ids(path: Path) -> set[str]:
    splits = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for fold in splits:
        ids.update(fold.get("train", []))
        ids.update(fold.get("val", []))
    return ids


def _read_n_ref_from_summary(path: Path) -> dict[str, int]:
    """
    Returns dict case_id -> n_ref (foreground voxels).
    """
    if not path.is_file():
        return {}
    j = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for entry in j.get("metric_per_case", []):
        metrics = entry.get("metrics", {}).get("1", {})
        n_ref = metrics.get("n_ref")
        ref_file = entry.get("reference_file") or ""
        # reference_file ends with .../<case>.nii.gz
        case = Path(ref_file).name.replace(".nii.gz", "")
        if isinstance(n_ref, (int, float)) and case:
            out[case] = int(n_ref)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fingerprint", type=Path, default=DEFAULT_FINGERPRINT)
    ap.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--patient-csv", type=Path, default=DEFAULT_PATIENT_CSV)
    ap.add_argument("--summary-fold0", type=Path, default=DEFAULT_SUMMARY_FOLD0)
    ap.add_argument("--summary-folds1-4", type=Path, default=DEFAULT_SUMMARY_FOLDS1_4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    plans = json.loads(args.plans.read_text(encoding="utf-8"))

    # --- counts / discrepancy (direction mismatch split) ---
    patient_cases = _read_patient_list_from_csv(args.patient_csv)
    patient_n = len(patient_cases)

    split_ids = _read_split_ids(args.splits)
    derived_n = len(split_ids)

    split_patient_cases = sorted({_as_base_case(x) for x in split_ids if x.endswith("_T1") or x.endswith("_T2")})
    split_patient_n = len(split_patient_cases)
    aligned_patient_n = patient_n - split_patient_n

    # --- spacing and slices/shapes (from fingerprint) ---
    spacings = fingerprint.get("spacings", [])
    z_sp = [s[0] for s in spacings if isinstance(s, list) and len(s) == 3]
    y_sp = [s[1] for s in spacings if isinstance(s, list) and len(s) == 3]
    x_sp = [s[2] for s in spacings if isinstance(s, list) and len(s) == 3]

    shapes = fingerprint.get("shapes_after_crop", [])
    z_slices = [sh[0] for sh in shapes if isinstance(sh, list) and len(sh) == 3]
    y_size = [sh[1] for sh in shapes if isinstance(sh, list) and len(sh) == 3]
    x_size = [sh[2] for sh in shapes if isinstance(sh, list) and len(sh) == 3]

    # --- tumor size proxy (foreground voxel counts from CV summaries) ---
    nref0 = _read_n_ref_from_summary(args.summary_fold0)
    nref1 = _read_n_ref_from_summary(args.summary_folds1_4)
    nref = {**nref0, **nref1}  # unique keys per case

    nref_by_base: dict[str, list[int]] = {}
    for case, n in nref.items():
        nref_by_base.setdefault(_as_base_case(case), []).append(int(n))

    # patient-level tumor size proxy: average across derived T1/T2 if present
    nref_patient_mean = {base: int(round(statistics.fmean(v))) for base, v in nref_by_base.items()}

    # sample-level tumor size proxy: include derived ids as-is
    nref_sample = list(nref.values())
    nref_patient = [nref_patient_mean[b] for b in sorted(nref_patient_mean)]

    # approximate mm^3 using median voxel volume (median spacing product)
    med_sp = plans.get("original_median_spacing_after_transp")
    voxel_volume_mm3 = None
    if isinstance(med_sp, list) and len(med_sp) == 3:
        voxel_volume_mm3 = float(med_sp[0]) * float(med_sp[1]) * float(med_sp[2])
    nref_patient_mm3 = [n * voxel_volume_mm3 for n in nref_patient] if voxel_volume_mm3 else []

    out = {
        "dataset": "Dataset503_ALT_T1T2",
        "counts": {
            "n_patients_patient_level_eval": patient_n,
            "n_samples_numTraining": derived_n,
            "n_direction_mismatch_patients_split": split_patient_n,
            "direction_mismatch_patients": split_patient_cases,
            "n_aligned_patients_single_sample": aligned_patient_n,
        },
        "spacing_mm": {
            "z": _summarize(z_sp),
            "y": _summarize(y_sp),
            "x": _summarize(x_sp),
            "in_plane_mean_xy": _summarize([(a + b) / 2.0 for a, b in zip(x_sp, y_sp)]),
        },
        "shape_after_crop_voxels": {
            "z_slices": _summarize(z_slices),
            "y": _summarize(y_size),
            "x": _summarize(x_size),
        },
        "plans_medians": {
            "original_median_spacing_after_transp": plans.get("original_median_spacing_after_transp"),
            "original_median_shape_after_transp": plans.get("original_median_shape_after_transp"),
            "target_spacing_3d_fullres": plans.get("configurations", {}).get("3d_fullres", {}).get("spacing"),
            "target_spacing_2d": plans.get("configurations", {}).get("2d", {}).get("spacing"),
            "patch_size_3d_fullres": plans.get("configurations", {}).get("3d_fullres", {}).get("patch_size"),
            "patch_size_2d": plans.get("configurations", {}).get("2d", {}).get("patch_size"),
            "batch_size_3d_fullres": plans.get("configurations", {}).get("3d_fullres", {}).get("batch_size"),
            "batch_size_2d": plans.get("configurations", {}).get("2d", {}).get("batch_size"),
        },
        "tumor_size_proxy": {
            "n_ref_voxels_sample_level": _summarize(nref_sample),
            "n_ref_voxels_patient_level_avg_T1T2": _summarize(nref_patient),
            "approx_voxel_volume_mm3_from_original_median_spacing": voxel_volume_mm3,
            "approx_tumor_volume_mm3_patient_level": _summarize(nref_patient_mm3) if nref_patient_mm3 else None,
            "notes": (
                "n_ref extracted from nnUNet postprocessed CV summaries (union_v2 3D). "
                "Patient-level values average derived <case>_T1/<case>_T2 if present. "
                "mm^3 volumes are approximated using the median voxel volume from nnUNet plans."
            ),
        },
        "sources": {
            "fingerprint": str(args.fingerprint),
            "plans": str(args.plans),
            "splits": str(args.splits),
            "patient_csv": str(args.patient_csv),
            "summary_fold0": str(args.summary_fold0),
            "summary_folds1_4": str(args.summary_folds1_4),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

