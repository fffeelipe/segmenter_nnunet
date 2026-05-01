#!/usr/bin/env python3
"""
Extract dataset statistics for Dataset503_ALT_T1T2 (T1+T2, fused ALT mask).

What this computes (per *training sample*, i.e. includes derived *_T1/_T2):
- slides per case: number of axial slices (n_slices)
- tumor extent: number of slices containing any tumor (n_tumor_slices)
- tumor size: foreground voxels (tumor_voxels) and physical volume (tumor_mm3)
- per-channel intensity mean/std (either within tumor mask, or whole image)

Outputs:
- JSON report (default: reports/dataset503_stats.json)
- Optional CSV (default: reports/dataset503_stats.csv when --csv is provided)

Dataset discovery:
- By default uses $nnUNet_raw/Dataset503_ALT_T1T2/{imagesTr,labelsTr}
- If $nnUNet_raw is not set, you must pass --raw-root (points to nnUNet_raw)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np


DATASET = "Dataset503_ALT_T1T2"


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v) if v else None


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    x = x.astype(np.float64, copy=False)
    return float(x.mean()), float(x.std(ddof=0))


def _quantiles(xs: list[float], qs: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {f"p{int(q*100):02d}": float("nan") for q in qs}
    out: dict[str, float] = {}
    for q in qs:
        out[f"p{int(q*100):02d}"] = float(np.quantile(arr, q))
    return out


def _aggregate_numeric(xs: list[float]) -> dict[str, Any]:
    arr = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "quantiles": _quantiles(xs, [0.05, 0.25, 0.5, 0.75, 0.95]),
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "quantiles": _quantiles(xs, [0.05, 0.25, 0.5, 0.75, 0.95]),
    }


def _case_id_from_label_path(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return p.stem


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    zooms = img.header.get_zooms()[:3]
    if len(zooms) != 3:
        raise RuntimeError(f"Unexpected zooms for {path}: {img.header.get_zooms()}")
    return data, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


@dataclass(frozen=True)
class CaseStats:
    case: str
    n_slices: int
    n_tumor_slices: int
    tumor_voxels: int
    tumor_mm3: float
    t1_mean: float | None
    t1_std: float | None
    t2_mean: float | None
    t2_std: float | None


def _count_tumor_slices(mask: np.ndarray) -> int:
    # Nibabel yields arrays in (X, Y, Z) for typical nnU-Net NIfTIs. We treat axis=2 as Z.
    if mask.ndim != 3:
        raise RuntimeError(f"Expected 3D mask, got shape={mask.shape}")
    return int(np.any(mask > 0, axis=(0, 1)).sum())


def _compute_case(
    case: str,
    *,
    labels_dir: Path,
    images_dir: Path | None,
    intensity_region: str,
) -> CaseStats:
    lbl_path = labels_dir / f"{case}.nii.gz"
    mask, (sx, sy, sz) = _load_nifti(lbl_path)
    mask_bool = mask > 0

    if mask_bool.ndim != 3:
        raise RuntimeError(f"Expected 3D label for {case}, got {mask_bool.shape}")

    n_slices = int(mask_bool.shape[2])
    n_tumor_slices = _count_tumor_slices(mask_bool)
    tumor_vox = int(mask_bool.sum())
    tumor_mm3 = float(tumor_vox * sx * sy * sz)

    t1_mean = t1_std = t2_mean = t2_std = None
    if images_dir is not None:
        t1_path = images_dir / f"{case}_0000.nii.gz"
        t2_path = images_dir / f"{case}_0001.nii.gz"
        if t1_path.is_file() and t2_path.is_file():
            t1, _ = _load_nifti(t1_path)
            t2, _ = _load_nifti(t2_path)

            if intensity_region == "tumor":
                region = mask_bool
            else:
                region = np.ones_like(mask_bool, dtype=bool)

            t1_mean, t1_std = _mean_std(t1[region])
            t2_mean, t2_std = _mean_std(t2[region])

    return CaseStats(
        case=case,
        n_slices=n_slices,
        n_tumor_slices=n_tumor_slices,
        tumor_voxels=tumor_vox,
        tumor_mm3=tumor_mm3,
        t1_mean=t1_mean,
        t1_std=t1_std,
        t2_mean=t2_mean,
        t2_std=t2_std,
    )


def _maybe_load_fingerprint(preprocessed_root: Path | None) -> dict[str, Any] | None:
    if preprocessed_root is None:
        return None
    fp = preprocessed_root / DATASET / "dataset_fingerprint.json"
    if not fp.is_file():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=None, help="Path to nnUNet_raw (defaults to $nnUNet_raw).")
    ap.add_argument("--preprocessed-root", type=Path, default=None, help="Path to nnUNet_preprocessed (defaults to $nnUNet_preprocessed).")
    ap.add_argument("--dataset", default=DATASET, help=f"Dataset folder name (default: {DATASET}).")
    ap.add_argument(
        "--intensity-region",
        choices=["tumor", "whole"],
        default="tumor",
        help="Compute mean/std within tumor mask, or over the whole image.",
    )
    ap.add_argument("--json", type=Path, default=Path("reports/dataset503_stats.json"), help="Output JSON path.")
    ap.add_argument("--csv", type=Path, default=None, help="If set, also write per-case CSV here.")
    args = ap.parse_args()

    raw_root = (args.raw_root or _env_path("nnUNet_raw"))
    if raw_root is None:
        raise SystemExit("Missing nnUNet_raw. Set $nnUNet_raw or pass --raw-root.")
    raw_root = raw_root.resolve()

    pre_root = (args.preprocessed_root or _env_path("nnUNet_preprocessed"))
    pre_root = pre_root.resolve() if pre_root is not None else None

    ds_dir = raw_root / args.dataset
    labels_dir = ds_dir / "labelsTr"
    images_dir = ds_dir / "imagesTr"

    if not labels_dir.is_dir():
        raise SystemExit(f"Missing labelsTr dir: {labels_dir}")

    label_files = sorted(labels_dir.glob("*.nii.gz"))
    cases = [_case_id_from_label_path(p) for p in label_files]
    if not cases:
        raise SystemExit(f"No label NIfTIs found under {labels_dir}")

    images_dir_opt: Path | None = images_dir if images_dir.is_dir() else None

    per_case: list[dict[str, Any]] = []
    n_slices_list: list[float] = []
    n_tumor_slices_list: list[float] = []
    tumor_vox_list: list[float] = []
    tumor_mm3_list: list[float] = []
    t1_mean_list: list[float] = []
    t1_std_list: list[float] = []
    t2_mean_list: list[float] = []
    t2_std_list: list[float] = []

    for cid in cases:
        st = _compute_case(
            cid,
            labels_dir=labels_dir,
            images_dir=images_dir_opt,
            intensity_region=args.intensity_region,
        )
        per_case.append(
            {
                "case": st.case,
                "n_slices": st.n_slices,
                "n_tumor_slices": st.n_tumor_slices,
                "tumor_voxels": st.tumor_voxels,
                "tumor_mm3": st.tumor_mm3,
                "t1_mean": st.t1_mean,
                "t1_std": st.t1_std,
                "t2_mean": st.t2_mean,
                "t2_std": st.t2_std,
            }
        )
        n_slices_list.append(float(st.n_slices))
        n_tumor_slices_list.append(float(st.n_tumor_slices))
        tumor_vox_list.append(float(st.tumor_voxels))
        tumor_mm3_list.append(float(st.tumor_mm3))
        if st.t1_mean is not None:
            t1_mean_list.append(float(st.t1_mean))
        if st.t1_std is not None:
            t1_std_list.append(float(st.t1_std))
        if st.t2_mean is not None:
            t2_mean_list.append(float(st.t2_mean))
        if st.t2_std is not None:
            t2_std_list.append(float(st.t2_std))

    fingerprint = _maybe_load_fingerprint(pre_root)

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "raw_root": str(raw_root),
        "labels_dir": str(labels_dir),
        "images_dir": str(images_dir) if images_dir_opt is not None else None,
        "intensity_region": args.intensity_region,
        "n_cases": len(per_case),
        "aggregates": {
            "n_slices": _aggregate_numeric(n_slices_list),
            "n_tumor_slices": _aggregate_numeric(n_tumor_slices_list),
            "tumor_voxels": _aggregate_numeric(tumor_vox_list),
            "tumor_mm3": _aggregate_numeric(tumor_mm3_list),
            "t1_mean": _aggregate_numeric(t1_mean_list),
            "t1_std": _aggregate_numeric(t1_std_list),
            "t2_mean": _aggregate_numeric(t2_mean_list),
            "t2_std": _aggregate_numeric(t2_std_list),
        },
        "fingerprint_fallback": fingerprint,
        "per_case": per_case,
    }

    out_json = args.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.csv is not None:
        out_csv = args.csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "case",
                    "n_slices",
                    "n_tumor_slices",
                    "tumor_voxels",
                    "tumor_mm3",
                    "t1_mean",
                    "t1_std",
                    "t2_mean",
                    "t2_std",
                ],
            )
            w.writeheader()
            for row in per_case:
                w.writerow(row)

    print(f"[wrote] {out_json}")
    if args.csv is not None:
        print(f"[wrote] {args.csv}")

    if images_dir_opt is None:
        print("[note] imagesTr not found; per-channel mean/std not computed from raw images.")
        if fingerprint is not None:
            print("[note] fingerprint_fallback was included from nnUNet_preprocessed.")
        else:
            print("[note] nnUNet_preprocessed fingerprint not found; means/std are missing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

