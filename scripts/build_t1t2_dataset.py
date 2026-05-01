#!/usr/bin/env python3
"""Build Dataset503_ALT_T1T2: a 2-channel (T1 + T2 on T1 grid) nnU-Net v2 dataset.

For each patient that has both T1/ and T2/ folders in the repo root:

1. Read T1 image, apply intensity fix, use it as the reference grid.
2. Read T1 label, align to T1 image grid.
3. Read T2 image, apply intensity fix, resample onto T1 grid (linear).
4. Read T2 label, align to T2 image grid, resample onto T1 grid (nearest).
5. Fuse the two masks (union by default) into a single binary GT.

Output layout (under ``nnUNet_raw``):

    Dataset503_ALT_T1T2/
        imagesTr/IOGxx_0000.nii.gz    # T1
        imagesTr/IOGxx_0001.nii.gz    # T2 resampled onto T1 grid
        labelsTr/IOGxx.nii.gz         # fused binary mask {0, 1}
        dataset.json
        fusion_report.json            # per-case QC metrics

Reuses helpers from :mod:`convert_to_nnunet` (``_fix_intensity``,
``_prebinarize``, ``align_label_to_image``, ``find_pair``,
``MIN_FOREGROUND_RETAINED``, ``GEOM_TOL``) so intensity repairs and
label-alignment guards stay in sync with the single-modality pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_nnunet import (  # noqa: E402
    GEOM_TOL,
    MIN_FOREGROUND_RETAINED,
    _fix_intensity,
    _geom_matches,
    _prebinarize,
    align_label_to_image,
    find_pair,
)


DATASET_NAME = "Dataset503_ALT_T1T2"
FUSION_MODES = ("union", "intersection", "staple")


def _resample_image_to(reference_img: sitk.Image, moving_img: sitk.Image) -> sitk.Image:
    """Linear-interp resample ``moving_img`` onto ``reference_img`` grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(sitk.Cast(moving_img, sitk.sitkFloat32))


def _resample_mask_to(reference_img: sitk.Image, moving_mask: sitk.Image) -> sitk.Image:
    """Nearest-neighbor resample of a binary mask onto ``reference_img`` grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(moving_mask)


def _direction_matches(a: sitk.Image, b: sitk.Image, tol: float = 0.05) -> bool:
    da = np.asarray(a.GetDirection())
    db = np.asarray(b.GetDirection())
    return bool(np.max(np.abs(da - db)) <= tol)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    denom = int(a_bool.sum()) + int(b_bool.sum())
    if denom == 0:
        return float("nan")
    inter = int(np.logical_and(a_bool, b_bool).sum())
    return 2.0 * inter / denom


def _fuse(t1_mask: np.ndarray, t2_on_t1_mask: np.ndarray, mode: str) -> np.ndarray:
    a = t1_mask.astype(bool)
    b = t2_on_t1_mask.astype(bool)
    if mode == "union":
        return np.logical_or(a, b).astype(np.uint8)
    if mode == "intersection":
        return np.logical_and(a, b).astype(np.uint8)
    if mode == "staple":
        img_a = sitk.GetImageFromArray(a.astype(np.uint8))
        img_b = sitk.GetImageFromArray(b.astype(np.uint8))
        staple = sitk.STAPLE([img_a, img_b], 1.0)
        staple_arr = sitk.GetArrayFromImage(staple)
        return (staple_arr > 0.5).astype(np.uint8)
    raise ValueError(f"Unknown fusion mode: {mode!r}")


def _save_image_like_reference(
    arr: np.ndarray, reference_img: sitk.Image, out_path: Path, pixel_type
) -> None:
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference_img)
    out = sitk.Cast(out, pixel_type)
    sitk.WriteImage(out, str(out_path))


def _zeros_like(reference_img: sitk.Image) -> sitk.Image:
    """Return an all-zero float32 image with the same geometry as reference_img."""
    out = sitk.Image(reference_img.GetSize(), sitk.sitkFloat32)
    out.SetSpacing(reference_img.GetSpacing())
    out.SetOrigin(reference_img.GetOrigin())
    out.SetDirection(reference_img.GetDirection())
    return out


def build_case(
    case_dir_t1: Path,
    case_dir_t2: Path,
    *,
    fusion_mode: str,
    strict: bool,
) -> dict | None:
    """Process one patient; return a dict bundle or None if skipped.

    Return format:
      {"mode": "aligned", "report": <dict>, "samples": [<sample>]}
      {"mode": "split_mismatch", "report": <dict>, "samples": [<sample>, <sample>]}

    Each sample is:
      {"case_id": str, "ch0": sitk.Image, "ch1": sitk.Image, "mask_arr": np.ndarray, "mask_ref": sitk.Image}
    """
    case_id = case_dir_t1.name
    t1_pair = find_pair(case_dir_t1)
    t2_pair = find_pair(case_dir_t2)
    if t1_pair is None or t2_pair is None:
        print(f"[{case_id}] skipped: missing T1 or T2 image/label pair")
        return None

    t1_img_path, t1_lbl_path = t1_pair
    t2_img_path, t2_lbl_path = t2_pair

    t1_img = sitk.ReadImage(str(t1_img_path))
    t1_img = _fix_intensity(t1_img, case_id=f"T1/{case_id}")

    t2_img_native = sitk.ReadImage(str(t2_img_path))
    t2_img_native = _fix_intensity(t2_img_native, case_id=f"T2/{case_id}")

    t1_lbl_raw = sitk.ReadImage(str(t1_lbl_path))
    t2_lbl_raw = sitk.ReadImage(str(t2_lbl_path))

    t1_lbl = align_label_to_image(
        t1_lbl_raw, reference_img=t1_img, case_id=f"T1/{case_id}"
    )
    t2_lbl_on_t2 = align_label_to_image(
        t2_lbl_raw, reference_img=t2_img_native, case_id=f"T2/{case_id}"
    )

    direction_mismatch = not _direction_matches(t1_img, t2_img_native)
    same_grid, _reason = _geom_matches(t1_img, t2_img_native, tol=GEOM_TOL)

    t1_arr = sitk.GetArrayFromImage(t1_lbl).astype(bool)
    t2_native_arr = sitk.GetArrayFromImage(t2_lbl_on_t2).astype(bool)

    # Strategy for direction-mismatch: do not create a misregistered 2-channel sample.
    # Instead, emit two derived single-modality samples (with a blank other channel).
    if direction_mismatch:
        derived_t1 = f"{case_id}_T1"
        derived_t2 = f"{case_id}_T2"
        report = {
            "case_id": case_id,
            "mode": "split_mismatch",
            "fusion_mode": fusion_mode,
            "derived_cases": [derived_t1, derived_t2],
            "direction_mismatch": True,
            "same_grid": same_grid,
            "t1_fg": int(t1_arr.sum()),
            "t2_fg_native": int(t2_native_arr.sum()),
            "t1_size": list(t1_img.GetSize()),
            "t2_size": list(t2_img_native.GetSize()),
            "t1_spacing": [round(s, 4) for s in t1_img.GetSpacing()],
            "t2_spacing": [round(s, 4) for s in t2_img_native.GetSpacing()],
        }
        return {
            "mode": "split_mismatch",
            "report": report,
            "samples": [
                {
                    "case_id": derived_t1,
                    "ch0": t1_img,
                    "ch1": _zeros_like(t1_img),
                    "mask_arr": t1_arr.astype(np.uint8),
                    "mask_ref": t1_img,
                },
                {
                    "case_id": derived_t2,
                    "ch0": _zeros_like(t2_img_native),
                    "ch1": t2_img_native,
                    "mask_arr": t2_native_arr.astype(np.uint8),
                    "mask_ref": t2_img_native,
                },
            ],
        }

    # Aligned strategy: resample T2 onto T1 grid and fuse masks.
    t2_img_on_t1 = _resample_image_to(t1_img, t2_img_native)
    t2_lbl_on_t1 = _resample_mask_to(t1_img, t2_lbl_on_t2)
    t2_on_t1_arr = sitk.GetArrayFromImage(t2_lbl_on_t1).astype(bool)

    t2_fg_native = int(t2_native_arr.sum())
    t2_fg_on_t1 = int(t2_on_t1_arr.sum())
    if t2_fg_native == 0:
        retained = float("nan")
    else:
        retained = t2_fg_on_t1 / t2_fg_native

    retained_ok = np.isnan(retained) or retained >= MIN_FOREGROUND_RETAINED
    if not retained_ok:
        msg = (
            f"[{case_id}] T2 label lost too much foreground after resampling "
            f"onto T1 grid ({t2_fg_native} -> {t2_fg_on_t1}, "
            f"retained {retained:.0%}, threshold "
            f"{MIN_FOREGROUND_RETAINED:.0%})."
        )
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg} case skipped")
        return None

    fused_arr = _fuse(t1_arr, t2_on_t1_arr, fusion_mode)

    report = {
        "case_id": case_id,
        "mode": "aligned",
        "fusion_mode": fusion_mode,
        "derived_cases": [case_id],
        "direction_mismatch": False,
        "same_grid": same_grid,
        "t1_fg": int(t1_arr.sum()),
        "t2_fg_native": t2_fg_native,
        "t2_fg_on_t1": t2_fg_on_t1,
        "retained": None if np.isnan(retained) else round(retained, 4),
        "dice_raters": round(_dice(t1_arr, t2_on_t1_arr), 4)
        if not np.isnan(_dice(t1_arr, t2_on_t1_arr))
        else None,
        "fused_fg": int(fused_arr.sum()),
        "t1_size": list(t1_img.GetSize()),
        "t2_size": list(t2_img_native.GetSize()),
        "t1_spacing": [round(s, 4) for s in t1_img.GetSpacing()],
        "t2_spacing": [round(s, 4) for s in t2_img_native.GetSpacing()],
    }

    return {
        "mode": "aligned",
        "report": report,
        "samples": [
            {
                "case_id": case_id,
                "ch0": t1_img,
                "ch1": t2_img_on_t1,
                "mask_arr": fused_arr.astype(np.uint8),
                "mask_ref": t1_img,
            }
        ],
    }


def build_dataset(
    src_root: Path,
    dst_root: Path,
    *,
    fusion_mode: str,
    strict: bool,
) -> int:
    dst = dst_root / DATASET_NAME
    images_dir = dst / "imagesTr"
    labels_dir = dst / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Ensure repeated runs are consistent: clear previous outputs if present.
    for d in (images_dir, labels_dir):
        for p in d.glob("*.nii.gz"):
            p.unlink()
    for p in (dst / "dataset.json", dst / "fusion_report.json"):
        if p.exists():
            p.unlink()

    t1_root = src_root / "T1"
    t2_root = src_root / "T2"
    if not t1_root.exists() or not t2_root.exists():
        raise FileNotFoundError(
            f"Both {t1_root} and {t2_root} must exist to build {DATASET_NAME}"
        )

    t1_cases = {p.name for p in t1_root.iterdir() if p.is_dir()}
    t2_cases = {p.name for p in t2_root.iterdir() if p.is_dir()}
    common = sorted(t1_cases & t2_cases)
    t1_only = sorted(t1_cases - t2_cases)
    t2_only = sorted(t2_cases - t1_cases)
    if t1_only:
        print(f"[info] T1-only cases (dropped): {t1_only}")
    if t2_only:
        print(f"[info] T2-only cases (dropped): {t2_only}")
    print(f"[info] building {DATASET_NAME} from {len(common)} common patients "
          f"(fusion={fusion_mode})")

    reports: list[dict] = []
    written = 0  # number of training cases written (derived cases, not patients)
    for case_id in common:
        case_dir_t1 = t1_root / case_id
        case_dir_t2 = t2_root / case_id
        try:
            result = build_case(
                case_dir_t1,
                case_dir_t2,
                fusion_mode=fusion_mode,
                strict=strict,
            )
        except ValueError as exc:
            raise ValueError(f"[{case_id}] {exc}") from exc
        if result is None:
            continue
        report = result["report"]
        samples = result["samples"]

        for s in samples:
            out_id = s["case_id"]
            sitk.WriteImage(s["ch0"], str(images_dir / f"{out_id}_0000.nii.gz"))
            sitk.WriteImage(s["ch1"], str(images_dir / f"{out_id}_0001.nii.gz"))
            _save_image_like_reference(
                s["mask_arr"], s["mask_ref"], labels_dir / f"{out_id}.nii.gz", sitk.sitkUInt8
            )
            written += 1

        reports.append(report)
        if report.get("mode") == "aligned":
            print(
                f"[{case_id}] mode=aligned T1fg={report['t1_fg']} T2fg->T1={report['t2_fg_on_t1']} "
                f"retained={report['retained']} dice_raters={report['dice_raters']} "
                f"fused_fg={report['fused_fg']} dir_mismatch={report['direction_mismatch']}"
            )
        else:
            print(
                f"[{case_id}] mode=split_mismatch derived={report['derived_cases']} "
                f"T1fg={report['t1_fg']} T2fg_native={report['t2_fg_native']}"
            )

    dataset_json = {
        "channel_names": {"0": "T1", "1": "T2"},
        "labels": {"background": 0, "ALT": 1},
        "numTraining": written,
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": (
            "Atypical lipomatous tumor (ALT) MRI segmentation, 2-channel. "
            f"Aligned cases use (T1 + T2 resampled onto T1 grid) with labels fused by {fusion_mode}. "
            "Cases with direction-mismatch are split into two derived samples "
            "(<case>_T1 and <case>_T2) with the missing channel filled with zeros "
            "and modality-native labels/grids."
        ),
    }
    with open(dst / "dataset.json", "w") as fh:
        json.dump(dataset_json, fh, indent=2)

    with open(dst / "fusion_report.json", "w") as fh:
        json.dump(
            {
                "fusion_mode": fusion_mode,
                "min_foreground_retained": MIN_FOREGROUND_RETAINED,
                "num_written": written,
                "num_common_patients": len(common),
                "notes": (
                    "num_written counts derived training cases. "
                    "Aligned patients emit one case_id. "
                    "direction-mismatch patients emit two derived cases: <case>_T1 and <case>_T2 "
                    "with a blank other channel and modality-native labels."
                ),
                "cases": reports,
            },
            fh,
            indent=2,
        )

    print(f"[done] wrote {written} cases to {dst}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing T1/ and T2/ folders.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output nnUNet_raw directory. Defaults to $nnUNet_raw.",
    )
    parser.add_argument(
        "--fusion",
        choices=FUSION_MODES,
        default="union",
        help="How to combine T1 and T2 expert masks. Default: union.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Abort on any case that loses too much label foreground when "
            "resampling onto the T1 grid (default: skip such cases with a "
            "warning)."
        ),
    )
    args = parser.parse_args()

    dst = args.dst
    if dst is None:
        env_raw = os.environ.get("nnUNet_raw")
        if not env_raw:
            raise SystemExit(
                "nnUNet_raw env var is not set and --dst was not provided."
            )
        dst = Path(env_raw)
    dst.mkdir(parents=True, exist_ok=True)

    build_dataset(args.src, dst, fusion_mode=args.fusion, strict=args.strict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
