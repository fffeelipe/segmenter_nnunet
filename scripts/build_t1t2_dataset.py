#!/usr/bin/env python3
"""Build Dataset503_ALT_T1T2: a 2-channel (T1 + T2 on a common reference grid) nnU-Net v2 dataset.

For each patient that has both T1/ and T2/ folders in the repo root:

1. Read T1/T2 images, apply intensity fixes.
2. Build a per-case common reference grid (isotropic, intersection-FOV).
3. Resample both images onto the common grid (linear).
4. Align each label to its modality image grid, then resample both labels onto
   the common grid (nearest).
5. Fuse the two masks (union by default) into a single binary GT.

Output layout (under ``nnUNet_raw``):

    Dataset503_ALT_T1T2/
        imagesTr/IOGxx_0000.nii.gz    # T1 resampled onto common grid
        imagesTr/IOGxx_0001.nii.gz    # T2 resampled onto common grid
        labelsTr/IOGxx.nii.gz         # fused binary mask {0, 1}
        dataset.json
        fusion_report.json            # per-case QC metrics
        holdout/images|labels/        # optional: ``--exclude-cases-file`` (patient IOGxx)

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
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_nnunet import (  # noqa: E402
    GEOM_TOL,
    MIN_FOREGROUND_RETAINED,
    _clip_percentiles,
    _default_workers,
    _fix_intensity,
    _geom_matches,
    _guess_physical_cores,
    _prebinarize,
    align_label_to_image,
    find_pair,
    load_exclude_case_ids,
)


DATASET_NAME = "Dataset503_ALT_T1T2"
DATASET_NAME_CLIP = "Dataset504_ALT_T1T2_clip"
FUSION_MODES = ("union", "intersection", "staple")
FOV_POLICY = "intersection"


def _tmp_path(final_path: Path) -> Path:
    # Same directory for atomic os.replace; include PID to avoid collisions.
    # IMPORTANT: keep a recognized suffix (e.g. .nii.gz) so SimpleITK can pick an ImageIO.
    name = final_path.name
    pid = os.getpid()
    if name.endswith(".nii.gz"):
        return final_path.with_name(name[:-7] + f".tmp.{pid}.nii.gz")
    if name.endswith(".nii"):
        return final_path.with_name(name[:-4] + f".tmp.{pid}.nii")
    # Fallback: preserve suffix if present.
    return final_path.with_name(name + f".tmp.{pid}{final_path.suffix}")


def _write_sitk_atomic(img: sitk.Image, final_path: Path) -> None:
    tmp = _tmp_path(final_path)
    try:
        sitk.WriteImage(img, str(tmp))
        os.replace(str(tmp), str(final_path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            # Best-effort cleanup only; atomicity is handled by os.replace.
            pass


def _write_label_atomic(
    arr: np.ndarray, reference_img: sitk.Image, final_path: Path, pixel_type
) -> None:
    tmp = _tmp_path(final_path)
    try:
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(reference_img)
        out = sitk.Cast(out, pixel_type)
        sitk.WriteImage(out, str(tmp))
        os.replace(str(tmp), str(final_path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _min_isotropic_spacing(a: sitk.Image, b: sitk.Image) -> float:
    """Pick isotropic spacing = min of both images' spacings (across all axes)."""
    sa = np.asarray(a.GetSpacing(), dtype=float)
    sb = np.asarray(b.GetSpacing(), dtype=float)
    s = float(np.min(np.concatenate([sa, sb])))
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Invalid spacing encountered (min spacing={s}).")
    return s


def _image_corners_physical(img: sitk.Image) -> list[tuple[float, float, float]]:
    """Return the 8 corner points of img in physical space."""
    size = img.GetSize()
    if len(size) != 3:
        raise ValueError(f"Expected 3D image, got size={size}")
    max_idx = (size[0] - 1, size[1] - 1, size[2] - 1)
    corners = []
    for ix in (0, max_idx[0]):
        for iy in (0, max_idx[1]):
            for iz in (0, max_idx[2]):
                corners.append(tuple(img.TransformIndexToPhysicalPoint((ix, iy, iz))))
    return corners


def _bounds_in_reference_index_space(
    reference_img: sitk.Image, moving_img: sitk.Image
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of moving_img expressed in reference index space.

    We transform the moving image's physical corners into the reference image's
    continuous index coordinates, then take min/max per axis.
    """
    pts = _image_corners_physical(moving_img)
    idxs = np.asarray(
        [reference_img.TransformPhysicalPointToContinuousIndex(p) for p in pts],
        dtype=float,
    )
    mins = np.min(idxs, axis=0)
    maxs = np.max(idxs, axis=0)
    return mins, maxs


def _build_reference_grid_intersection_isotropic(
    t1_img: sitk.Image, t2_img: sitk.Image
) -> tuple[sitk.Image | None, dict]:
    """Create a per-case reference grid: isotropic spacing, intersection-FOV.

    Output grid uses T1 direction as axes. The FOV is the intersection of the two
    images' physical extents expressed in that coordinate frame (so no padding).
    """
    iso = _min_isotropic_spacing(t1_img, t2_img)
    t1_spacing = np.asarray(t1_img.GetSpacing(), dtype=float)

    t1_mins, t1_maxs = _bounds_in_reference_index_space(t1_img, t1_img)
    t2_mins, t2_maxs = _bounds_in_reference_index_space(t1_img, t2_img)
    inter_mins = np.maximum(t1_mins, t2_mins)
    inter_maxs = np.minimum(t1_maxs, t2_maxs)

    # Continuous-index extent in T1 voxel units.
    extent_idx = inter_maxs - inter_mins
    if np.any(~np.isfinite(extent_idx)) or np.any(extent_idx <= 0):
        return None, {
            "ok": False,
            "reason": "empty_intersection_fov",
            "iso_spacing": round(float(iso), 6),
        }

    extent_mm = extent_idx * t1_spacing
    out_spacing = np.asarray([iso, iso, iso], dtype=float)
    out_size = np.maximum(1, np.ceil(extent_mm / out_spacing)).astype(int)

    out = sitk.Image([int(x) for x in out_size.tolist()], sitk.sitkFloat32)
    out.SetSpacing(tuple(float(x) for x in out_spacing.tolist()))
    out.SetDirection(t1_img.GetDirection())
    out.SetOrigin(tuple(t1_img.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in inter_mins))))
    return out, {
        "ok": True,
        "reason": "ok",
        "iso_spacing": round(float(iso), 6),
        "fov_policy": FOV_POLICY,
        "ref_size": [int(x) for x in out_size.tolist()],
        "ref_spacing": [round(float(x), 6) for x in out_spacing.tolist()],
    }


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
    _write_label_atomic(arr, reference_img, out_path, pixel_type)


def _zeros_like(reference_img: sitk.Image) -> sitk.Image:
    """Return an all-zero float32 image with the same geometry as reference_img."""
    out = sitk.Image(reference_img.GetSize(), sitk.sitkFloat32)
    out.SetSpacing(reference_img.GetSpacing())
    out.SetOrigin(reference_img.GetOrigin())
    out.SetDirection(reference_img.GetDirection())
    return out


def _build_and_write_one_case(
    case_id: str,
    case_dir_t1: str,
    case_dir_t2: str,
    images_dir: str,
    labels_dir: str,
    *,
    fusion_mode: str,
    strict: bool,
    clip_p_lo: float | None = None,
    clip_p_hi: float | None = None,
) -> dict:
    """Worker entrypoint: build one patient and write outputs for its samples."""
    result = build_case(
        Path(case_dir_t1),
        Path(case_dir_t2),
        fusion_mode=fusion_mode,
        strict=strict,
        clip_p_lo=clip_p_lo,
        clip_p_hi=clip_p_hi,
    )
    if result is None:
        return {"case_id": case_id, "mode": "skipped", "report": None, "n_written": 0}

    report = result["report"]
    samples = result["samples"]

    images_dir_p = Path(images_dir)
    labels_dir_p = Path(labels_dir)

    n_written = 0
    for s in samples:
        out_id = s["case_id"]
        _write_sitk_atomic(s["ch0"], images_dir_p / f"{out_id}_0000.nii.gz")
        _write_sitk_atomic(s["ch1"], images_dir_p / f"{out_id}_0001.nii.gz")
        _save_image_like_reference(
            s["mask_arr"], s["mask_ref"], labels_dir_p / f"{out_id}.nii.gz", sitk.sitkUInt8
        )
        n_written += 1

    return {"case_id": case_id, "mode": report.get("mode"), "report": report, "n_written": n_written}


def build_case(
    case_dir_t1: Path,
    case_dir_t2: Path,
    *,
    fusion_mode: str,
    strict: bool,
    clip_p_lo: float | None = None,
    clip_p_hi: float | None = None,
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

    if clip_p_lo is not None and clip_p_hi is not None:
        t1_img = _clip_percentiles(
            t1_img, p_lo=clip_p_lo, p_hi=clip_p_hi, case_id=f"T1/{case_id}"
        )
        t2_img_native = _clip_percentiles(
            t2_img_native,
            p_lo=clip_p_lo,
            p_hi=clip_p_hi,
            case_id=f"T2/{case_id}",
        )

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

    t1_native_arr = sitk.GetArrayFromImage(t1_lbl).astype(bool)
    t2_native_arr = sitk.GetArrayFromImage(t2_lbl_on_t2).astype(bool)

    # Build per-case common grid (intersection-FOV, isotropic min spacing).
    ref_img, ref_meta = _build_reference_grid_intersection_isotropic(t1_img, t2_img_native)
    if ref_img is None:
        # Cannot form a meaningful shared FOV. Direction mismatch or not, emit split samples.
        derived_t1 = f"{case_id}_T1"
        derived_t2 = f"{case_id}_T2"
        report = {
            "case_id": case_id,
            "mode": "split_mismatch",
            "fusion_mode": fusion_mode,
            "derived_cases": [derived_t1, derived_t2],
            "direction_mismatch": direction_mismatch,
            "same_grid": same_grid,
            "join_attempted": False,
            "join_qc_passed": False,
            "join_fail_reason": ref_meta.get("reason"),
            "ref": ref_meta,
            # Legacy fields used by inspect_fusion.py and old logs:
            "t1_fg": int(t1_native_arr.sum()),
            "t2_fg_native": int(t2_native_arr.sum()),
            "t2_fg_on_t1": None,
            "retained": None,
            "t1_fg_native": int(t1_native_arr.sum()),
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
                    "mask_arr": t1_native_arr.astype(np.uint8),
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

    # Attempt join: resample both modalities + both labels onto the common grid.
    t1_img_on_ref = _resample_image_to(ref_img, t1_img)
    t2_img_on_ref = _resample_image_to(ref_img, t2_img_native)
    t1_lbl_on_ref = _resample_mask_to(ref_img, t1_lbl)
    t2_lbl_on_ref = _resample_mask_to(ref_img, t2_lbl_on_t2)

    t1_on_ref_arr = sitk.GetArrayFromImage(t1_lbl_on_ref).astype(bool)
    t2_on_ref_arr = sitk.GetArrayFromImage(t2_lbl_on_ref).astype(bool)

    def _retained(native: np.ndarray, on_ref: np.ndarray) -> float:
        n0 = int(native.sum())
        n1 = int(on_ref.sum())
        if n0 == 0:
            return float("nan")
        return n1 / n0

    t1_retained = _retained(t1_native_arr, t1_on_ref_arr)
    t2_retained = _retained(t2_native_arr, t2_on_ref_arr)
    dice_raters = _dice(t1_on_ref_arr, t2_on_ref_arr)

    retained_ok = (
        (np.isnan(t1_retained) or t1_retained >= MIN_FOREGROUND_RETAINED)
        and (np.isnan(t2_retained) or t2_retained >= MIN_FOREGROUND_RETAINED)
    )

    qc_ok = retained_ok
    qc_reason = None
    if not retained_ok:
        qc_reason = "foreground_lost_on_resample"

    # Under direction mismatch, require an extra plausibility check: non-trivial agreement.
    # (Many real cases can have low Dice, so we keep this conservative and rely primarily on retention.)
    if direction_mismatch and qc_ok:
        if np.isnan(dice_raters) or dice_raters == 0.0:
            qc_ok = False
            qc_reason = "direction_mismatch_and_zero_dice"

    if not qc_ok:
        msg = (
            f"[{case_id}] join QC failed on common grid (reason={qc_reason}, "
            f"t1_retained={t1_retained:.3f}, t2_retained={t2_retained:.3f}, "
            f"dice={dice_raters:.3f} dir_mismatch={direction_mismatch})."
        )
        if strict and (qc_reason == "foreground_lost_on_resample"):
            raise ValueError(msg)
        print(f"WARNING: {msg} falling back to split samples")

        derived_t1 = f"{case_id}_T1"
        derived_t2 = f"{case_id}_T2"
        report = {
            "case_id": case_id,
            "mode": "split_mismatch",
            "fusion_mode": fusion_mode,
            "derived_cases": [derived_t1, derived_t2],
            "direction_mismatch": direction_mismatch,
            "same_grid": same_grid,
            "join_attempted": True,
            "join_qc_passed": False,
            "join_fail_reason": qc_reason,
            "ref": ref_meta,
            # Legacy fields used by inspect_fusion.py and old logs:
            "t1_fg": int(t1_native_arr.sum()),
            "t2_fg_native": int(t2_native_arr.sum()),
            "t2_fg_on_t1": int(t2_on_ref_arr.sum()),
            "retained": None if np.isnan(t2_retained) else round(float(t2_retained), 4),
            "t1_fg_native": int(t1_native_arr.sum()),
            "t1_fg_on_ref": int(t1_on_ref_arr.sum()),
            "t2_fg_on_ref": int(t2_on_ref_arr.sum()),
            "t1_retained": None if np.isnan(t1_retained) else round(float(t1_retained), 4),
            "t2_retained": None if np.isnan(t2_retained) else round(float(t2_retained), 4),
            "dice_raters_on_ref": None if np.isnan(dice_raters) else round(float(dice_raters), 4),
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
                    "mask_arr": t1_native_arr.astype(np.uint8),
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

    fused_arr = _fuse(t1_on_ref_arr, t2_on_ref_arr, fusion_mode)

    report = {
        "case_id": case_id,
        "mode": "aligned",
        "fusion_mode": fusion_mode,
        "derived_cases": [case_id],
        "direction_mismatch": direction_mismatch,
        "same_grid": same_grid,
        "join_attempted": True,
        "join_qc_passed": True,
        "join_fail_reason": None,
        "ref": ref_meta,
        # Legacy fields used by inspect_fusion.py and old logs:
        "t1_fg": int(t1_on_ref_arr.sum()),
        "t2_fg_on_t1": int(t2_on_ref_arr.sum()),
        "retained": None if np.isnan(t2_retained) else round(float(t2_retained), 4),
        "t1_fg_native": int(t1_native_arr.sum()),
        "t2_fg_native": int(t2_native_arr.sum()),
        "t1_fg_on_ref": int(t1_on_ref_arr.sum()),
        "t2_fg_on_ref": int(t2_on_ref_arr.sum()),
        "t1_retained": None if np.isnan(t1_retained) else round(float(t1_retained), 4),
        "t2_retained": None if np.isnan(t2_retained) else round(float(t2_retained), 4),
        "dice_raters": None if np.isnan(dice_raters) else round(float(dice_raters), 4),
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
                "ch0": t1_img_on_ref,
                "ch1": t2_img_on_ref,
                "mask_arr": fused_arr.astype(np.uint8),
                "mask_ref": ref_img,
            }
        ],
    }


def build_dataset(
    src_root: Path,
    dst_root: Path,
    *,
    fusion_mode: str,
    strict: bool,
    workers: int,
    exclude_patients: set[str] | None = None,
    clip_p_lo: float | None = None,
    clip_p_hi: float | None = None,
) -> int:
    use_clip = clip_p_lo is not None and clip_p_hi is not None
    ds_name = DATASET_NAME_CLIP if use_clip else DATASET_NAME
    dst = dst_root / ds_name
    images_dir = dst / "imagesTr"
    labels_dir = dst / "labelsTr"
    holdout_images = dst / "holdout" / "images"
    holdout_labels = dst / "holdout" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exclude_patients = exclude_patients or set()

    # Ensure repeated runs are consistent: clear previous outputs if present.
    for d in (images_dir, labels_dir, holdout_images, holdout_labels):
        if d.exists():
            for p in d.glob("*.nii.gz"):
                p.unlink()
    for p in (dst / "dataset.json", dst / "fusion_report.json"):
        if p.exists():
            p.unlink()

    t1_root = src_root / "T1"
    t2_root = src_root / "T2"
    if not t1_root.exists() or not t2_root.exists():
        raise FileNotFoundError(
            f"Both {t1_root} and {t2_root} must exist to build {ds_name}"
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
    n_excluded = len([c for c in common if c in exclude_patients])
    print(
        f"[info] building {ds_name} from {len(common)} common patients "
        f"(fusion={fusion_mode}, holdout_patients={n_excluded}"
        f"{', clip=on' if use_clip else ''})"
    )

    reports: list[dict] = []
    written = 0  # number of training cases written (derived cases, not patients)

    workers = int(workers)
    if workers < 1:
        workers = 1
    print(f"[info] alignment workers: {workers}")

    unknown_ex = sorted(exclude_patients - set(common))
    if unknown_ex:
        print(
            f"[warn] exclude list mentions patients not in T1∩T2 common set "
            f"(ignored): {unknown_ex}"
        )

    # Parallelize across patients; each worker writes its own outputs atomically.
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for case_id in common:
            case_dir_t1 = t1_root / case_id
            case_dir_t2 = t2_root / case_id
            if case_id in exclude_patients:
                holdout_images.mkdir(parents=True, exist_ok=True)
                holdout_labels.mkdir(parents=True, exist_ok=True)
                img_out = str(holdout_images)
                lbl_out = str(holdout_labels)
            else:
                img_out = str(images_dir)
                lbl_out = str(labels_dir)
            fut = ex.submit(
                _build_and_write_one_case,
                case_id,
                str(case_dir_t1),
                str(case_dir_t2),
                img_out,
                lbl_out,
                fusion_mode=fusion_mode,
                strict=strict,
                clip_p_lo=clip_p_lo,
                clip_p_hi=clip_p_hi,
            )
            futures[fut] = case_id

        for fut in as_completed(futures):
            case_id = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:
                # Record a synthetic report entry and keep going.
                msg = f"{type(exc).__name__}: {exc}"
                print(f"ERROR: [{case_id}] worker failed: {msg}")
                reports.append(
                    {
                        "case_id": case_id,
                        "mode": "error",
                        "fusion_mode": fusion_mode,
                        "derived_cases": [],
                        "join_attempted": None,
                        "join_qc_passed": False,
                        "join_fail_reason": "worker_exception",
                        "error": msg,
                    }
                )
                continue

            report = out.get("report")
            if report:
                reports.append(report)
                tag = " [HOLDOUT]" if case_id in exclude_patients else ""
                if report.get("mode") == "aligned":
                    print(
                        f"[{case_id}] mode=aligned T1fg={report.get('t1_fg')} T2fg_on_ref={report.get('t2_fg_on_t1')} "
                        f"t2_retained={report.get('retained')} dice_raters={report.get('dice_raters')} "
                        f"fused_fg={report.get('fused_fg')} dir_mismatch={report.get('direction_mismatch')}{tag}"
                    )
                else:
                    print(
                        f"[{case_id}] mode=split_mismatch derived={report.get('derived_cases')} "
                        f"T1fg={report.get('t1_fg')} T2fg_native={report.get('t2_fg_native')}{tag}"
                    )
            if case_id not in exclude_patients:
                written += int(out.get("n_written") or 0)

    # Make fusion_report.json deterministic across runs.
    reports = sorted(reports, key=lambda r: r.get("case_id", ""))

    dataset_json = {
        "channel_names": {"0": "T1", "1": "T2"},
        "labels": {"background": 0, "ALT": 1},
        "numTraining": written,
        "file_ending": ".nii.gz",
        "name": ds_name,
        "description": (
            "Atypical lipomatous tumor (ALT) MRI segmentation, 2-channel. "
            f"Aligned cases use (T1 + T2 resampled onto a common isotropic intersection-FOV grid) "
            f"with labels fused by {fusion_mode}. "
            + (
                f"Intensities clipped at p{clip_p_lo}/p{clip_p_hi} per channel after _fix_intensity. "
                if use_clip else ""
            )
            + "Cases that fail join QC are split into two derived samples "
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
                "reference_grid": {
                    "fov_policy": FOV_POLICY,
                    "spacing_policy": "min_isotropic_per_case",
                },
                "min_foreground_retained": MIN_FOREGROUND_RETAINED,
                "num_written": written,
                "num_common_patients": len(common),
                "notes": (
                    "num_written counts derived training cases. "
                    "Aligned patients emit one case_id. "
                    "Patients that fail join QC emit two derived cases: <case>_T1 and <case>_T2 "
                    "with a blank other channel and modality-native labels/grids."
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
        "--workers",
        type=int,
        default=_default_workers(),
        help="Number of parallel worker processes for alignment (default: auto ~ half physical cores).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Abort on any case that loses too much label foreground when "
            "resampling onto the common reference grid (default: skip such cases with a "
            "warning)."
        ),
    )
    parser.add_argument(
        "--exclude-cases-file",
        type=Path,
        default=None,
        help=(
            "Patient folder names (IOGxx) to write only under "
            "Dataset503_ALT_T1T2/holdout/{images,labels}/; excluded from imagesTr/labelsTr. "
            "Applies to the whole patient (all derived *_T1 / *_T2 rows for that patient)."
        ),
    )
    parser.add_argument(
        "--percentile-clip",
        action="store_true",
        help=(
            "Clip both channel intensities at --clip-p-lo / --clip-p-hi after "
            "_fix_intensity. Writes to Dataset504_ALT_T1T2_clip so the "
            "un-clipped baseline cache (Dataset503_ALT_T1T2) stays intact."
        ),
    )
    parser.add_argument(
        "--clip-p-lo",
        type=float,
        default=0.5,
        help="Lower percentile for --percentile-clip (default 0.5).",
    )
    parser.add_argument(
        "--clip-p-hi",
        type=float,
        default=99.5,
        help="Upper percentile for --percentile-clip (default 99.5).",
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

    exclude_patients: set[str] = set()
    if args.exclude_cases_file is not None:
        if not args.exclude_cases_file.is_file():
            raise SystemExit(f"--exclude-cases-file not found: {args.exclude_cases_file}")
        exclude_patients = load_exclude_case_ids(args.exclude_cases_file)

    clip_kwargs = (
        {"clip_p_lo": args.clip_p_lo, "clip_p_hi": args.clip_p_hi}
        if args.percentile_clip
        else {}
    )
    build_dataset(
        args.src,
        dst,
        fusion_mode=args.fusion,
        strict=args.strict,
        workers=args.workers,
        exclude_patients=exclude_patients,
        **clip_kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
