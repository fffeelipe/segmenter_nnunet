"""Programmatic equivalent of "open in ITK-SNAP and look": quantify
the *mode* of failure of a segmentation prediction against GT.

Computes, for each of up to N predictions:

1. Per-slice Dice (sorted by the GT area on that slice).
2. Complete-miss slices: GT has tumor, prediction has zero voxels.
3. Over-/under-segmentation profile: mean(|pred| - |gt|) per tumor slice.
4. Core vs shell Dice: splits GT by a 1-voxel 3D erosion. Core Dice
   isolates "does the network find the center?" from "how are the
   boundaries?".
5. Confidence bands (when a prediction is given as a probability map):
   skipped here — nnU-Net argmax predictions are hard masks.

Usage
-----

    python scripts/analyze_failure_mode.py IOG35 \
        --gt nnunet_env/nnUNet_raw/Dataset501_ALT_T1/labelsTr/IOG35.nii.gz \
        --image nnunet_env/nnUNet_raw/Dataset501_ALT_T1/imagesTr/IOG35_0000.nii.gz \
        --pred baseline=nnunet_env/nnUNet_results/.../fold_2/validation/IOG35.nii.gz \
        --pred invgamma=/path/to/invgamma/fold_2/validation/IOG35.nii.gz

If ``--image`` is omitted, image-based summaries (tumor intensity in
FN vs TP regions) are skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from scipy import ndimage as ndi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return float(2 * inter / s) if s else float("nan")


def load_mask(path: Path) -> np.ndarray:
    nii = nib.load(str(path))
    arr = np.asarray(nii.dataobj)
    return (arr > 0).astype(np.uint8)


def detect_slice_axis(gt: np.ndarray) -> int:
    """Pick the axis with the fewest positions that have tumor (i.e. the
    through-plane axis in a typical axial/sagittal/coronal NIfTI).
    Breaks ties toward the largest-indexed axis."""
    best_axis = gt.ndim - 1
    best_count = None
    for axis in range(gt.ndim):
        other_axes = tuple(a for a in range(gt.ndim) if a != axis)
        with_tumor = (gt.sum(axis=other_axes) > 0).sum()
        if best_count is None or with_tumor < best_count:
            best_count = with_tumor
            best_axis = axis
    return best_axis


def per_slice_dice(gt: np.ndarray, pred: np.ndarray, axis: int) -> list[tuple[int, int, int, float]]:
    """Returns list of (z, gt_vox, pred_vox, dice) for every slice along
    ``axis`` where either GT or pred has any tumor voxel. Sorted by
    gt_vox desc."""
    rows = []
    n = gt.shape[axis]
    for z in range(n):
        g = np.take(gt, z, axis=axis) > 0
        p = np.take(pred, z, axis=axis) > 0
        if not g.any() and not p.any():
            continue
        rows.append((z, int(g.sum()), int(p.sum()), dice(g, p)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def core_shell_dice(gt: np.ndarray, pred: np.ndarray, slice_axis: int,
                     erode_iters: int = 1) -> tuple[float, float]:
    """Core = in-plane erosion of GT (per-slice), shell = rest of GT.

    3D erosion erases thin-Z tumors entirely; per-slice 2D erosion is
    more informative for "did the network miss the center or just
    the boundary?"."""
    if not HAS_SCIPY:
        return float("nan"), float("nan")
    if gt.sum() == 0:
        return float("nan"), float("nan")
    gt_b = gt > 0
    core = np.zeros_like(gt_b)
    n = gt.shape[slice_axis]
    for z in range(n):
        g_slice = np.take(gt_b, z, axis=slice_axis)
        if not g_slice.any():
            continue
        core_slice = ndi.binary_erosion(g_slice, iterations=erode_iters)
        slicer = [slice(None)] * gt.ndim
        slicer[slice_axis] = z
        core[tuple(slicer)] = core_slice
    if core.sum() == 0:
        return float("nan"), dice(gt, pred)
    shell = gt_b & ~core
    pred_b = pred > 0
    core_dice = dice(core, pred_b & core)
    shell_dice = dice(shell, pred_b & shell)
    return core_dice, shell_dice


def intensity_stats(image: np.ndarray | None, gt: np.ndarray, pred: np.ndarray) -> dict:
    if image is None:
        return {}
    gt_b = gt > 0
    pr_b = pred > 0
    tp = gt_b & pr_b
    fn = gt_b & ~pr_b
    fp = ~gt_b & pr_b
    out = {}
    for name, mask in [("TP", tp), ("FN", fn), ("FP", fp)]:
        vals = image[mask]
        if vals.size == 0:
            out[name] = None
            continue
        out[name] = {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p50": float(np.percentile(vals, 50)),
        }
    return out


def analyse_pred(name: str, gt: np.ndarray, pred: np.ndarray, image: np.ndarray | None,
                  slice_axis: int) -> None:
    if gt.shape != pred.shape:
        print(f"[{name}] shape mismatch {gt.shape} vs {pred.shape}; resample/skip")
        return

    d = dice(gt, pred)
    core, shell = core_shell_dice(gt, pred, slice_axis)
    n_gt = int(gt.sum())
    n_pred = int(pred.sum())
    tp = int(((gt > 0) & (pred > 0)).sum())
    fn = n_gt - tp
    fp = n_pred - tp

    print(f"\n=== {name} ==========================================")
    print(f"  Dice total           : {d:.4f}")
    print(f"  Dice core / shell    : {core:.4f}  /  {shell:.4f}")
    print(f"  voxels TP/FN/FP      : {tp} / {fn} / {fp}")
    print(f"  |pred| / |GT|        : {n_pred} / {n_gt}  (ratio {n_pred/max(n_gt,1):.3f})")

    rows = per_slice_dice(gt, pred, slice_axis)
    n_slices_gt = sum(1 for r in rows if r[1] > 0)
    n_slices_empty = sum(1 for r in rows if r[1] > 0 and r[2] == 0)
    print(f"  slices with GT       : {n_slices_gt}")
    print(f"  slices completely    : {n_slices_empty}  (GT>0 but pred=0)")
    print(f"    missed              ")

    print("  Top 10 slices by GT area:")
    print(f"    {'z':>4}  {'|GT|':>6}  {'|pred|':>6}  {'dice':>6}")
    for r in rows[:10]:
        z, gv, pv, ds = r
        flag = "  <-- miss" if pv == 0 else ("  <-- over" if pv > 2 * max(gv, 1) else "")
        print(f"    {z:>4}  {gv:>6}  {pv:>6}  {ds:>6.3f}{flag}")

    if n_slices_gt > 10:
        rest = rows[10:n_slices_gt]
        print(f"  ... + {len(rest)} more slices with GT. "
              f"Mean dice on them: {np.mean([r[3] for r in rest]):.3f}")

    stats = intensity_stats(image, gt, pred)
    if stats:
        print("  Intensity on tumor-region voxels:")
        print(f"    {'region':<4}  {'n':>7}  {'mean':>8}  {'std':>7}  {'p50':>7}")
        for name_r, s in stats.items():
            if s is None:
                continue
            print(f"    {name_r:<4}  {s['n']:>7}  {s['mean']:>8.1f}  {s['std']:>7.1f}  {s['p50']:>7.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", help="Case name, e.g. IOG35 (label for plots only)")
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--pred", action="append", default=[],
                        help="Prediction in form name=/path/to/pred.nii.gz. May repeat.")
    args = parser.parse_args()

    if not args.pred:
        print("no --pred given", file=sys.stderr)
        sys.exit(2)

    gt = load_mask(args.gt)
    image = None
    if args.image is not None and args.image.exists():
        image = np.asarray(nib.load(str(args.image)).dataobj, dtype=np.float32)

    slice_axis = detect_slice_axis(gt)
    print(f"Case {args.case}")
    print(f"  GT: {args.gt} shape={gt.shape} vox={int(gt.sum())}  "
          f"slice_axis={slice_axis}")
    if image is not None:
        print(f"  Image: {args.image} shape={image.shape}  "
              f"mean={image.mean():.1f} max={image.max():.1f}")

    for item in args.pred:
        if "=" not in item:
            print(f"skip (no name=): {item}", file=sys.stderr)
            continue
        name, path = item.split("=", 1)
        pred_path = Path(path)
        if not pred_path.exists():
            print(f"[skip] {name}: file not found {pred_path}", file=sys.stderr)
            continue
        pred = load_mask(pred_path)
        analyse_pred(name, gt, pred, image, slice_axis)


if __name__ == "__main__":
    main()
