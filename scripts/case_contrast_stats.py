"""Compute contrast / SNR statistics for ALT cases.

Reproduces the ``Contraste`` column of ANALYSIS.md §2 using the formula

    SNR = (mean_tumor - mean_bg) / std_bg

Three background definitions are computed so we can compare:

* ``raw``   — all voxels outside the tumor (includes air). Dominant by
  zeros; SNRs close to tumor_mean / std(image). Useful as a floor.
* ``brain`` — voxels with intensity ``> brain_thr * image.max()`` AND
  outside the tumor. Excludes air. This matches the definition that
  yielded IOG38 ≈ −0.06 in the historical table (§2).
* ``shell`` — dilate the tumor mask by ``shell_mm`` (default 5 mm) and
  take the voxels inside the dilated mask minus the tumor. That is the
  *local* background against which the tumor has to be discriminated.
  Usually the most informative of the three.

Also prints tumor volume, number of slices with tumor, and both tumor
and background intensity percentiles to characterise each case.

Usage
-----
    python scripts/case_contrast_stats.py IOG28 IOG38 IOG35

With no case arguments, runs on the default watchlist (IOG28, IOG35,
IOG38 as of Exp. 2b').
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


REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "nnunet_env" / "nnUNet_raw" / "Dataset501_ALT_T1" / "imagesTr"
LABELS_DIR = REPO_ROOT / "nnunet_env" / "nnUNet_raw" / "Dataset501_ALT_T1" / "labelsTr"

DEFAULT_CASES = ["IOG28", "IOG35", "IOG38"]


def load_case(case: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    img_path = IMAGES_DIR / f"{case}_0000.nii.gz"
    lbl_path = LABELS_DIR / f"{case}.nii.gz"
    if not img_path.exists():
        raise FileNotFoundError(f"missing image: {img_path}")
    if not lbl_path.exists():
        raise FileNotFoundError(f"missing label: {lbl_path}")

    img_nii = nib.load(str(img_path))
    lbl_nii = nib.load(str(lbl_path))
    img = np.asarray(img_nii.dataobj, dtype=np.float32)
    lbl = np.asarray(lbl_nii.dataobj, dtype=np.uint8)

    if img.shape != lbl.shape:
        raise ValueError(
            f"shape mismatch for {case}: img {img.shape} vs lbl {lbl.shape}"
        )
    spacing = tuple(float(s) for s in img_nii.header.get_zooms()[:3])
    return img, lbl, spacing


def snr(mean_tumor: float, mean_bg: float, std_bg: float) -> float:
    if std_bg <= 1e-6:
        return float("nan")
    return (mean_tumor - mean_bg) / std_bg


def dilate_mm(mask: np.ndarray, spacing: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    """Binary dilation by ``radius_mm`` mm using a ball structuring element.

    Falls back to a separable per-axis dilation when SciPy is unavailable.
    """
    if HAS_SCIPY:
        radius_vox = np.array([radius_mm / max(s, 1e-3) for s in spacing])
        shape = tuple(int(np.ceil(r)) * 2 + 1 for r in radius_vox)
        centre = tuple((s - 1) // 2 for s in shape)
        zz, yy, xx = np.meshgrid(
            *(np.arange(s) - c for s, c in zip(shape, centre)), indexing="ij"
        )
        dist = np.sqrt((zz / max(radius_vox[0], 1e-3)) ** 2
                       + (yy / max(radius_vox[1], 1e-3)) ** 2
                       + (xx / max(radius_vox[2], 1e-3)) ** 2)
        struct = dist <= 1.0
        return ndi.binary_dilation(mask, structure=struct)

    out = mask.copy()
    for axis, sp in enumerate(spacing):
        n = max(1, int(np.round(radius_mm / max(sp, 1e-3))))
        for _ in range(n):
            shifted_pos = np.roll(out, 1, axis=axis)
            shifted_neg = np.roll(out, -1, axis=axis)
            out = out | shifted_pos | shifted_neg
    return out


def bg_masks(img: np.ndarray, tumor: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, np.ndarray]:
    not_tumor = ~tumor
    raw = not_tumor
    # brain: exclude ~air using a simple threshold on image.max().
    brain_thr = 0.02 * img.max()
    brain = (img > brain_thr) & not_tumor
    # shell: 5 mm around tumor (reasonable local neighbourhood).
    shell = dilate_mm(tumor, spacing, radius_mm=5.0) & not_tumor
    return {"raw": raw, "brain": brain, "shell": shell}


def percentile_strip(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return f"{p5:7.1f} / {p50:7.1f} / {p95:7.1f}"


def analyse(case: str) -> dict:
    img, lbl, spacing = load_case(case)
    tumor = lbl > 0
    tumor_vox = int(tumor.sum())
    slices_with_tumor = int((tumor.sum(axis=(1, 2)) > 0).sum()) if tumor.ndim == 3 else 0

    tumor_vals = img[tumor]
    mean_t = float(tumor_vals.mean()) if tumor_vox else float("nan")
    std_t = float(tumor_vals.std()) if tumor_vox else float("nan")

    masks = bg_masks(img, tumor, spacing)
    rows = []
    for name, mask in masks.items():
        bg_vals = img[mask]
        if bg_vals.size == 0:
            rows.append((name, float("nan"), float("nan"), float("nan")))
            continue
        mean_bg = float(bg_vals.mean())
        std_bg = float(bg_vals.std())
        rows.append((name, mean_bg, std_bg, snr(mean_t, mean_bg, std_bg)))

    return {
        "case": case,
        "spacing": spacing,
        "tumor_vox": tumor_vox,
        "slices_with_tumor": slices_with_tumor,
        "tumor_stats": (mean_t, std_t),
        "bg_rows": rows,
        "tumor_pct": percentile_strip(tumor_vals),
        "shell_pct": percentile_strip(img[masks["shell"]]),
        "brain_pct": percentile_strip(img[masks["brain"]]),
        "img_max": float(img.max()),
        "img_p95": float(np.percentile(img, 95)),
    }


def fmt_row(r: dict) -> str:
    sp = r["spacing"]
    sp_str = f"{sp[0]:.2f}x{sp[1]:.2f}x{sp[2]:.2f}"
    mt, st = r["tumor_stats"]
    lines = [
        f"=== {r['case']}  vox={r['tumor_vox']:>7}  slices={r['slices_with_tumor']:>3}"
        f"  spacing={sp_str}  img.max={r['img_max']:.0f} img.p95={r['img_p95']:.0f}",
        f"    tumor intensity   mean={mt:7.1f} std={st:6.1f}  p5/50/95 = {r['tumor_pct']}",
        f"    shell (5mm)  p5/50/95 = {r['shell_pct']}",
        f"    brain        p5/50/95 = {r['brain_pct']}",
        f"    {'bg_def':<6} {'mean_bg':>8}  {'std_bg':>7}  {'SNR':>7}",
    ]
    for name, mean_bg, std_bg, s in r["bg_rows"]:
        tag = "  <-- inverted" if s < 0 else ""
        lines.append(f"    {name:<6} {mean_bg:8.1f}  {std_bg:7.1f}  {s:7.2f}{tag}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", default=DEFAULT_CASES)
    args = parser.parse_args()

    results = []
    for case in args.cases:
        try:
            results.append(analyse(case))
        except FileNotFoundError as exc:
            print(f"[skip] {case}: {exc}", file=sys.stderr)

    if not results:
        sys.exit(1)

    for r in results:
        print(fmt_row(r))
        print()

    print("Summary (SNR with shell background, the most informative):")
    print(f"    {'case':<8}  {'tumor_vox':>9}  {'SNR_shell':>10}  {'SNR_brain':>10}  {'SNR_raw':>8}")
    for r in results:
        bg = {name: s for name, _, _, s in r["bg_rows"]}
        print(
            f"    {r['case']:<8}  {r['tumor_vox']:>9}  "
            f"{bg.get('shell', float('nan')):>10.2f}  "
            f"{bg.get('brain', float('nan')):>10.2f}  "
            f"{bg.get('raw', float('nan')):>8.2f}"
        )


if __name__ == "__main__":
    main()
