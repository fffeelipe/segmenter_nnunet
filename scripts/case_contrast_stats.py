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
import csv as _csv
import os
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
DEFAULT_DATASET = "Dataset501_ALT_T1"
DEFAULT_CASES = ["IOG28", "IOG35", "IOG38"]

# Resolved at parse time from --dataset / --images-dir / --labels-dir.
IMAGES_DIR: Path
LABELS_DIR: Path
# Set when --raw-modality is used (T1 / T2 per-patient folders at repo root).
RAW_MODALITY_DIR: Path | None = None


def _resolve_raw_modality_dir(modality: str) -> Path:
    """Resolve the per-patient raw folder (T1/ or T2/ at the repo root)."""
    if modality not in ("T1", "T2"):
        raise SystemExit(f"--raw-modality must be T1 or T2, got: {modality}")
    candidates = [
        REPO_ROOT / modality,
        REPO_ROOT.parent / modality,
        REPO_ROOT.parent.parent / modality,
        REPO_ROOT.parent.parent.parent / modality,
        REPO_ROOT.parent.parent.parent.parent / modality,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit(
        f"Could not locate raw {modality}/ folder. Tried: "
        f"{[str(c) for c in candidates]}"
    )


def _resolve_dataset_dirs(dataset: str) -> tuple[Path, Path]:
    """Resolve images/labels for ``dataset`` honoring $nnUNet_raw when set,
    else falling back to the repo-local ``nnunet_env/nnUNet_raw`` layout used
    by ``run_training.sh``."""
    raw_env = os.environ.get("nnUNet_raw")
    bases: list[Path] = []
    if raw_env:
        bases.append(Path(raw_env))
    bases.extend([
        REPO_ROOT / "nnunet_env" / "nnUNet_raw",
        REPO_ROOT / "nnunet_env_base" / "nnUNet_raw",
    ])
    for base in bases:
        cand = base / dataset
        if (cand / "imagesTr").is_dir() and (cand / "labelsTr").is_dir():
            return cand / "imagesTr", cand / "labelsTr"
    raise SystemExit(
        f"Could not resolve dataset {dataset} under any of: "
        f"{[str(b / dataset) for b in bases]}"
    )


def load_case(case: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    if RAW_MODALITY_DIR is not None:
        # Raw mode: T<N>/<case>/<case>_<seq>.nii.gz + <case>_<seq>_roi.nii.gz
        patient_dir = RAW_MODALITY_DIR / case
        if not patient_dir.is_dir():
            raise FileNotFoundError(f"missing patient dir: {patient_dir}")
        niis = sorted(patient_dir.glob("*.nii.gz"))
        img_path = lbl_path = None
        for p in niis:
            if p.name.endswith("_roi.nii.gz"):
                lbl_path = p
            else:
                img_path = p
        if img_path is None:
            raise FileNotFoundError(f"no image .nii.gz under {patient_dir}")
        if lbl_path is None:
            raise FileNotFoundError(f"no _roi.nii.gz under {patient_dir}")
    else:
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
        if RAW_MODALITY_DIR is not None:
            # Raw mode: the label often lives on a smaller ROI grid than the
            # image. Image-level intensity stats are still valid; tumor /
            # shell / brain stats are zeroed out via an empty mask.
            print(
                f"[warn] {case}: img/lbl shape mismatch in raw mode "
                f"({img.shape} vs {lbl.shape}); tumor stats will be n/a",
                file=sys.stderr,
            )
            lbl = np.zeros(img.shape, dtype=np.uint8)
        else:
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

    img_pcts = np.percentile(
        img, [0.1, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.5, 99.9]
    )
    if tumor_vox:
        tumor_p005, tumor_p050, tumor_p995 = np.percentile(
            tumor_vals, [0.5, 50.0, 99.5]
        )
    else:
        tumor_p005 = tumor_p050 = tumor_p995 = float("nan")

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
        "img_min": float(img.min()),
        "img_max": float(img.max()),
        "img_p001": float(img_pcts[0]),
        "img_p005": float(img_pcts[1]),
        "img_p010": float(img_pcts[2]),
        "img_p050": float(img_pcts[3]),
        "img_p500": float(img_pcts[4]),
        "img_p950": float(img_pcts[5]),
        "img_p990": float(img_pcts[6]),
        "img_p995": float(img_pcts[7]),
        "img_p999": float(img_pcts[8]),
        "img_p95": float(img_pcts[5]),  # back-compat with fmt_row
        "tumor_p005": float(tumor_p005),
        "tumor_p500": float(tumor_p050),
        "tumor_p995": float(tumor_p995),
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


CSV_FIELDS = [
    "case", "dataset",
    "spacing_z", "spacing_y", "spacing_x",
    "img_min", "img_max",
    "img_p001", "img_p005", "img_p010", "img_p050",
    "img_p500", "img_p950", "img_p990", "img_p995", "img_p999",
    "tumor_vox", "slices_with_tumor",
    "tumor_mean", "tumor_std",
    "tumor_p005", "tumor_p500", "tumor_p995",
    "shell_mean_bg", "shell_std_bg", "snr_shell",
    "brain_mean_bg", "brain_std_bg", "snr_brain",
    "raw_mean_bg",   "raw_std_bg",   "snr_raw",
]


def _row_to_csv(r: dict, dataset: str) -> dict:
    bg = {name: (mean_bg, std_bg, s) for name, mean_bg, std_bg, s in r["bg_rows"]}
    sp = r["spacing"]
    mean_t, std_t = r["tumor_stats"]

    def _bg(key, idx):
        if key in bg:
            return bg[key][idx]
        return float("nan")

    return {
        "case": r["case"],
        "dataset": dataset,
        "spacing_z": sp[0],
        "spacing_y": sp[1],
        "spacing_x": sp[2],
        "img_min": r["img_min"],
        "img_max": r["img_max"],
        "img_p001": r["img_p001"],
        "img_p005": r["img_p005"],
        "img_p010": r["img_p010"],
        "img_p050": r["img_p050"],
        "img_p500": r["img_p500"],
        "img_p950": r["img_p950"],
        "img_p990": r["img_p990"],
        "img_p995": r["img_p995"],
        "img_p999": r["img_p999"],
        "tumor_vox": r["tumor_vox"],
        "slices_with_tumor": r["slices_with_tumor"],
        "tumor_mean": mean_t,
        "tumor_std": std_t,
        "tumor_p005": r["tumor_p005"],
        "tumor_p500": r["tumor_p500"],
        "tumor_p995": r["tumor_p995"],
        "shell_mean_bg": _bg("shell", 0),
        "shell_std_bg": _bg("shell", 1),
        "snr_shell": _bg("shell", 2),
        "brain_mean_bg": _bg("brain", 0),
        "brain_std_bg": _bg("brain", 1),
        "snr_brain": _bg("brain", 2),
        "raw_mean_bg": _bg("raw", 0),
        "raw_std_bg": _bg("raw", 1),
        "snr_raw": _bg("raw", 2),
    }


def _list_all_cases() -> list[str]:
    if RAW_MODALITY_DIR is not None:
        cases = sorted(p.name for p in RAW_MODALITY_DIR.iterdir() if p.is_dir())
        if not cases:
            raise SystemExit(f"No patient dirs under {RAW_MODALITY_DIR}")
        return cases
    cases = sorted(p.name.replace(".nii.gz", "") for p in LABELS_DIR.glob("*.nii.gz"))
    if not cases:
        raise SystemExit(f"No labels under {LABELS_DIR}")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", default=[])
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset folder under $nnUNet_raw / nnunet_env/nnUNet_raw "
             "(default Dataset501_ALT_T1). Use Dataset502_ALT_T2 for the "
             "T2 audit.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Override imagesTr path (rare; --dataset usually suffices).",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Override labelsTr path (rare; --dataset usually suffices).",
    )
    parser.add_argument(
        "--raw-modality",
        choices=["T1", "T2"],
        default=None,
        help="Read directly from the raw repo-level T1/ or T2/ patient "
             "folders (image + *_roi.nii.gz). Bypasses convert_to_nnunet "
             "so you see PRE-_fix_intensity stats — exactly what's needed "
             "to decide T2 clipping percentiles.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Iterate every case under the dataset's labelsTr/ (overrides "
             "positional case ids).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Write a per-case CSV to this path (one row per case). When "
             "set, suppresses the human-readable text dump unless --verbose.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Always print the human-readable per-case dump (default when "
             "--csv is not set).",
    )
    args = parser.parse_args()

    global IMAGES_DIR, LABELS_DIR, RAW_MODALITY_DIR
    if args.raw_modality:
        RAW_MODALITY_DIR = _resolve_raw_modality_dir(args.raw_modality)
        print(f"[stats] raw_modality={args.raw_modality}", file=sys.stderr)
        print(f"[stats] root={RAW_MODALITY_DIR}", file=sys.stderr)
    elif args.images_dir or args.labels_dir:
        if not (args.images_dir and args.labels_dir):
            raise SystemExit("--images-dir and --labels-dir must be set together.")
        IMAGES_DIR = Path(args.images_dir)
        LABELS_DIR = Path(args.labels_dir)
        if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
            raise SystemExit(
                f"Bad --images-dir/--labels-dir: {IMAGES_DIR}, {LABELS_DIR}"
            )
        print(f"[stats] images={IMAGES_DIR}", file=sys.stderr)
        print(f"[stats] labels={LABELS_DIR}", file=sys.stderr)
    else:
        IMAGES_DIR, LABELS_DIR = _resolve_dataset_dirs(args.dataset)
        print(f"[stats] dataset={args.dataset}", file=sys.stderr)
        print(f"[stats] images={IMAGES_DIR}", file=sys.stderr)
        print(f"[stats] labels={LABELS_DIR}", file=sys.stderr)

    if args.all:
        cases = _list_all_cases()
    elif args.cases:
        cases = args.cases
    else:
        cases = DEFAULT_CASES

    results = []
    for case in cases:
        try:
            results.append(analyse(case))
        except FileNotFoundError as exc:
            print(f"[skip] {case}: {exc}", file=sys.stderr)

    if not results:
        sys.exit(1)

    show_text = args.verbose or args.csv is None
    if show_text:
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

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            ds_tag = (
                f"raw_{args.raw_modality}" if args.raw_modality else args.dataset
            )
            for r in results:
                w.writerow(_row_to_csv(r, ds_tag))
        print(f"[stats] wrote {len(results)} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
