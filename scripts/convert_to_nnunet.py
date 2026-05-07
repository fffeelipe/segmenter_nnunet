#!/usr/bin/env python3
"""Convert raw T1/T2 patient folders into nnU-Net v2 raw dataset layout.

Input layout (repository root):
    T1/IOGxx/IOGxx_<seq>.nii.gz           # image
    T1/IOGxx/IOGxx_<seq>_roi.nii.gz       # binary label
    T2/IOGxx/IOGxx_<seq>.nii.gz
    T2/IOGxx/IOGxx_<seq>_roi.nii.gz

Output layout (under nnUNet_raw):
    Dataset501_ALT_T1/
        imagesTr/IOGxx_0000.nii.gz
        labelsTr/IOGxx.nii.gz
        dataset.json
        holdout/images/IOGyy_0000.nii.gz   # optional: patients listed in --exclude-cases-file
        holdout/labels/IOGyy.nii.gz
    Dataset502_ALT_T2/
        imagesTr/IOGxx_0000.nii.gz
        labelsTr/IOGxx.nii.gz
        dataset.json
        holdout/...

Labels are re-saved as uint8 with values in {0, 1} (asserted).

Holdout: ``--exclude-cases-file`` lists patient folder names (e.g. IOG12), one per
line; ``#`` starts a comment. Those cases are written only under ``holdout/`` and
are omitted from ``imagesTr``/``labelsTr`` so nnU-Net never pretrains or trains
on them until you run inference separately.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _guess_physical_cores() -> int:
    """Best-effort estimate of physical core count (Linux), else fallback."""
    try:
        out = subprocess.check_output(["lscpu", "-p=Core,Socket"], text=True)
        cores = set()
        for line in out.splitlines():
            if not line or line.startswith("#"):
                continue
            core, sock = line.split(",")[:2]
            cores.add((core.strip(), sock.strip()))
        if cores:
            return int(len(cores))
    except Exception:
        pass
    return int(os.cpu_count() or 1)


def _default_workers() -> int:
    phys = max(1, _guess_physical_cores())
    return max(1, phys // 2)


DATASETS = {
    "T1": ("Dataset501_ALT_T1", "T1"),
    "T2": ("Dataset502_ALT_T2", "T2"),
}

# Clipped variants written when --percentile-clip is set. Distinct dataset ids
# so the un-clipped baseline preprocessed cache stays intact for A/B.
DATASETS_CLIP = {
    "T1": ("Dataset505_ALT_T1_clip", "T1"),
    "T2": ("Dataset506_ALT_T2_clip", "T2"),
}


def load_exclude_case_ids(path: Path) -> set[str]:
    """Load patient ids to hold out (one per line; ``#`` comments; blank lines skipped)."""
    text = path.read_text(encoding="utf-8")
    out: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def find_pair(patient_dir: Path) -> tuple[Path, Path] | None:
    """Return (image, label) paths inside a patient directory or None if missing."""
    niis = sorted(patient_dir.glob("*.nii.gz"))
    image = None
    label = None
    for p in niis:
        name = p.name
        if name.endswith("_roi.nii.gz"):
            label = p
        elif name.endswith(".nii.gz"):
            image = p
    if image is None or label is None:
        return None
    return image, label


GEOM_TOL = 1e-3
MIN_FOREGROUND_RETAINED = 0.2


def _geom_matches(a: sitk.Image, b: sitk.Image, tol: float = GEOM_TOL) -> tuple[bool, str]:
    """Check that two images share the same grid within a small tolerance."""
    if a.GetSize() != b.GetSize():
        return False, f"size {a.GetSize()} vs {b.GetSize()}"
    da, db = np.asarray(a.GetDirection()), np.asarray(b.GetDirection())
    if np.max(np.abs(da - db)) > tol:
        return False, f"direction max|Δ|={np.max(np.abs(da - db)):.2e}"
    oa, ob = np.asarray(a.GetOrigin()), np.asarray(b.GetOrigin())
    if np.max(np.abs(oa - ob)) > tol:
        return False, f"origin max|Δ|={np.max(np.abs(oa - ob)):.2e}"
    sa, sb = np.asarray(a.GetSpacing()), np.asarray(b.GetSpacing())
    if np.max(np.abs(sa - sb)) > tol:
        return False, f"spacing max|Δ|={np.max(np.abs(sa - sb)):.2e}"
    return True, "ok"


def _prebinarize(label_img: sitk.Image) -> sitk.Image:
    """Cast/round to uint8 {0, 1} while keeping the existing geometry."""
    arr = sitk.GetArrayFromImage(label_img)
    arr = np.rint(arr).astype(np.int32)
    arr[arr > 0] = 1
    arr[arr < 0] = 0
    if not set(np.unique(arr).tolist()).issubset({0, 1}):
        raise ValueError(
            f"Label has unexpected values after binarization: {np.unique(arr)}"
        )
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(label_img)
    return out


def _resample_label_to(reference_img: sitk.Image, label_img: sitk.Image) -> sitk.Image:
    """Resample ``label_img`` onto ``reference_img``'s grid with nearest-neighbor."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(label_img)


def align_label_to_image(
    label_img: sitk.Image, reference_img: sitk.Image, *, case_id: str = ""
) -> sitk.Image:
    """Produce a label that sits on the same grid as ``reference_img``.

    Strategy:
    1. Binarize the incoming label to uint8 {0, 1} in its own grid.
    2. If it already matches the reference grid within ``GEOM_TOL``, just
       copy the reference's exact header onto it (eliminates header drift
       that triggers nnU-Net's "direction mismatch" warning).
    3. Otherwise, resample onto the reference grid with nearest-neighbor.
       If the resample loses more than ``1 - MIN_FOREGROUND_RETAINED`` of
       the original foreground voxels, raise — that indicates a
       genuinely misaligned case that needs manual inspection.
    """
    lbl = _prebinarize(label_img)
    ok, reason = _geom_matches(lbl, reference_img)
    if ok:
        lbl.SetDirection(reference_img.GetDirection())
        lbl.SetOrigin(reference_img.GetOrigin())
        lbl.SetSpacing(reference_img.GetSpacing())
        return lbl

    orig_fg = int(np.count_nonzero(sitk.GetArrayFromImage(lbl)))
    resampled = _resample_label_to(reference_img, lbl)
    new_fg = int(np.count_nonzero(sitk.GetArrayFromImage(resampled)))

    tag = f"[{case_id}] " if case_id else ""
    print(
        f"  {tag}resampled label to image grid ({reason}); "
        f"foreground voxels {orig_fg} -> {new_fg}"
    )

    if orig_fg == 0:
        raise ValueError(f"{tag}label is empty (0 foreground voxels).")
    retained = new_fg / orig_fg
    if retained < MIN_FOREGROUND_RETAINED:
        raise ValueError(
            f"{tag}resampling retained only {retained:.0%} of label foreground "
            f"({orig_fg} -> {new_fg}). This likely means the label and image "
            f"are genuinely misaligned (not just a FOV crop). Mismatch was: {reason}."
        )
    return resampled


def _clip_percentiles(
    img: sitk.Image,
    *,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    case_id: str = "",
) -> sitk.Image:
    """Clip image intensities at the given percentiles.

    Designed to remove the upper-tail spikes / scanner uint12 ceiling
    artifacts surfaced by ``reports/{t1,t2}_intensity_audit.csv``: several
    cases hit ``max=4095`` (uint12 clamp), which inflates the apparent
    dynamic range and degrades z-score normalisation. Percentile clipping
    leaves the bulk of the distribution alone but caps the top/bottom
    fractions of voxels at their respective percentile values.

    Defaults (0.5 / 99.5) match the canonical nnU-Net CT clipping recipe
    and the ``ANALYSIS.md §3`` follow-up. Pass different values to A/B.
    """
    arr = sitk.GetArrayFromImage(img)
    lo, hi = float(np.percentile(arr, p_lo)), float(np.percentile(arr, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return img
    n_lo = int((arr < lo).sum())
    n_hi = int((arr > hi).sum())
    if n_lo == 0 and n_hi == 0:
        return img
    tag = f"[{case_id}] " if case_id else ""
    print(
        f"  {tag}percentile clip "
        f"(p{p_lo:g}={lo:.1f}, p{p_hi:g}={hi:.1f}): "
        f"low_clipped={n_lo}, high_clipped={n_hi}"
    )
    out_arr = np.clip(arr, lo, hi).astype(arr.dtype, copy=False)
    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(img)
    return out


def _fix_intensity(img: sitk.Image, case_id: str = "") -> sitk.Image:
    """Repair intensity-range artifacts from DICOM/NIfTI conversion.

    Two distinct failure modes observed on Dataset501_ALT_T1:

    1. uint16 values saved as int16 (IOG47): every voxel is negative
       (typical range [-32768, -29681]) because values >= 2**15 wrapped
       around to the negative int16 range. The whole image, including the
       tumor, sits below zero. Previously this case was passing through the
       pipeline because the raw file read as negatives carried enough
       *relative* contrast for the Z-Score to still separate tumor from
       background. A naive clip-to-zero destroys the case entirely
       (image becomes flat). Fix: add 2**16 so the range returns to
       unsigned; tumor contrast is preserved.
    2. Rare isolated negative voxels from bias-field correction or
       resampling padding: clip them at zero so they don't inflate the
       Z-Score std on an otherwise-positive image.

    If neither case applies the image is returned unchanged.
    """
    arr = sitk.GetArrayFromImage(img)
    lo, hi = float(arr.min()), float(arr.max())
    if lo >= 0:
        return img

    tag = f"[{case_id}] " if case_id else ""
    INT16_MIN = -32768.0

    if hi < 0 and lo >= INT16_MIN:
        arr32 = arr.astype(np.int32, copy=True) + 65536
        print(
            f"  {tag}uint16-as-int16 wraparound detected "
            f"(range [{lo:.0f}, {hi:.0f}] -> "
            f"[{int(arr32.min())}, {int(arr32.max())}])"
        )
        out_arr = arr32.astype(np.int32)
    else:
        neg = int((arr < 0).sum())
        print(
            f"  {tag}clipping {neg} negative voxels (min={lo:.1f}) -> 0"
        )
        out_arr = np.clip(arr, 0, None).astype(arr.dtype, copy=False)

    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(img)
    return out


def _convert_one_case(
    case_dir: str,
    modality_dir: str,
    images_dir: str,
    labels_dir: str,
    holdout_img: str,
    holdout_lbl: str,
    *,
    is_holdout: bool,
    clip_p_lo: float | None,
    clip_p_hi: float | None,
) -> dict:
    """Worker entrypoint: convert a single patient.

    Returns a dict with keys ``case_id``, ``status`` ("train" | "holdout" |
    "no_pair"), and ``image`` / ``label`` source filenames for logging.
    Errors during ``align_label_to_image`` are re-raised with the case id
    embedded so the parent's ``future.result()`` surfaces them clearly.
    """
    case_dir_p = Path(case_dir)
    case_id = case_dir_p.name
    pair = find_pair(case_dir_p)
    if pair is None:
        return {"case_id": case_id, "status": "no_pair"}

    image_src, label_src = pair
    img = sitk.ReadImage(str(image_src))
    img = _fix_intensity(img, case_id=f"{modality_dir}/{case_id}")
    if clip_p_lo is not None and clip_p_hi is not None:
        img = _clip_percentiles(
            img, p_lo=clip_p_lo, p_hi=clip_p_hi,
            case_id=f"{modality_dir}/{case_id}",
        )

    lbl = sitk.ReadImage(str(label_src))
    try:
        lbl_bin = align_label_to_image(
            lbl, reference_img=img, case_id=f"{modality_dir}/{case_id}"
        )
    except ValueError as exc:
        raise ValueError(f"[{modality_dir}] {case_id}: {exc}") from exc

    if is_holdout:
        Path(holdout_img).mkdir(parents=True, exist_ok=True)
        Path(holdout_lbl).mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(Path(holdout_img) / f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(lbl_bin, str(Path(holdout_lbl) / f"{case_id}.nii.gz"))
        return {
            "case_id": case_id, "status": "holdout",
            "image": image_src.name, "label": label_src.name,
        }
    else:
        sitk.WriteImage(img, str(Path(images_dir) / f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(lbl_bin, str(Path(labels_dir) / f"{case_id}.nii.gz"))
        return {
            "case_id": case_id, "status": "train",
            "image": image_src.name, "label": label_src.name,
        }


def convert_modality(
    src_root: Path,
    dst_root: Path,
    modality_dir: str,
    *,
    exclude_cases: set[str] | None = None,
    clip_p_lo: float | None = None,
    clip_p_hi: float | None = None,
    workers: int = 1,
) -> tuple[int, int]:
    """Convert one modality. Returns ``(n_train, n_holdout)``.

    When ``clip_p_lo`` / ``clip_p_hi`` are set, intensities are clipped at
    those percentiles after ``_fix_intensity`` and written under the
    ``DATASETS_CLIP`` dataset id (``Dataset505_*`` / ``Dataset506_*``) so
    the un-clipped baseline cache stays intact.

    When ``workers > 1`` patients are processed in a ``ProcessPoolExecutor``;
    each worker writes its own outputs atomically. Per-case stdout may
    interleave but is prefixed with ``[<modality>] <case>`` so it stays
    readable. Stays single-process when ``workers == 1`` to keep tracebacks
    direct for debugging.
    """
    use_clip = clip_p_lo is not None and clip_p_hi is not None
    if use_clip:
        ds_name, channel_name = DATASETS_CLIP[modality_dir]
    else:
        ds_name, channel_name = DATASETS[modality_dir]
    dst = dst_root / ds_name
    images_dir = dst / "imagesTr"
    labels_dir = dst / "labelsTr"
    holdout_img = dst / "holdout" / "images"
    holdout_lbl = dst / "holdout" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exclude_cases = exclude_cases or set()
    seen_ids: set[str] = set()

    src_modality = src_root / modality_dir
    if not src_modality.exists():
        raise FileNotFoundError(f"Missing source modality folder: {src_modality}")

    patients = sorted([p for p in src_modality.iterdir() if p.is_dir()])
    n_train = 0
    n_holdout = 0
    skipped: list[str] = []

    workers = max(1, int(workers))
    if workers > 1:
        print(f"[{modality_dir}] converting {len(patients)} patients with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _convert_one_case,
                    str(p),
                    modality_dir,
                    str(images_dir),
                    str(labels_dir),
                    str(holdout_img),
                    str(holdout_lbl),
                    is_holdout=p.name in exclude_cases,
                    clip_p_lo=clip_p_lo,
                    clip_p_hi=clip_p_hi,
                ): p.name
                for p in patients
            }
            for fut in as_completed(futs):
                case_id = futs[fut]
                res = fut.result()  # re-raises ValueError with case id embedded
                status = res["status"]
                if status == "no_pair":
                    skipped.append(case_id)
                    continue
                seen_ids.add(case_id)
                if status == "holdout":
                    n_holdout += 1
                    print(
                        f"  [{modality_dir}] {case_id}: HOLDOUT -> holdout/ "
                        f"image={res['image']} label={res['label']}"
                    )
                else:
                    n_train += 1
                    print(f"  [{modality_dir}] {case_id}: image={res['image']} label={res['label']}")
    else:
        for p in patients:
            res = _convert_one_case(
                str(p),
                modality_dir,
                str(images_dir),
                str(labels_dir),
                str(holdout_img),
                str(holdout_lbl),
                is_holdout=p.name in exclude_cases,
                clip_p_lo=clip_p_lo,
                clip_p_hi=clip_p_hi,
            )
            status = res["status"]
            if status == "no_pair":
                skipped.append(p.name)
                continue
            seen_ids.add(p.name)
            if status == "holdout":
                n_holdout += 1
                print(
                    f"  [{modality_dir}] {p.name}: HOLDOUT -> holdout/ "
                    f"image={res['image']} label={res['label']}"
                )
            else:
                n_train += 1
                print(f"  [{modality_dir}] {p.name}: image={res['image']} label={res['label']}")

    unknown = sorted(exclude_cases - seen_ids)
    if unknown:
        print(
            f"[{modality_dir}] WARNING: exclude list mentions unknown case ids "
            f"(not in T1/T2 folders or missing pairs): {unknown}"
        )

    dataset_json = {
        "channel_names": {"0": channel_name},
        "labels": {"background": 0, "ALT": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": ds_name,
        "description": "Atypical lipomatous tumor (ALT) MRI segmentation.",
    }
    with open(dst / "dataset.json", "w") as fh:
        json.dump(dataset_json, fh, indent=2)

    print(f"[{modality_dir}] wrote {n_train} train + {n_holdout} holdout under {dst}")
    if skipped:
        print(f"[{modality_dir}] skipped (no valid pair): {skipped}")
    return n_train, n_holdout


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
        "--modalities",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Which modality folders to convert. Default: all ('T1' 'T2').",
    )
    parser.add_argument(
        "--t1-only",
        action="store_true",
        help="Shortcut for --modalities T1 (ignores --modalities if set).",
    )
    parser.add_argument(
        "--t2-only",
        action="store_true",
        help="Shortcut for --modalities T2 (ignores --modalities if set).",
    )
    parser.add_argument(
        "--exclude-cases-file",
        type=Path,
        default=None,
        help=(
            "Text file of patient folder names (IOGxx) to write only under "
            "dataset/holdout/{images,labels}/, excluded from imagesTr/labelsTr."
        ),
    )
    parser.add_argument(
        "--percentile-clip",
        action="store_true",
        help=(
            "Apply percentile intensity clipping (p_lo=0.5, p_hi=99.5 by "
            "default) after _fix_intensity. Writes to Dataset505_ALT_T1_clip "
            "/ Dataset506_ALT_T2_clip so the un-clipped baseline cache "
            "(Dataset501/502) stays intact for A/B."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help=(
            "Number of parallel worker processes for per-patient conversion "
            "(default: auto ~ half physical cores). Set to 1 to keep the "
            "single-process loop (clearer tracebacks)."
        ),
    )
    args = parser.parse_args()
    if args.t1_only and args.t2_only:
        raise SystemExit("--t1-only and --t2-only are mutually exclusive")

    dst = args.dst
    if dst is None:
        env_raw = os.environ.get("nnUNet_raw")
        if not env_raw:
            raise SystemExit(
                "nnUNet_raw env var is not set and --dst was not provided."
            )
        dst = Path(env_raw)
    dst.mkdir(parents=True, exist_ok=True)

    if args.t1_only:
        modalities = ["T1"]
    elif args.t2_only:
        modalities = ["T2"]
    else:
        modalities = list(args.modalities)
    exclude: set[str] = set()
    if args.exclude_cases_file is not None:
        if not args.exclude_cases_file.is_file():
            raise SystemExit(f"--exclude-cases-file not found: {args.exclude_cases_file}")
        exclude = load_exclude_case_ids(args.exclude_cases_file)
    total_train = 0
    total_holdout = 0
    clip_kwargs = (
        {"clip_p_lo": args.clip_p_lo, "clip_p_hi": args.clip_p_hi}
        if args.percentile_clip
        else {}
    )
    for modality in modalities:
        tr, ho = convert_modality(
            args.src, dst, modality,
            exclude_cases=exclude,
            workers=args.workers,
            **clip_kwargs,
        )
        total_train += tr
        total_holdout += ho
    print(
        f"Done. Train cases={total_train}, holdout cases={total_holdout} "
        f"across {'+'.join(modalities)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
