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
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


DATASETS = {
    "T1": ("Dataset501_ALT_T1", "T1"),
    "T2": ("Dataset502_ALT_T2", "T2"),
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


def convert_modality(
    src_root: Path,
    dst_root: Path,
    modality_dir: str,
    *,
    exclude_cases: set[str] | None = None,
) -> tuple[int, int]:
    """Convert one modality. Returns ``(n_train, n_holdout)``."""
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

    for p in patients:
        pair = find_pair(p)
        if pair is None:
            skipped.append(p.name)
            continue
        image_src, label_src = pair
        case_id = p.name
        seen_ids.add(case_id)

        img = sitk.ReadImage(str(image_src))
        img = _fix_intensity(img, case_id=f"{modality_dir}/{case_id}")

        lbl = sitk.ReadImage(str(label_src))
        try:
            lbl_bin = align_label_to_image(
                lbl, reference_img=img, case_id=f"{modality_dir}/{case_id}"
            )
        except ValueError as exc:
            raise ValueError(f"[{modality_dir}] {case_id}: {exc}") from exc

        if case_id in exclude_cases:
            holdout_img.mkdir(parents=True, exist_ok=True)
            holdout_lbl.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img, str(holdout_img / f"{case_id}_0000.nii.gz"))
            sitk.WriteImage(lbl_bin, str(holdout_lbl / f"{case_id}.nii.gz"))
            n_holdout += 1
            print(
                f"  [{modality_dir}] {case_id}: HOLDOUT -> holdout/ "
                f"image={image_src.name} label={label_src.name}"
            )
        else:
            sitk.WriteImage(img, str(images_dir / f"{case_id}_0000.nii.gz"))
            sitk.WriteImage(lbl_bin, str(labels_dir / f"{case_id}.nii.gz"))
            n_train += 1
            print(f"  [{modality_dir}] {case_id}: image={image_src.name} label={label_src.name}")

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
    for modality in modalities:
        tr, ho = convert_modality(args.src, dst, modality, exclude_cases=exclude)
        total_train += tr
        total_holdout += ho
    print(
        f"Done. Train cases={total_train}, holdout cases={total_holdout} "
        f"across {'+'.join(modalities)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
