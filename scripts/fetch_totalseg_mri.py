#!/usr/bin/env python3
"""Download TotalSegmentator MRI pretrained weights and stage them for nnU-Net v2.

What it does:
- Downloads a TotalSegmentator MRI release ZIP from GitHub (no license required
  for open-source tasks 850/851/852/853/756/597/598).
- Extracts into ``$nnUNet_results/Dataset850_TotalSegMRI/`` (our local source
  dataset ID used for ``move_plans_between_datasets``).
- Copies ``plans.json`` to ``$nnUNet_preprocessed/Dataset850_TotalSegMRI/nnUNetPlans.json``
  so ``nnUNetv2_move_plans_between_datasets -s 850 -sp nnUNetPlans`` works.
- Copies ``dataset.json`` to ``$nnUNet_raw/Dataset850_TotalSegMRI/dataset.json``
  (+ empty ``imagesTr``/``labelsTr``) so the raw folder exists.
- Writes ``$nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_manifest.json``
  recording per-fold checkpoint paths for ``3d_fullres`` so ``run_training.sh``
  can pass ``-pretrained_weights`` to ``nnUNetv2_train``.

Default task is ``852`` (``Dataset852_TotalSegMRI_total_3mm_1088subj``) because
it is a generalist full-body MRI model (widest coverage for ALT cases
in limbs/trunk). Override via ``--task``.

Notes:
- TotalSegmentator only ships the ``3d_fullres`` configuration. Therefore we
  only pretrain the 3d_fullres config of our target datasets; 2d is trained
  from scratch.
- Some TS zips extract to ``<dsname>/`` at the root and some to a single
  ``nnUNetTrainer__nnUNetPlans__3d_fullres/`` directly. The script handles
  both by searching for ``plans.json`` recursively.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen


TASK_URLS = {
    850: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset850_TotalSegMRI_part1_organs_1088subj.zip",
    851: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset851_TotalSegMRI_part2_muscles_1088subj.zip",
    852: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset852_TotalSegMRI_total_3mm_1088subj.zip",
    853: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset853_TotalSegMRI_total_6mm_1088subj.zip",
    597: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset597_mri_body_139subj.zip",
    756: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset756_mri_vertebrae_1076subj.zip",
}

LOCAL_DATASET_NAME = "Dataset850_TotalSegMRI"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[fetch] already downloaded: {dest}")
        return
    print(f"[fetch] downloading {url} -> {dest}")
    with urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    print(f"[fetch] done ({dest.stat().st_size / 1e6:.1f} MB)")


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] extracting {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    return extract_to


def find_trainer_dir(root: Path) -> Path:
    """Find the ``nnUNetTrainer*__nnUNetPlans__3d_fullres`` folder inside root."""
    candidates = [p for p in root.rglob("*__3d_fullres") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a 3d_fullres trainer folder under {root}"
        )
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def env_path(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"env var {name} is not set")
    p = Path(v)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        type=int,
        default=852,
        choices=sorted(TASK_URLS.keys()),
        help="TotalSegmentator MRI task id to use as pretraining source (default: 852).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".cache" / "totalseg_mri",
        help="Where to cache the downloaded zip + extracted folder.",
    )
    args = parser.parse_args()

    nnunet_raw = env_path("nnUNet_raw")
    nnunet_preprocessed = env_path("nnUNet_preprocessed")
    nnunet_results = env_path("nnUNet_results")

    url = TASK_URLS[args.task]
    zip_name = url.rsplit("/", 1)[-1]
    zip_path = args.cache / zip_name
    extract_dir = args.cache / zip_name.replace(".zip", "")

    download(url, zip_path)
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_zip(zip_path, extract_dir)
    else:
        print(f"[fetch] using existing extraction: {extract_dir}")

    trainer_dir = find_trainer_dir(extract_dir)
    print(f"[fetch] source trainer dir: {trainer_dir}")

    plans_json = trainer_dir / "plans.json"
    dataset_json = trainer_dir / "dataset.json"
    if not plans_json.is_file() or not dataset_json.is_file():
        raise FileNotFoundError(
            f"Missing plans.json or dataset.json under {trainer_dir}"
        )

    local_preproc = nnunet_preprocessed / LOCAL_DATASET_NAME
    local_preproc.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plans_json, local_preproc / "nnUNetPlans.json")
    shutil.copy2(dataset_json, local_preproc / "dataset.json")
    print(f"[fetch] wrote {local_preproc}/nnUNetPlans.json")

    local_raw = nnunet_raw / LOCAL_DATASET_NAME
    (local_raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    (local_raw / "labelsTr").mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_json, local_raw / "dataset.json")
    print(f"[fetch] wrote {local_raw}/dataset.json")

    local_results = nnunet_results / LOCAL_DATASET_NAME / trainer_dir.name
    local_results.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plans_json, local_results / "plans.json")
    shutil.copy2(dataset_json, local_results / "dataset.json")

    fold_dirs = sorted([p for p in trainer_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* folders found in {trainer_dir}")

    checkpoints: dict[str, str] = {}
    for fd in fold_dirs:
        ckpt = fd / "checkpoint_final.pth"
        if not ckpt.is_file():
            ckpt = fd / "checkpoint_best.pth"
        if not ckpt.is_file():
            print(f"[fetch] warning: no checkpoint in {fd}")
            continue
        dst_fold = local_results / fd.name
        dst_fold.mkdir(parents=True, exist_ok=True)
        dst_ckpt = dst_fold / ckpt.name
        if not dst_ckpt.exists():
            shutil.copy2(ckpt, dst_ckpt)
        checkpoints[fd.name] = str(dst_ckpt.resolve())

    if not checkpoints:
        raise RuntimeError("No checkpoints were staged; aborting.")

    default_ckpt = checkpoints.get("fold_0") or next(iter(checkpoints.values()))

    manifest = {
        "source_task_id": args.task,
        "source_dataset_name": LOCAL_DATASET_NAME,
        "source_plans_identifier": "nnUNetPlans",
        "config": "3d_fullres",
        "checkpoints_by_fold": checkpoints,
        "default_checkpoint": default_ckpt,
    }
    manifest_path = local_preproc / "pretrain_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[fetch] wrote manifest: {manifest_path}")
    print(f"[fetch] default pretrained checkpoint: {default_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
