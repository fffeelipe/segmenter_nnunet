#!/usr/bin/env python3
"""Prepare nnU-Net v2 plans for the T1 and T2 datasets, including TS-MRI pretrain plans.

Steps (for each of Dataset501_ALT_T1 and Dataset502_ALT_T2):

1. Run ``nnUNetv2_plan_and_preprocess -d <id> -c 3d_fullres 2d --verify_dataset_integrity``
   to create the default ``nnUNetPlans`` and preprocessed data for both configs.
   The 2d config will always be trained with these default plans.

2. Run ``nnUNetv2_move_plans_between_datasets -s 850 -t <id> -sp nnUNetPlans
   -tp nnUNetTSMRIPlans`` to clone TotalSegmentator-MRI's 3d_fullres plans
   (network topology + normalization + patch size) into our dataset.

3. Run ``nnUNetv2_preprocess -d <id> -plans_name nnUNetTSMRIPlans
   -c 3d_fullres`` to produce preprocessed data compatible with TS weights.

After this, ``run_training.sh`` can launch:
- 3d_fullres with ``-p nnUNetTSMRIPlans -pretrained_weights <ts_ckpt>``
- 2d with default ``-p nnUNetPlans`` (no pretraining; TS has no 2d config)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TARGET_DATASETS = [501, 502]
SOURCE_DATASET = 850
SOURCE_PLANS = "nnUNetPlans"
TARGET_PLANS = "nnUNetTSMRIPlans"

# Datasets whose input channel count differs from TS-MRI (1-ch). For these
# we only produce the default nnUNetPlans; move_plans_between_datasets
# from 850 would emit an unusable network.
MULTICHANNEL_DATASETS = {503}


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def require_env(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"env var {name} is not set")
    p = Path(v)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ts_plans_ready(nnunet_preprocessed: Path) -> bool:
    return (nnunet_preprocessed / "Dataset850_TotalSegMRI" / "nnUNetPlans.json").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip --verify_dataset_integrity (faster reruns).",
    )
    parser.add_argument(
        "--datasets",
        type=int,
        nargs="+",
        default=TARGET_DATASETS,
        help="Target dataset ids to plan/preprocess. Default: 501 502.",
    )
    parser.add_argument(
        "--t1-only",
        action="store_true",
        help="Shortcut: only plan/preprocess Dataset501_ALT_T1.",
    )
    parser.add_argument(
        "--t2-only",
        action="store_true",
        help="Shortcut: only plan/preprocess Dataset502_ALT_T2.",
    )
    parser.add_argument(
        "--t1t2-only",
        action="store_true",
        help="Shortcut: only plan/preprocess Dataset503_ALT_T1T2 (no TS transfer).",
    )
    args = parser.parse_args()
    mode_flags = [args.t1_only, args.t2_only, args.t1t2_only]
    if sum(bool(x) for x in mode_flags) > 1:
        raise SystemExit(
            "--t1-only, --t2-only and --t1t2-only are mutually exclusive"
        )

    nnunet_preprocessed = require_env("nnUNet_preprocessed")
    require_env("nnUNet_raw")
    require_env("nnUNet_results")

    if args.t1_only:
        target_datasets = [501]
    elif args.t2_only:
        target_datasets = [502]
    elif args.t1t2_only:
        target_datasets = [503]
    else:
        target_datasets = list(args.datasets)

    needs_ts = any(ds not in MULTICHANNEL_DATASETS for ds in target_datasets)
    if needs_ts and not ts_plans_ready(nnunet_preprocessed):
        raise SystemExit(
            "TS-MRI source plans not found. Run scripts/fetch_totalseg_mri.py first."
        )

    for ds_id in target_datasets:
        plan_cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(ds_id),
            "-c", "3d_fullres", "2d",
        ]
        if not args.skip_verify:
            plan_cmd.append("--verify_dataset_integrity")
        run(plan_cmd)

        if ds_id in MULTICHANNEL_DATASETS:
            print(
                f"[prepare] dataset {ds_id} has a non-TS channel layout; "
                f"skipping {TARGET_PLANS} transfer."
            )
            continue

        run([
            "nnUNetv2_move_plans_between_datasets",
            "-s", str(SOURCE_DATASET),
            "-t", str(ds_id),
            "-sp", SOURCE_PLANS,
            "-tp", TARGET_PLANS,
        ])

        run([
            "nnUNetv2_preprocess",
            "-d", str(ds_id),
            "-plans_name", TARGET_PLANS,
            "-c", "3d_fullres",
        ])

    print("\n[prepare] All plans prepared.")
    print("  2d config       -> plans: nnUNetPlans (default)")
    print(f"  3d_fullres cfg  -> plans: {TARGET_PLANS} (TotalSegmentator-MRI-compatible)")
    print(
        f"  multichannel (e.g. {sorted(MULTICHANNEL_DATASETS)})"
        " -> plans: nnUNetPlans for every config (no TS pretrain)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
