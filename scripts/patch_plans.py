#!/usr/bin/env python3
"""Persist the minimal-plan overrides into ``nnUNetPlans.json`` on disk.

The custom trainer ``nnUNetTrainerALT_*`` flips ``batch_dice`` to ``False``
at runtime through ``self.configuration_manager`` so no file edit is
strictly needed. However persisting the override on disk:

1. Makes the behavior visible to tooling that inspects the plans file
   (e.g. ``nnUNetv2_find_best_configuration`` when computing the best
   ensemble).
2. Ensures any other trainer run with ``-p nnUNetPlans`` on this dataset
   also uses the corrected setting.

This script sets, for the target dataset's ``nnUNetPlans.json``:
- ``configurations.2d.batch_dice = False``

It does *not* touch ``nnUNetTSMRIPlans.json`` because the 3D config
already has ``batch_dice = False`` in the upstream TotalSegMRI plans.

Idempotent: re-running will overwrite the same keys with the same values.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def env_path(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"env var {name} is not set")
    return Path(v)


def resolve_dataset_name(nnunet_preproc: Path, dataset_id: int) -> str:
    prefix = f"Dataset{dataset_id:03d}_"
    matches = sorted(p.name for p in nnunet_preproc.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise SystemExit(f"No preprocessed folder starting with '{prefix}' under {nnunet_preproc}")
    if len(matches) > 1:
        raise SystemExit(f"Ambiguous preprocessed folders for id {dataset_id}: {matches}")
    return matches[0]


def patch_plans_file(plans_path: Path, overrides: dict) -> None:
    if not plans_path.is_file():
        raise SystemExit(f"Missing plans file: {plans_path}")
    plans = json.loads(plans_path.read_text())
    cfgs = plans.get("configurations", {})
    for cfg_name, cfg_overrides in overrides.items():
        if cfg_name not in cfgs:
            print(f"[patch] config '{cfg_name}' not in plans, skipping")
            continue
        for k, v in cfg_overrides.items():
            before = cfgs[cfg_name].get(k, "<unset>")
            cfgs[cfg_name][k] = v
            print(f"[patch] {plans_path.name} : configurations.{cfg_name}.{k}: {before} -> {v}")
    plans_path.write_text(json.dumps(plans, indent=2, sort_keys=False))
    print(f"[patch] wrote {plans_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", type=int, help="Dataset id (e.g. 501).")
    g.add_argument("--dataset-name", type=str, help="Full dataset folder name.")
    parser.add_argument(
        "--plans",
        type=str,
        default="nnUNetPlans",
        help="Plans identifier (filename without .json), default: nnUNetPlans.",
    )
    args = parser.parse_args()

    nnunet_preproc = env_path("nnUNet_preprocessed")

    if args.dataset_name:
        ds_name = args.dataset_name
    else:
        ds_name = resolve_dataset_name(nnunet_preproc, args.dataset)

    plans_path = nnunet_preproc / ds_name / f"{args.plans}.json"
    overrides = {
        "2d": {"batch_dice": False},
    }
    patch_plans_file(plans_path, overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
