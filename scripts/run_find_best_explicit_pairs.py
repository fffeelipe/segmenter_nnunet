#!/usr/bin/env python3
"""
Work around nnUNetv2_find_best_configuration Cartesian-product bug.

The CLI expands (-p plans) × (-c configs) and may request impossible pairs
like nnUNetTSMRIPlans__2d. This script calls the underlying API with an
explicit list of valid (plans, config) pairs, so ensembling + summary.json
are regenerated deterministically.

Example:
  export nnUNet_raw=/path/to/nnUNet_raw
  export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
  export nnUNet_results=/path/to/nnUNet_results
  python nnunet_env_T1T2/scripts/run_find_best_explicit_pairs.py \
    --dataset 501 --trainer nnUNetTrainerALT_os033_250epochs \
    --folds 0 1 2 3 4 \
    --pair nnUNetPlans 2d \
    --pair nnUNetTSMRIPlans 3d_fullres
"""

from __future__ import annotations

import argparse

from nnunetv2.evaluation.find_best_configuration import find_best_configuration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=int, required=True)
    p.add_argument("--trainer", type=str, required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("PLANS", "CONFIG"),
        required=True,
        help="Valid (plans, config) pair. Repeat for multiple.",
    )
    p.add_argument("--np", type=int, default=8, help="Evaluation/merge processes.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing merged/ensemble outputs.")
    p.add_argument(
        "--disable-ensembling",
        action="store_true",
        help="If set, do not build/evaluate crossval ensembles across pairs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    allowed_trained_models = [
        {"plans": plans, "configuration": config, "trainer": args.trainer}
        for plans, config in args.pair
    ]

    result = find_best_configuration(
        args.dataset,
        allowed_trained_models=allowed_trained_models,
        allow_ensembling=not args.disable_ensembling,
        num_processes=args.np,
        overwrite=args.overwrite,
        folds=tuple(args.folds),
        strict=False,
    )

    # Print the returned dict so it can be grepped/logged in terminals.
    # (The function already prints a human-readable summary + inference commands.)
    print("\n--- return_dict (machine readable) ---")
    print(result)


if __name__ == "__main__":
    main()

