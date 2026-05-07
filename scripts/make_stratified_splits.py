#!/usr/bin/env python3
"""Write a volume-stratified 5-fold ``splits_final.json`` for a given nnU-Net dataset.

Why:
  Default nnU-Net splits are random. For our ALT dataset the tumor volume
  ranges from 405 to 267 190 voxels (factor 660x). Random splits cause one
  fold to receive several tiny / low-contrast lesions together and collapse
  (see Dataset501_ALT_T1 fold 2 with IOG1 + IOG38 + IOG40 and fold 4 with
  IOG7 + IOG9 + IOG10). Stratifying the folds by tumor volume keeps each
  fold balanced and reduces variance across folds.

How:
  - Reads every label under ``$nnUNet_raw/<dataset>/labelsTr/`` to compute
    the tumor voxel count per case.
  - Groups ``<patient>_T1`` / ``<patient>_T2`` (direction-mismatch derived
    rows from ``build_t1t2_dataset.py``) so **both always share the same
    validation fold**; stratification volume for the pair is ``max(vol_T1,
    vol_T2)``. Ungrouped case ids (e.g. ``IOG12``) are unchanged.
  - Sorts those groups by that volume, then assigns consecutive groups to
    folds round-robin so each fold keeps a similar size distribution.
  - Writes ``$nnUNet_preprocessed/<dataset>/splits_final.json`` with the
    exact schema nnU-Net expects (list of ``{"train": [...], "val": [...]}``).

Usage:
  python scripts/make_stratified_splits.py --dataset 501
  python scripts/make_stratified_splits.py --dataset-name Dataset501_ALT_T1 --n-folds 5
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


def env_path(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"env var {name} is not set")
    return Path(v)


def resolve_dataset_name(nnunet_raw: Path, dataset_id: int) -> str:
    prefix = f"Dataset{dataset_id:03d}_"
    matches = sorted(p.name for p in nnunet_raw.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise SystemExit(
            f"No dataset folder starting with '{prefix}' under {nnunet_raw}."
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Ambiguous: {len(matches)} folders match '{prefix}*' under {nnunet_raw}: {matches}"
        )
    return matches[0]


def tumor_voxels(label_file: Path) -> int:
    arr = nib.load(str(label_file)).get_fdata()
    return int((arr > 0).sum())


# Dataset503 direction-mismatch: paired training rows share one patient.
_DERIVED_T1T2 = re.compile(r"^(.+)_T([12])$")


def group_case_volumes(
    case_volumes: list[tuple[str, int]],
) -> list[tuple[str, list[str], int]]:
    """Merge *_T1 / *_T2 into one stratification unit (same fold always)."""
    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for case_id, vol in case_volumes:
        m = _DERIVED_T1T2.match(case_id)
        key = m.group(1) if m else case_id
        buckets[key].append((case_id, vol))
    groups: list[tuple[str, list[str], int]] = []
    for key, members in buckets.items():
        cids = sorted(cid for cid, _ in members)
        vmax = max(v for _, v in members)
        groups.append((key, cids, vmax))
    return groups


def stratified_folds_from_groups(
    groups: list[tuple[str, list[str], int]], n_folds: int
) -> list[list[str]]:
    """Round-robin assignment of volume-sorted *groups* to folds.

    Every case id in a group lands in the same validation fold.
    """
    sorted_groups = sorted(groups, key=lambda g: g[2])
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, (_key, case_ids, _vol) in enumerate(sorted_groups):
        folds[i % n_folds].extend(case_ids)
    return folds


def _load_contrast_csv(path: Path) -> dict[str, float]:
    """Load patient-key -> contrast value from a CSV with columns
    ``case,contrast`` (extra columns ignored). Empty / non-numeric rows are
    skipped silently; the caller falls back to volume-only stratification
    when not enough patients have a value.
    """
    out: dict[str, float] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if "case" not in reader.fieldnames or "contrast" not in reader.fieldnames:
            raise SystemExit(
                f"contrast CSV must have 'case' and 'contrast' columns, got: {reader.fieldnames}"
            )
        for row in reader:
            cid = (row.get("case") or "").strip()
            raw = (row.get("contrast") or "").strip()
            if not cid or not raw:
                continue
            try:
                out[cid] = float(raw)
            except ValueError:
                continue
    return out


def pick_stratified_holdout(
    groups: list[tuple[str, list[str], int]],
    *,
    n: int = 7,
    contrast_by_key: dict[str, float] | None = None,
    seed: int = 20260507,
) -> tuple[list[str], dict]:
    """Pick ``n`` patient keys stratified by volume quartile (and contrast
    bin when ``contrast_by_key`` is provided).

    Strategy:
      * Sort groups by ``vol_max`` and split into 4 quartile bins.
      * If contrast info covers all keys, split each quartile into
        low/high vs the global median contrast (8 cells total). Otherwise
        use volume-only (4 cells).
      * Round-robin pick one random group from each non-empty cell
        (seeded), repeating cells until ``n`` picks are collected.

    Returns ``(picked_keys, recipe_meta)``. ``picked_keys`` is the list of
    patient keys (e.g. ``IOG41``); ``recipe_meta`` documents the cells and
    their assignments for the file header.
    """
    if n <= 0:
        return [], {}
    rng = random.Random(seed)
    by_vol = sorted(groups, key=lambda g: g[2])
    if not by_vol:
        return [], {}
    n_groups = len(by_vol)
    vol_quartile_edges = [
        by_vol[max(0, n_groups * q // 4 - 1)][2] for q in range(1, 5)
    ]

    def vol_bin(v: int) -> int:
        for i, edge in enumerate(vol_quartile_edges):
            if v <= edge:
                return i
        return 3

    has_contrast = (
        contrast_by_key is not None
        and sum(1 for k, _, _ in by_vol if k in contrast_by_key) >= n_groups - 2
    )
    if has_contrast:
        contrasts = sorted(contrast_by_key[k] for k, _, _ in by_vol if k in contrast_by_key)
        cmedian = contrasts[len(contrasts) // 2]
    else:
        cmedian = None

    cells: dict[tuple[int, int], list[tuple[str, list[str], int]]] = defaultdict(list)
    for g in by_vol:
        key = g[0]
        vb = vol_bin(g[2])
        if cmedian is not None and key in contrast_by_key:
            cb = 0 if contrast_by_key[key] <= cmedian else 1
        else:
            cb = 0
        cells[(vb, cb)].append(g)

    cell_order = sorted(cells.keys())
    picked: list[str] = []
    picked_set: set[str] = set()
    pool: dict[tuple[int, int], list[tuple[str, list[str], int]]] = {
        c: list(rng.sample(cells[c], len(cells[c]))) for c in cell_order
    }
    while len(picked) < n:
        progress = False
        for c in cell_order:
            if not pool[c]:
                continue
            g = pool[c].pop()
            if g[0] in picked_set:
                continue
            picked.append(g[0])
            picked_set.add(g[0])
            progress = True
            if len(picked) >= n:
                break
        if not progress:
            break

    recipe = {
        "n_requested": n,
        "n_groups_total": n_groups,
        "vol_quartile_edges": vol_quartile_edges,
        "contrast_median": cmedian,
        "stratified_by": "vol_quartile x contrast_bin" if has_contrast else "vol_quartile",
        "seed": seed,
        "picked": picked,
    }
    return picked, recipe


def build_splits(folds: list[list[str]]) -> list[dict]:
    """Convert per-fold validation lists into nnU-Net's 5-fold split schema."""
    all_cases = sorted({c for f in folds for c in f})
    splits = []
    for vi, val_cases in enumerate(folds):
        val_set = set(val_cases)
        train = [c for c in all_cases if c not in val_set]
        splits.append({"train": sorted(train), "val": sorted(val_cases)})
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--dataset", type=int, help="Dataset id (e.g. 501).")
    g.add_argument(
        "--dataset-name",
        type=str,
        help="Full dataset folder name (e.g. Dataset501_ALT_T1).",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the folds but do not write splits_final.json.",
    )
    parser.add_argument(
        "--write-holdout",
        type=str,
        default=None,
        help="Path to write a stratified holdout list (one patient key per line). "
             "When set, the script picks --n holdout cases instead of writing "
             "splits_final.json.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=7,
        help="Holdout size for --write-holdout (default 7).",
    )
    parser.add_argument(
        "--contrast-csv",
        type=str,
        default=None,
        help="Optional CSV (columns: case,contrast) used to refine the "
             "stratification with a low/high contrast bin per volume "
             "quartile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260507,
        help="Seed for --write-holdout (default 20260507 = 2026-05-07).",
    )
    args = parser.parse_args()

    nnunet_raw = env_path("nnUNet_raw")
    nnunet_preproc = env_path("nnUNet_preprocessed")

    if args.dataset_name:
        ds_name = args.dataset_name
    else:
        if args.dataset is None:
            raise SystemExit("Provide --dataset or --dataset-name.")
        ds_name = resolve_dataset_name(nnunet_raw, args.dataset)

    labels_dir = nnunet_raw / ds_name / "labelsTr"
    if not labels_dir.is_dir():
        raise SystemExit(f"Missing labelsTr folder: {labels_dir}")

    label_files = sorted(labels_dir.glob("*.nii.gz"))
    if not label_files:
        raise SystemExit(f"No .nii.gz files under {labels_dir}")

    print(f"[splits] computing tumor volumes for {len(label_files)} cases in {ds_name}")
    case_volumes: list[tuple[str, int]] = []
    for lf in label_files:
        case_id = lf.name.replace(".nii.gz", "")
        vol = tumor_voxels(lf)
        case_volumes.append((case_id, vol))

    vol_by_case = dict(case_volumes)
    groups = group_case_volumes(case_volumes)
    n_paired = sum(1 for _k, cids, _v in groups if len(cids) > 1)
    if n_paired:
        print(
            f"[splits] grouped {n_paired} patient(s) with _T1/_T2 pairs "
            f"({len(case_volumes)} files -> {len(groups)} stratification units)"
        )

    strat_volumes = np.array([g[2] for g in groups])
    print(
        f"[splits] per-unit tumor volume stats: min={strat_volumes.min()}, "
        f"median={int(np.median(strat_volumes))}, max={strat_volumes.max()}, "
        f"factor={strat_volumes.max()/max(strat_volumes.min(),1):.1f}x"
    )

    if args.write_holdout:
        contrast_by_key: dict[str, float] | None = None
        if args.contrast_csv:
            cpath = Path(args.contrast_csv)
            if not cpath.is_file():
                raise SystemExit(f"--contrast-csv not found: {cpath}")
            contrast_by_key = _load_contrast_csv(cpath)
            print(
                f"[holdout] loaded contrast for {len(contrast_by_key)} cases "
                f"from {cpath}"
            )
        picked, recipe = pick_stratified_holdout(
            groups,
            n=args.n,
            contrast_by_key=contrast_by_key,
            seed=args.seed,
        )
        if not picked:
            raise SystemExit("[holdout] no cases picked (empty groups)")
        out_path = Path(args.write_holdout)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(out_path, "w") as fh:
            fh.write(f"# stratified holdout — generated {ts}\n")
            fh.write(f"# script: scripts/make_stratified_splits.py --write-holdout\n")
            fh.write(f"# dataset: {ds_name}\n")
            fh.write(f"# n_requested: {recipe['n_requested']}\n")
            fh.write(f"# n_groups_total: {recipe['n_groups_total']}\n")
            fh.write(f"# stratified_by: {recipe['stratified_by']}\n")
            fh.write(f"# vol_quartile_edges (vox): {recipe['vol_quartile_edges']}\n")
            if recipe['contrast_median'] is not None:
                fh.write(f"# contrast_median: {recipe['contrast_median']}\n")
            fh.write(f"# seed: {recipe['seed']}\n")
            fh.write("# one patient key per line (excluded from train + held out for unseen test)\n")
            for k in picked:
                fh.write(f"{k}\n")
        print(f"[holdout] wrote {len(picked)} ids -> {out_path}")
        vmax_by_key = {key: vmax for key, _, vmax in groups}
        for k in picked:
            print(f"  {k}: vol_max={vmax_by_key.get(k, 'n/a')}")
        return 0

    folds = stratified_folds_from_groups(groups, args.n_folds)
    print(f"[splits] {args.n_folds} folds (validation cases per fold):")
    for i, vf in enumerate(folds):
        vols_in_fold = [vol_by_case[c] for c in vf]
        print(
            f"  fold {i}: n={len(vf)}, vol_min={min(vols_in_fold)}, "
            f"vol_max={max(vols_in_fold)}, vol_sum={sum(vols_in_fold)} -> {sorted(vf)}"
        )

    splits = build_splits(folds)
    out = nnunet_preproc / ds_name / "splits_final.json"
    if args.dry_run:
        print(f"[splits] dry-run: would write {out}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(splits, fh, indent=2)
    print(f"[splits] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
