#!/usr/bin/env python3
"""
Export per-case Dice CSVs for baseline and mixed (ensemble/gated) variants.

Supports two input formats:
1) nnU-Net evaluation summary.json with "metric_per_case" (folder evaluation)
2) Custom gated ensemble summary.json with top-level "per_case"

Writes a wide CSV: one row per case, one column per method.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def _case_id_from_pred_path(p: str) -> str:
    b = os.path.basename(p)
    if b.endswith(".nii.gz"):
        b = b[: -len(".nii.gz")]
    elif b.endswith(".nii"):
        b = b[: -len(".nii")]
    return b


def load_per_case_dice(summary_json: Path) -> dict[str, float]:
    data = json.loads(summary_json.read_text())

    if "metric_per_case" in data:
        out: dict[str, float] = {}
        for c in data.get("metric_per_case", []):
            case_id = _case_id_from_pred_path(c["prediction_file"])
            out[case_id] = float(c["metrics"]["1"]["Dice"])
        return out

    if "per_case" in data:
        # Our gated ensemble format includes several dice_* fields.
        # Prefer dice_gated when present, else dice_ens as a reasonable default.
        out = {}
        for c in data.get("per_case", []):
            case_id = str(c["case"])
            if "dice_gated" in c:
                out[case_id] = float(c["dice_gated"])
            elif "dice_ens" in c:
                out[case_id] = float(c["dice_ens"])
            else:
                raise SystemExit(f"{summary_json}: per_case missing dice_gated/dice_ens for {case_id}")
        return out

    raise SystemExit(f"{summary_json}: unsupported format (no metric_per_case or per_case)")

def load_per_case_dice_from_cv_trainer_dir(trainer_dir: Path) -> dict[str, float]:
    """
    Merge per-case Dice across fold_*/validation/summary.json (each case appears once).
    """
    out: dict[str, float] = {}
    any_found = False
    for f in range(5):
        p = trainer_dir / f"fold_{f}" / "validation" / "summary.json"
        if not p.is_file():
            continue
        any_found = True
        fold_cases = load_per_case_dice(p)
        # safety: do not allow duplicates (a case must belong to exactly one fold)
        dup = set(out).intersection(fold_cases)
        if dup:
            raise SystemExit(f"{trainer_dir}: duplicate cases across folds: {sorted(dup)[:5]}...")
        out.update(fold_cases)
    if not any_found:
        raise SystemExit(f"{trainer_dir}: no fold_<f>/validation/summary.json found")
    return out


def write_wide_csv(rows: dict[str, dict[str, float]], methods: list[str], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case"] + methods)
        for case_id in sorted(rows.keys()):
            r = rows[case_id]
            w.writerow([case_id] + [r.get(m, "") for m in methods])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--method",
        action="append",
        nargs=2,
        metavar=("NAME", "SUMMARY_JSON"),
        required=True,
        help="Method name and path to its summary.json. Repeatable.",
    )
    args = ap.parse_args()

    method_paths: list[tuple[str, Path]] = [(n, Path(p)) for (n, p) in args.method]

    per_method: dict[str, dict[str, float]] = {}
    for name, p in method_paths:
        if p.is_dir():
            per_method[name] = load_per_case_dice_from_cv_trainer_dir(p)
        else:
            if not p.is_file():
                raise SystemExit(f"Missing file: {p}")
            per_method[name] = load_per_case_dice(p)

    all_cases = sorted({c for d in per_method.values() for c in d.keys()})
    rows: dict[str, dict[str, float]] = {c: {} for c in all_cases}
    for name, d in per_method.items():
        for case_id, dice in d.items():
            rows[case_id][name] = dice

    write_wide_csv(rows, [n for n, _ in method_paths], Path(args.out))


if __name__ == "__main__":
    main()

