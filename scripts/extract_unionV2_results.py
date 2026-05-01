#!/usr/bin/env python3
"""
Extract a compact results summary for Dataset503 union_v2 experiments.

Inputs (defaults match this repo layout):
- reports/per_case_baseline_and_mixed.csv
- nnUNet_results_union_V2/Dataset503_ALT_T1T2/gated_ensemble_alt_gated_v2c/summary.json

Outputs:
- reports/union_v2_503_results.json

This script is presentation-oriented: it pulls the headline mean/global Dice
for 2D, 3D, and gated v2c, and also emits a short list of best/worst cases
using the per-case rows in the CSV (patient-level, not derived *_T1/_T2).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CSV = Path("reports/per_case_baseline_and_mixed.csv")
DEFAULT_GATED = Path(
    "nnUNet_results_union_V2/Dataset503_ALT_T1T2/gated_ensemble_alt_gated_v2c/summary.json"
)
DEFAULT_OUT = Path("reports/union_v2_503_results.json")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _row_by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case = row.get("case", "")
        if case:
            out[case] = row
    return out


def _f(x: str | None) -> float | None:
    if x is None:
        return None
    s = x.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _topk(items: list[dict[str, Any]], *, key: str, k: int, reverse: bool) -> list[dict[str, Any]]:
    def _key(it: dict[str, Any]) -> float:
        v = it.get(key)
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    return sorted(items, key=_key, reverse=reverse)[:k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Per-case CSV table.")
    ap.add_argument("--gated-summary", type=Path, default=DEFAULT_GATED, help="Gated v2c summary.json.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path.")
    ap.add_argument("--k", type=int, default=8, help="How many best/worst cases to emit.")
    args = ap.parse_args()

    rows = _read_csv(args.csv)
    by_case = _row_by_case(rows)

    global_row = by_case.get("__GLOBAL_DICE__")
    mean_row = by_case.get("__MEAN_DICE__")
    if global_row is None or mean_row is None:
        raise SystemExit(f"Missing __GLOBAL_DICE__ or __MEAN_DICE__ rows in {args.csv}")

    cols = {
        "2d": "fusion_union_v2_503_2d",
        "3d": "fusion_union_v2_503_3d",
        "gated_v2c": "fusion_union_v2_503_gated_v2c",
        "softavg": "fusion_union_v2_503_softavg_2d3d_tta",
        "gated_v2b": "fusion_union_v2_503_gated_v2b",
    }

    headline = {
        "mean_dice": {k: _f(mean_row.get(c)) for k, c in cols.items()},
        "global_dice": {k: _f(global_row.get(c)) for k, c in cols.items()},
    }

    per_case = []
    for case, row in by_case.items():
        if case.startswith("__"):
            continue
        rec = {"case": case}
        for k, c in cols.items():
            rec[k] = _f(row.get(c))
        if rec.get("gated_v2c") is not None and rec.get("3d") is not None:
            rec["delta_gated_minus_3d"] = rec["gated_v2c"] - rec["3d"]
        else:
            rec["delta_gated_minus_3d"] = None
        per_case.append(rec)

    best_gated = _topk(
        [r for r in per_case if r.get("gated_v2c") is not None],
        key="gated_v2c",
        k=args.k,
        reverse=True,
    )
    worst_gated = _topk(
        [r for r in per_case if r.get("gated_v2c") is not None],
        key="gated_v2c",
        k=args.k,
        reverse=False,
    )
    best_delta = _topk(
        [r for r in per_case if r.get("delta_gated_minus_3d") is not None],
        key="delta_gated_minus_3d",
        k=args.k,
        reverse=True,
    )
    worst_delta = _topk(
        [r for r in per_case if r.get("delta_gated_minus_3d") is not None],
        key="delta_gated_minus_3d",
        k=args.k,
        reverse=False,
    )

    gated_meta: dict[str, Any] | None = None
    if args.gated_summary.is_file():
        gated_meta = json.loads(args.gated_summary.read_text(encoding="utf-8"))
        # Keep only the configuration-level keys, not the full per-case list.
        gated_meta = {
            k: gated_meta.get(k)
            for k in [
                "dataset",
                "trainer_2d",
                "trainer_3d",
                "plans_2d",
                "config_2d",
                "plans_3d",
                "config_3d",
                "gate_mode",
                "min_fg_voxels",
                "min_fg_ratio",
                "tau",
                "use_confidence",
                "conf_power",
                "n_cases",
                "mean_dice_2d",
                "mean_dice_3d",
                "mean_dice_ens",
                "mean_dice_gated",
            ]
        }

    out = {
        "source": {
            "csv": str(args.csv),
            "gated_summary": str(args.gated_summary) if args.gated_summary else None,
        },
        "headline": headline,
        "gated_v2c_summary_meta": gated_meta,
        "rankings": {
            "best_gated_v2c": best_gated,
            "worst_gated_v2c": worst_gated,
            "best_delta_gated_minus_3d": best_delta,
            "worst_delta_gated_minus_3d": worst_delta,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

