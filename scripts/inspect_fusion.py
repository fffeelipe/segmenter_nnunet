#!/usr/bin/env python3
"""Pretty-print the ``fusion_report.json`` produced by ``build_t1t2_dataset.py``.

Renders one markdown row per case with the inter-rater metrics that matter
for QC (fg counts in both modalities, retained fraction after resampling,
rater Dice, fused fg, whether the two acquisitions share direction).

Usage:

    python scripts/inspect_fusion.py \
        [--report $nnUNet_raw/Dataset503_ALT_T1T2/fusion_report.json] \
        [--sort-by dice|retained|t1_fg|case_id] [--desc]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SORT_KEYS = {
    "case_id": lambda c: c.get("case_id", ""),
    "dice": lambda c: c.get("dice_raters") or -1,
    "retained": lambda c: c.get("retained") if c.get("retained") is not None else -1,
    "t1_fg": lambda c: c.get("t1_fg", 0),
    "t2_fg": lambda c: c.get("t2_fg_on_t1", 0),
    "fused_fg": lambda c: c.get("fused_fg", 0),
}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _default_report_path() -> Path | None:
    env_raw = os.environ.get("nnUNet_raw")
    if env_raw:
        p = Path(env_raw) / "Dataset503_ALT_T1T2" / "fusion_report.json"
        if p.is_file():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Path to fusion_report.json. Defaults to "
            "$nnUNet_raw/Dataset503_ALT_T1T2/fusion_report.json when set."
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=sorted(SORT_KEYS.keys()),
        default="case_id",
        help="Column to sort by (default: case_id).",
    )
    parser.add_argument("--desc", action="store_true", help="Sort descending.")
    args = parser.parse_args()

    report_path = args.report or _default_report_path()
    if report_path is None or not report_path.is_file():
        raise SystemExit(
            "fusion_report.json not found. Pass --report <path> or set nnUNet_raw."
        )

    with open(report_path) as fh:
        data = json.load(fh)

    fusion_mode = data.get("fusion_mode", "?")
    cases = data.get("cases", [])
    cases_sorted = sorted(cases, key=SORT_KEYS[args.sort_by], reverse=args.desc)

    n_dir_mismatch = sum(1 for c in cases if c.get("direction_mismatch"))
    n_same_grid = sum(1 for c in cases if c.get("same_grid"))

    print(f"# fusion report: {report_path}")
    print()
    print(f"- fusion_mode: **{fusion_mode}**")
    print(f"- min_foreground_retained: {data.get('min_foreground_retained')}")
    print(f"- common patients: {data.get('num_common_patients')}")
    print(f"- written: {data.get('num_written')}")
    print(f"- same grid (T1≡T2): {n_same_grid}")
    print(f"- direction mismatch: {n_dir_mismatch}")
    print()

    header = (
        "| case | T1 fg | T2 fg (native) | T2 fg → ref | t2_retained | "
        "Dice raters | fused fg | dir mismatch | join ok | iso(mm) |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    print(header)
    print(sep)
    for c in cases_sorted:
        ref = c.get("ref") or {}
        iso = ref.get("iso_spacing")
        print(
            "| {case} | {t1} | {t2n} | {t2t1} | {ret} | {dice} | {fused} | "
            "{dirm} | {joinok} | {iso} |".format(
                case=c.get("case_id", "?"),
                t1=_fmt(c.get("t1_fg")),
                t2n=_fmt(c.get("t2_fg_native")),
                t2t1=_fmt(c.get("t2_fg_on_t1")),
                ret=_fmt(c.get("retained")),
                dice=_fmt(c.get("dice_raters")),
                fused=_fmt(c.get("fused_fg")),
                dirm="yes" if c.get("direction_mismatch") else "no",
                joinok="yes" if c.get("join_qc_passed") else "no",
                iso=_fmt(iso),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
