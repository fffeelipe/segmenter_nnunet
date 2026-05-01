#!/usr/bin/env python3
"""Preview what ``FUSION=intersection`` would produce for Dataset503_ALT_T1T2.

Reads T1/<case>/ and T2/<case>/ from the repo root, resamples the T2 label
onto the T1 grid (same logic as ``build_t1t2_dataset.py``), and reports:

- t1_fg, t2_fg_on_t1, union_fg, inter_fg per case.
- Ratio inter / t1 (how much the model would still "see" of the T1 annotation).
- Cases where inter == 0 (all-background labels if we trained intersection).
- Per-case comparison to union_fg (how much territory the fusion change drops).

No images are written. Safe to run while training.

Usage:
    python scripts/preview_intersection.py [--src .] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_t1t2_dataset import (  # noqa: E402
    _direction_matches,
    _resample_mask_to,
)
from convert_to_nnunet import (  # noqa: E402
    _prebinarize,
    align_label_to_image,
    find_pair,
)


def preview_case(case_dir_t1: Path, case_dir_t2: Path) -> dict | None:
    case_id = case_dir_t1.name
    t1_pair = find_pair(case_dir_t1)
    t2_pair = find_pair(case_dir_t2)
    if t1_pair is None or t2_pair is None:
        print(f"[{case_id}] missing T1 or T2 image/label pair, skipping")
        return None
    t1_img_path, t1_lbl_path = t1_pair
    t2_img_path, t2_lbl_path = t2_pair
    if not all([t1_img_path, t1_lbl_path, t2_img_path, t2_lbl_path]):
        print(f"[{case_id}] incomplete T1/T2 pair, skipping")
        return None

    t1_img = sitk.ReadImage(str(t1_img_path))
    t1_lbl_raw = _prebinarize(sitk.ReadImage(str(t1_lbl_path)))
    t2_img = sitk.ReadImage(str(t2_img_path))
    t2_lbl_raw = _prebinarize(sitk.ReadImage(str(t2_lbl_path)))

    t1_lbl = align_label_to_image(
        t1_lbl_raw, reference_img=t1_img, case_id=f"T1/{case_id}"
    )
    t2_lbl_on_t2 = align_label_to_image(
        t2_lbl_raw, reference_img=t2_img, case_id=f"T2/{case_id}"
    )

    t2_lbl_on_t1 = _resample_mask_to(t1_img, t2_lbl_on_t2)

    t1_arr = sitk.GetArrayFromImage(t1_lbl).astype(bool)
    t2_on_t1_arr = sitk.GetArrayFromImage(t2_lbl_on_t1).astype(bool)

    union = t1_arr | t2_on_t1_arr
    inter = t1_arr & t2_on_t1_arr

    t1_fg = int(t1_arr.sum())
    t2_fg = int(t2_on_t1_arr.sum())
    u = int(union.sum())
    i = int(inter.sum())

    return {
        "case_id": case_id,
        "direction_mismatch": not _direction_matches(t1_img, t2_img),
        "t1_fg": t1_fg,
        "t2_fg_on_t1": t2_fg,
        "union_fg": u,
        "inter_fg": i,
        "inter_over_t1": round(i / t1_fg, 4) if t1_fg else None,
        "inter_over_union": round(i / u, 4) if u else None,
        "empty_under_intersection": i == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=".", type=Path)
    parser.add_argument("--json", default=None, type=Path)
    parser.add_argument(
        "--min-inter", type=int, default=1000,
        help="Flag cases with inter_fg < this as problematic (default 1000)."
    )
    args = parser.parse_args()

    t1_root = args.src / "T1"
    t2_root = args.src / "T2"
    if not t1_root.exists() or not t2_root.exists():
        raise FileNotFoundError(f"Need both {t1_root} and {t2_root}")

    common = sorted(
        {p.name for p in t1_root.iterdir() if p.is_dir()}
        & {p.name for p in t2_root.iterdir() if p.is_dir()}
    )
    print(f"Previewing intersection fusion for {len(common)} common cases")

    rows = []
    for case_id in common:
        row = preview_case(t1_root / case_id, t2_root / case_id)
        if row is None:
            continue
        rows.append(row)

    print()
    print(f"{'case':<8} {'dir_mm':>6} {'t1_fg':>8} {'t2_fg':>8} "
          f"{'union':>8} {'inter':>8} {'i/t1':>6} {'i/u':>6}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: x["inter_fg"]):
        print(
            f"{r['case_id']:<8} "
            f"{'Y' if r['direction_mismatch'] else '.':>6} "
            f"{r['t1_fg']:>8} {r['t2_fg_on_t1']:>8} "
            f"{r['union_fg']:>8} {r['inter_fg']:>8} "
            f"{(r['inter_over_t1'] or 0):>6.2f} "
            f"{(r['inter_over_union'] or 0):>6.2f}"
        )

    empty = [r for r in rows if r["empty_under_intersection"]]
    tiny = [r for r in rows if 0 < r["inter_fg"] < args.min_inter]
    print()
    print(f"Cases with inter_fg == 0 (would train as all-background): "
          f"n={len(empty)}")
    for r in empty:
        print(f"  {r['case_id']}  t1={r['t1_fg']} t2={r['t2_fg_on_t1']} "
              f"union={r['union_fg']} dir_mm={r['direction_mismatch']}")

    print()
    print(f"Cases with 0 < inter_fg < {args.min_inter} "
          f"(tiny-tumor risk under intersection): n={len(tiny)}")
    for r in tiny:
        print(f"  {r['case_id']}  inter={r['inter_fg']} t1={r['t1_fg']} "
              f"union={r['union_fg']} i/t1={r['inter_over_t1']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\n[wrote] {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
