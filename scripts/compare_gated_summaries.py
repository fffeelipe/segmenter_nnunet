"""Compare two gated-ensemble summary.json files on the shared subset of
cases and report per-case deltas + mean deltas.

Example
-------

    python scripts/compare_gated_summaries.py \
        --baseline nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_v2c_sigmoid/summary.json \
        --candidate nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_invgamma2d_vs_base3d_folds0_2/summary.json \
        --label-baseline "gated v2c (baseline 2D + 3D)" \
        --label-candidate "gated v2c (invgamma 2D + 3D)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_per_case(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for pc in data.get("per_case", []):
        out[pc["case"]] = pc
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--label-baseline", default="baseline")
    ap.add_argument("--label-candidate", default="candidate")
    ap.add_argument("--field", default="dice_gated",
                    help="Field to compare (default: dice_gated). "
                         "Try 'dice_2d' to check raw 2D means.")
    args = ap.parse_args()

    base = load_per_case(args.baseline)
    cand = load_per_case(args.candidate)
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise SystemExit("No shared cases between the two summaries")

    only_base = sorted(set(base) - set(cand))
    only_cand = sorted(set(cand) - set(base))

    print(f"{args.label_baseline} cases: {len(base)}")
    print(f"{args.label_candidate} cases: {len(cand)}")
    print(f"Shared cases : {len(shared)}")
    if only_cand:
        print(f"  only in candidate ({len(only_cand)}): "
              f"{', '.join(only_cand)}")
    if only_base:
        print(f"  only in baseline ({len(only_base)}): "
              f"{', '.join(only_base[:10])}{'...' if len(only_base) > 10 else ''}")

    print(f"\n{'case':<8}  {args.label_baseline[:14]:>14}  "
          f"{args.label_candidate[:14]:>14}  {'Δ':>8}  flag")
    print("-" * 70)
    diffs = []
    for cid in shared:
        b = float(base[cid].get(args.field, float("nan")))
        c = float(cand[cid].get(args.field, float("nan")))
        d = c - b
        flag = ""
        if d >= 0.1:
            flag = "  <-- WIN"
        elif d <= -0.1:
            flag = "  <-- REG"
        elif d >= 0.02:
            flag = "  +"
        elif d <= -0.02:
            flag = "  -"
        print(f"{cid:<8}  {b:>14.4f}  {c:>14.4f}  {d:>+8.4f}{flag}")
        diffs.append((cid, b, c, d))

    mean_b = sum(d[1] for d in diffs) / len(diffs)
    mean_c = sum(d[2] for d in diffs) / len(diffs)
    print("-" * 70)
    print(f"{'mean':<8}  {mean_b:>14.4f}  {mean_c:>14.4f}  "
          f"{mean_c - mean_b:>+8.4f}")

    n_wins = sum(1 for _, _, _, d in diffs if d > 0.02)
    n_reg = sum(1 for _, _, _, d in diffs if d < -0.02)
    n_flat = len(diffs) - n_wins - n_reg
    big_wins = [c for c in diffs if c[3] >= 0.1]
    big_regs = [c for c in diffs if c[3] <= -0.1]
    print()
    print(f"Summary: {n_wins} wins (>+0.02), {n_reg} regressions "
          f"(<-0.02), {n_flat} flat")
    if big_wins:
        print("  big wins (+≥0.10):")
        for cid, b, c, d in big_wins:
            print(f"    {cid}  {b:.3f} -> {c:.3f}  ({d:+.3f})")
    if big_regs:
        print("  big regressions (-≥0.10):")
        for cid, b, c, d in big_regs:
            print(f"    {cid}  {b:.3f} -> {c:.3f}  ({d:+.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
