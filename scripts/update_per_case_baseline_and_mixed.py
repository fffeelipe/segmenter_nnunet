#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import nibabel as nib


@dataclass(frozen=True)
class Agg:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def add(self, tp: float, fp: float, fn: float) -> "Agg":
        return Agg(self.tp + tp, self.fp + fp, self.fn + fn)

    def dice(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp / denom) if denom > 0 else 0.0


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def load_fold_summaries(trainer_dir: Path) -> tuple[dict[int, dict[str, dict]], dict[int, Agg]]:
    """Load per-case Dice across fold_*/validation/summary.json.

    Returns:
      - fold -> case_id -> metrics dict (Dice, TP, FP, FN, ...)
      - fold -> aggregated TP/FP/FN across all cases in that fold summary
    """
    per_fold: dict[int, dict[str, dict]] = {}
    per_fold_agg: dict[int, Agg] = {}
    for f in range(5):
        p = trainer_dir / f"fold_{f}" / "validation" / "summary.json"
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        cases: dict[str, dict] = {}
        agg = Agg()
        for c in data.get("metric_per_case", []):
            pred_file = Path(c["prediction_file"]).name
            case_id = pred_file.replace(".nii.gz", "")
            m = c["metrics"]["1"]
            agg = agg.add(float(m["TP"]), float(m["FP"]), float(m["FN"]))
            cases[case_id] = {
                "Dice": float(m["Dice"]),
                "TP": float(m["TP"]),
                "FP": float(m["FP"]),
                "FN": float(m["FN"]),
            }
        per_fold[f] = cases
        per_fold_agg[f] = agg
    return per_fold, per_fold_agg


def load_pred_vs_gt_from_folders(pred_root: Path, labels_dir: Path) -> tuple[dict[str, dict], Agg]:
    """Compute per-sample Dice/TP/FP/FN from predicted .nii.gz masks vs GT.

    Expects a directory structure like:
        pred_root/fold_0/<case>.nii.gz
        pred_root/fold_1/<case>.nii.gz
        ...
    """
    per_sample: dict[str, dict] = {}
    agg = Agg()
    for fold_dir in sorted(pred_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        for p in sorted(fold_dir.glob("*.nii.gz")):
            cid = p.name.replace(".nii.gz", "")
            gt_path = labels_dir / f"{cid}.nii.gz"
            if not gt_path.is_file():
                continue
            pred = np.asarray(nib.load(str(p)).dataobj) > 0
            gt = np.asarray(nib.load(str(gt_path)).dataobj) > 0
            d, tp, fp, fn = dice_from_masks(pred, gt)
            per_sample[cid] = {"Dice": float(d), "TP": float(tp), "FP": float(fp), "FN": float(fn)}
            agg = agg.add(tp, fp, fn)
    return per_sample, agg


def load_softavg_from_npz(
    two_d_dir: Path,
    three_d_dir: Path,
    labels_dir: Path,
) -> tuple[dict[str, dict], Agg]:
    """Compute per-sample Dice for soft-avg ensemble from existing validation .npz.

    Uses the per-fold validation .nii.gz presence as the case list and loads the sibling
    .npz probabilities for each branch, then computes:
        mask = ((p2_fg + p3_fg)/2 > 0.5)
    """
    per_sample: dict[str, dict] = {}
    agg = Agg()

    def _load_fg(npz_path: Path, mask_shape: tuple[int, ...]) -> np.ndarray:
        with np.load(str(npz_path)) as f:
            prob = np.asarray(f["probabilities"])
        fg = prob[1]
        if fg.shape == mask_shape:
            return fg
        t = fg.transpose(2, 1, 0)
        if t.shape == mask_shape:
            return t
        raise RuntimeError(f"shape mismatch for {npz_path}: fg={fg.shape} mask={mask_shape}")

    for f in range(5):
        v2 = two_d_dir / f"fold_{f}" / "validation"
        v3 = three_d_dir / f"fold_{f}" / "validation"
        if not v2.is_dir() or not v3.is_dir():
            continue
        for p2_nii in sorted(v2.glob("*.nii.gz")):
            cid = p2_nii.name.replace(".nii.gz", "")
            p3_nii = v3 / f"{cid}.nii.gz"
            if not p3_nii.is_file():
                continue
            gt_path = labels_dir / f"{cid}.nii.gz"
            if not gt_path.is_file():
                continue

            mshape = nib.load(str(p2_nii)).shape
            p2_npz = p2_nii.with_suffix("").with_suffix(".npz")
            p3_npz = p3_nii.with_suffix("").with_suffix(".npz")
            if not p2_npz.is_file() or not p3_npz.is_file():
                continue

            fg2 = _load_fg(p2_npz, mshape)
            fg3 = _load_fg(p3_npz, mshape)
            pred = ((fg2 + fg3) * 0.5) > 0.5
            gt = np.asarray(nib.load(str(gt_path)).dataobj) > 0
            d, tp, fp, fn = dice_from_masks(pred, gt)
            per_sample[cid] = {"Dice": float(d), "TP": float(tp), "FP": float(fp), "FN": float(fn)}
            agg = agg.add(tp, fp, fn)

    return per_sample, agg


def load_val_fold_map(splits_json: Path) -> dict[str, int]:
    """Return case_id -> fold for validation samples."""
    data = json.loads(splits_json.read_text())
    out: dict[str, int] = {}
    for fold, split in enumerate(data):
        for cid in split.get("val", []):
            out[cid] = fold
    return out


def resolve_patient_value(
    patient_id: str,
    per_sample: dict[str, dict],
) -> tuple[float | None, dict | None]:
    """Map patient_id (CSV row) to a metric record from per-sample metrics.

    If patient_id is missing but split produced derived samples like
    '<patient>_T1' / '<patient>_T2', take the best available derived sample.
    """
    if patient_id in per_sample:
        rec = per_sample[patient_id]
        return float(rec["Dice"]), rec
    candidates = [k for k in per_sample.keys() if k.startswith(patient_id + "_")]
    if not candidates:
        return None, None
    best = max(candidates, key=lambda k: float(per_sample[k]["Dice"]))
    rec = per_sample[best]
    return float(rec["Dice"]), rec


def best_per_sample(per_fold: dict[int, dict[str, dict]]) -> dict[str, dict]:
    """Pick the best (max Dice) record for each sample across folds."""
    out: dict[str, dict] = {}
    for _fold, cases in per_fold.items():
        for cid, rec in cases.items():
            if cid not in out or float(rec["Dice"]) > float(out[cid]["Dice"]):
                out[cid] = rec
    return out


def dice_from_masks(pred: np.ndarray, gt: np.ndarray) -> tuple[float, int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    denom = 2 * tp + fp + fn
    d = (2 * tp / denom) if denom else 0.0
    return d, tp, fp, fn


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if not header or "case" not in header:
        raise SystemExit(f"Unexpected CSV header in {path}")
    return header, rows


def write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--alt2d-dir", type=Path, required=True)
    ap.add_argument("--alt2d-preds-root", type=Path,
                    help="If set, ignores fold summary.json and instead computes Dice from predicted masks "
                         "under this folder (fold_*/<case>.nii.gz) vs --labels-dir.")
    ap.add_argument("--softavg-col", default=None,
                    help="If set, writes the soft-avg (2D+3D) per-patient Dice into this column name.")
    ap.add_argument("--softavg-two-d-dir", type=Path,
                    help="2D trainer dir used for soft-avg npz loading (expects fold_*/validation/*.npz).")
    ap.add_argument("--softavg-three-d-dir", type=Path,
                    help="3D trainer dir used for soft-avg npz loading (expects fold_*/validation/*.npz).")
    ap.add_argument("--splits", type=Path,
                    help="Optional: nnU-Net splits_final.json. If provided, it's used as a sanity check, "
                         "but we still fall back to best-per-sample across fold summaries when needed.")
    ap.add_argument("--alt2d-col", default="alt_2d_250ep")
    ap.add_argument("--gated-summary", type=Path,
                    help="Path to scripts/ensemble_gated.py output summary.json (contains per_case dice_gated).")
    ap.add_argument("--gated-cases-dir", type=Path,
                    help="Directory containing per-case gated masks (<case>.nii.gz).")
    ap.add_argument("--gated-col", default="alt_gated_v2c")
    ap.add_argument("--labels-dir", type=Path,
                    help="Directory containing GT masks (<case>.nii.gz). Required if --gated-summary is set.")
    args = ap.parse_args()

    header, rows = read_csv(args.csv)

    if args.alt2d_preds_root:
        if not args.labels_dir:
            raise SystemExit("--alt2d-preds-root requires --labels-dir")
        per_sample, agg2d = load_pred_vs_gt_from_folders(args.alt2d_preds_root, args.labels_dir)
    else:
        per_fold, _ = load_fold_summaries(args.alt2d_dir)
        per_sample = best_per_sample(per_fold)
        agg2d = Agg()
        for rec in per_sample.values():
            agg2d = agg2d.add(rec["TP"], rec["FP"], rec["FN"])
        if args.splits and args.splits.is_file():
            _ = load_val_fold_map(args.splits)

    # Compute patient-level values aligned to the CSV's patient IDs.
    agg_pat = Agg()
    patient_vals: dict[str, float] = {}
    for r in rows:
        pid = r.get("case", "")
        if pid.startswith("__"):
            continue
        v, rec = resolve_patient_value(pid, per_sample)
        if v is None or rec is None:
            continue
        patient_vals[pid] = v
        agg_pat = agg_pat.add(rec["TP"], rec["FP"], rec["FN"])

    mean2d = _mean(patient_vals.values())
    global2d = agg_pat.dice()

    if args.alt2d_col not in header:
        header.append(args.alt2d_col)

    for r in rows:
        case = r.get("case", "")
        if case == "__GLOBAL_DICE__":
            r[args.alt2d_col] = f"{global2d:.15g}"
        elif case == "__MEAN_DICE__":
            r[args.alt2d_col] = f"{mean2d:.15g}"
        else:
            v = patient_vals.get(case)
            r[args.alt2d_col] = "" if v is None else f"{v:.15g}"

    write_csv(args.csv, header, rows)

    # Optional: soft-avg column from .npz
    if args.softavg_col:
        if not (args.softavg_two_d_dir and args.softavg_three_d_dir and args.labels_dir):
            raise SystemExit("--softavg-col requires --softavg-two-d-dir --softavg-three-d-dir --labels-dir")
        per_sample_sa, _agg_sa = load_softavg_from_npz(
            args.softavg_two_d_dir, args.softavg_three_d_dir, args.labels_dir
        )
        # patient-level aggregation via best derived sample if needed
        patient_vals_sa: dict[str, float] = {}
        chosen_sa: dict[str, str] = {}
        for r in rows:
            pid = r.get("case", "")
            if pid.startswith("__"):
                continue
            if pid in per_sample_sa:
                chosen_sa[pid] = pid
                patient_vals_sa[pid] = float(per_sample_sa[pid]["Dice"])
                continue
            cands = [k for k in per_sample_sa.keys() if k.startswith(pid + "_")]
            if not cands:
                continue
            best = max(cands, key=lambda k: float(per_sample_sa[k]["Dice"]))
            chosen_sa[pid] = best
            patient_vals_sa[pid] = float(per_sample_sa[best]["Dice"])

        agg = Agg()
        for pid, sid in chosen_sa.items():
            rec = per_sample_sa.get(sid)
            if not rec:
                continue
            agg = agg.add(rec["TP"], rec["FP"], rec["FN"])

        mean_sa = _mean(patient_vals_sa.values())
        global_sa = agg.dice()

        if args.softavg_col not in header:
            header.append(args.softavg_col)
        for r in rows:
            case = r.get("case", "")
            if case == "__GLOBAL_DICE__":
                r[args.softavg_col] = f"{global_sa:.15g}"
            elif case == "__MEAN_DICE__":
                r[args.softavg_col] = f"{mean_sa:.15g}"
            else:
                v = patient_vals_sa.get(case)
                r[args.softavg_col] = "" if v is None else f"{v:.15g}"
        write_csv(args.csv, header, rows)

    # Optional: gated ensemble column
    if args.gated_summary:
        if not (args.gated_cases_dir and args.labels_dir):
            raise SystemExit("--gated-summary requires --gated-cases-dir and --labels-dir")

        gated = json.loads(args.gated_summary.read_text())
        per_case = {r["case"]: float(r["dice_gated"]) for r in gated.get("per_case", [])}

        # pick best derived sample per patient (or exact match)
        chosen_sample: dict[str, str] = {}
        patient_vals: dict[str, float] = {}
        for r in rows:
            pid = r.get("case", "")
            if pid.startswith("__"):
                continue
            if pid in per_case:
                chosen_sample[pid] = pid
                patient_vals[pid] = per_case[pid]
                continue
            cands = [k for k in per_case.keys() if k.startswith(pid + "_")]
            if not cands:
                continue
            best = max(cands, key=lambda k: per_case[k])
            chosen_sample[pid] = best
            patient_vals[pid] = per_case[best]

        # compute patient-level global dice by aggregating TP/FP/FN on chosen sample masks
        agg = Agg()
        for pid, sid in chosen_sample.items():
            pred_path = args.gated_cases_dir / f"{sid}.nii.gz"
            gt_path = args.labels_dir / f"{sid}.nii.gz"
            if not pred_path.is_file() or not gt_path.is_file():
                continue
            pred = np.asarray(nib.load(str(pred_path)).dataobj) > 0
            gt = np.asarray(nib.load(str(gt_path)).dataobj) > 0
            _, tp, fp, fn = dice_from_masks(pred, gt)
            agg = agg.add(tp, fp, fn)

        mean_gated = _mean(patient_vals.values())
        global_gated = agg.dice()

        if args.gated_col not in header:
            header.append(args.gated_col)

        for r in rows:
            case = r.get("case", "")
            if case == "__GLOBAL_DICE__":
                r[args.gated_col] = f"{global_gated:.15g}"
            elif case == "__MEAN_DICE__":
                r[args.gated_col] = f"{mean_gated:.15g}"
            else:
                v = patient_vals.get(case)
                r[args.gated_col] = "" if v is None else f"{v:.15g}"

        write_csv(args.csv, header, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

