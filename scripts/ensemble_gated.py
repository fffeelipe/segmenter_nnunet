#!/usr/bin/env python3
"""Gated ensemble of 2D and 3D nnU-Net validation predictions.

The soft-average ensemble that ``nnUNetv2_find_best_configuration`` produces
collapses to ~0 on the cases where *one* config predicts (almost) empty:
e.g. IOG48 (2D=0.63, 3D=0.00, ensemble=0.00), IOG31 (2D=0.75, 3D=0.00,
ensemble=0.09), IOG36 (2D=0.00, 3D=0.77, ensemble=0.22). The softmax
of the empty model is ~0 everywhere, and averaging it with the good
model drags the argmax below 0.5.

Two gate modes:

* ``--gate-mode hard`` (default, Semana 1 winner = gated v2b = 0.7805):
  binary rule per case. A prediction counts as "empty" when *either*:

      |pred_X| < MIN_FG_VOXELS                             (absolute gate)
      |pred_X| < MIN_FG_RATIO * |pred_Y|   (0 disables)    (relative gate)

  Then:

      if pred_2d empty and pred_3d not empty: use 3D
      if pred_3d empty and pred_2d not empty: use 2D
      else: keep the soft-averaged ensemble prediction

  The relative gate catches cases like IOG48 (2D=13 135 vox, 3D=589 vox,
  ratio 4.5 %) where the minority prediction is a tiny speck in the wrong
  place that drags the soft-avg below argmax 0.5.

* ``--gate-mode sigmoid`` (Semana 2, Exp. 2a): continuous per-case
  weights. Requires ``.npz`` softmax files (written when training used
  ``--npz``). For each case:

      w2 = sigmoid((n_pred_2d − v_min) / tau)
      w3 = sigmoid((n_pred_3d − v_min) / tau)
      if --use-confidence:
          w2 *= mean(p2_softmax[fg] | mask_2d)
          w3 *= mean(p3_softmax[fg] | mask_3d)
      w2, w3 := w2/(w2+w3), w3/(w2+w3)
      p_final = w2 * p2_softmax[fg] + w3 * p3_softmax[fg]
      mask    = (p_final > 0.5)

  Collapses to the hard gate at the extremes (n ≫ v_min ⇒ w=1,
  n ≪ v_min ⇒ w=0) and to soft-avg when both configs predict similar
  volumes (w2 = w3 = 0.5). The hyper-parameter ``tau`` controls the
  softness of the gate; ``use-confidence`` adds a tiebreaker on cases
  where both configs predict non-empty volumes but one is more certain.

No retraining; just re-combines the argmax NIfTIs (and optionally the
softmax .npz) that nnU-Net wrote during per-fold validation.

Usage (with nnU-Net env vars set):

    # Hard gate (Semana 1 baseline)
    python scripts/ensemble_gated.py --min-fg-voxels 50 --min-fg-ratio 0.40

    # Sigmoid gate (Semana 2, Exp. 2a)
    python scripts/ensemble_gated.py --gate-mode sigmoid --tau 200 \\
        --out-name gated_v4_sig200

    # Sigmoid + confidence tiebreaker
    python scripts/ensemble_gated.py --gate-mode sigmoid --tau 200 \\
        --use-confidence --out-name gated_v4_sig200_conf

Writes:
    $nnUNet_results/<dataset>/gated_ensemble_<out_name>/
        <case>.nii.gz    -> final binary segmentation per case
        summary.json     -> per-case Dice (2D / 3D / ens / gated) + means
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def env_path(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"env var {name} is not set")
    return Path(v)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0 if tp == 0 and fp == 0 and fn == 0 else 0.0
    return 2.0 * tp / denom


def load_per_fold(trainer_dir: Path) -> dict[str, Path]:
    """Return ``{case_id: nii_path}`` across all ``fold_*/validation/``."""
    out: dict[str, Path] = {}
    for f in range(5):
        vdir = trainer_dir / f"fold_{f}" / "validation"
        if not vdir.is_dir():
            continue
        for p in sorted(vdir.glob("*.nii.gz")):
            cid = p.name.replace(".nii.gz", "")
            out[cid] = p
    return out


def load_flat(dir_: Path) -> dict[str, Path]:
    """Return ``{case_id: nii_path}`` directly under ``dir_``."""
    out: dict[str, Path] = {}
    for p in sorted(dir_.glob("*.nii.gz")):
        cid = p.name.replace(".nii.gz", "")
        out[cid] = p
    return out


def load_fg_softmax(nii_path: Path, mask_shape: tuple[int, ...]) -> np.ndarray | None:
    """Load foreground-channel softmax from the sibling ``.npz`` of a
    validation NIfTI prediction.

    nnU-Net v2 stores ``probabilities`` as ``(C, Z, Y, X)`` at original
    resolution, while the NIfTI prediction is ``(X, Y, Z)``. This helper
    returns the foreground-channel probability transposed to match the
    NIfTI/mask shape.
    """
    npz_path = nii_path.with_suffix("").with_suffix(".npz")
    if not npz_path.is_file():
        return None
    with np.load(str(npz_path)) as f:
        if "probabilities" not in f.files:
            return None
        prob = np.asarray(f["probabilities"])
    if prob.ndim != 4 or prob.shape[0] < 2:
        return None
    fg = prob[1]
    if fg.shape == mask_shape:
        return fg
    if fg.transpose(2, 1, 0).shape == mask_shape:
        return fg.transpose(2, 1, 0)
    return None


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", default="Dataset501_ALT_T1",
                    help="Dataset folder under $nnUNet_results.")
    ap.add_argument("--trainer", default="nnUNetTrainerALT_os033_250epochs",
                    help="Default trainer used for BOTH 2D and 3D branches. "
                         "Override per-branch with --trainer-2d / "
                         "--trainer-3d when mixing trainers (e.g. invgamma "
                         "2D + baseline 3D).")
    ap.add_argument("--trainer-2d", default=None,
                    help="Override trainer for the 2D branch (defaults to "
                         "--trainer).")
    ap.add_argument("--trainer-3d", default=None,
                    help="Override trainer for the 3D branch (defaults to "
                         "--trainer).")
    ap.add_argument("--plans-2d", default="nnUNetPlans")
    ap.add_argument("--plans-3d", default="nnUNetTSMRIPlans")
    ap.add_argument("--config-2d", default="2d")
    ap.add_argument("--config-3d", default="3d_fullres")
    ap.add_argument("--folds", default="0 1 2 3 4",
                    help="Space-separated fold ids used in the ensemble "
                         "folder name (default: '0 1 2 3 4').")
    ap.add_argument("--gate-mode", choices=["hard", "sigmoid"], default="hard",
                    help="Gate strategy. 'hard' uses --min-fg-voxels / "
                         "--min-fg-ratio as binary thresholds (Semana 1 "
                         "winner = gated v2b = 0.7805). 'sigmoid' "
                         "computes continuous per-case weights from the "
                         "softmax .npz files (Exp. 2a). Default: hard.")
    ap.add_argument("--min-fg-voxels", type=int, default=50,
                    help="[hard mode] Predictions with fewer foreground "
                         "voxels than this are treated as empty and "
                         "replaced by the other config. [sigmoid mode] "
                         "Center of the volume-sigmoid (n=v_min -> "
                         "w=0.5). Default: 50.")
    ap.add_argument("--min-fg-ratio", type=float, default=0.1,
                    help="[hard mode only] A prediction is ALSO treated "
                         "as empty when it has fewer voxels than this "
                         "fraction of the other config's volume (e.g. "
                         "0.1 means 'less than 10%% of the other "
                         "model'). Set to 0 to disable. Default: 0.1.")
    ap.add_argument("--tau", type=float, default=200.0,
                    help="[sigmoid mode] Softness of the volume-sigmoid "
                         "(smaller = sharper gate). Default: 200 "
                         "(n=v_min+200 -> w≈0.73; n=v_min+500 -> w≈0.92).")
    ap.add_argument("--use-confidence", action="store_true",
                    help="[sigmoid mode] Multiply each config's weight by "
                         "the mean softmax prob inside its predicted "
                         "foreground mask. Helps break ties where both "
                         "configs predict non-empty volumes.")
    ap.add_argument("--conf-power", type=float, default=1.0,
                    help="[sigmoid mode] Raise each config's confidence to "
                         "this power before multiplying by the volume "
                         "weight. Larger values (e.g. 4, 8) amplify the "
                         "tie-breaking effect of confidence. Default: 1.0.")
    ap.add_argument("--out-name", default="gated",
                    help="Suffix for the output folder (default: 'gated').")
    ap.add_argument("--labels-dir",
                    help="Path to labelsTr (defaults to "
                         "$nnUNet_raw/<dataset>/labelsTr).")
    args = ap.parse_args()

    results = env_path("nnUNet_results")
    ds_dir = results / args.dataset
    if not ds_dir.is_dir():
        raise SystemExit(f"Missing dataset dir: {ds_dir}")

    trainer_2d = args.trainer_2d or args.trainer
    trainer_3d = args.trainer_3d or args.trainer
    d2_dir = ds_dir / f"{trainer_2d}__{args.plans_2d}__{args.config_2d}"
    d3_dir = ds_dir / f"{trainer_3d}__{args.plans_3d}__{args.config_3d}"
    fold_tag = "_".join(args.folds.split())
    ens_dir = (ds_dir / "ensembles"
               / f"ensemble___{trainer_2d}__{args.plans_2d}__{args.config_2d}"
                 f"___{trainer_3d}__{args.plans_3d}__{args.config_3d}"
                 f"___{fold_tag}")
    for p, lab in [(d2_dir, "2D trainer"), (d3_dir, "3D trainer")]:
        if not p.is_dir():
            raise SystemExit(f"Missing {lab} dir: {p}")
    ens_available = ens_dir.is_dir()
    if not ens_available:
        print(f"[warn] ensemble dir not found: {ens_dir}")
        print("[warn] soft-avg will be synthesized on the fly from softmax "
              "(requires --gate-mode sigmoid or .npz available).")

    p2 = load_per_fold(d2_dir)
    p3 = load_per_fold(d3_dir)
    pe = load_flat(ens_dir) if ens_available else {}

    shared_23 = set(p2) & set(p3)
    if ens_available:
        shared = sorted(shared_23 & set(pe))
    else:
        shared = sorted(shared_23)
    if not shared:
        raise SystemExit("No shared cases across 2D / 3D (/ ensemble) outputs.")

    labels_dir = (Path(args.labels_dir) if args.labels_dir
                  else env_path("nnUNet_raw") / args.dataset / "labelsTr")

    out_dir = ds_dir / f"gated_ensemble_{args.out_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir
    cases_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] 2D dir : {d2_dir}")
    print(f"[info] 3D dir : {d3_dir}")
    print(f"[info] ens dir: {ens_dir}")
    print(f"[info] labels : {labels_dir}")
    print(f"[info] out    : {out_dir}")
    if args.gate_mode == "hard":
        print(f"[info] gate   : hard  min_fg_voxels={args.min_fg_voxels}  "
              f"min_fg_ratio={args.min_fg_ratio}")
    else:
        print(f"[info] gate   : sigmoid  v_min={args.min_fg_voxels}  "
              f"tau={args.tau}  use_confidence={args.use_confidence}")
    print(f"[info] cases  : {len(shared)}")
    print()

    rows = []
    per_case = []
    for cid in shared:
        img2 = nib.load(str(p2[cid]))
        img3 = nib.load(str(p3[cid]))
        m2 = np.asarray(img2.dataobj) > 0
        m3 = np.asarray(img3.dataobj) > 0

        n2 = int(m2.sum())
        n3 = int(m3.sum())

        w2 = w3 = 0.5
        conf2 = conf3 = 0.0
        fg2 = fg3 = None

        if ens_available:
            me = np.asarray(nib.load(str(pe[cid])).dataobj) > 0
        else:
            fg2 = load_fg_softmax(p2[cid], m2.shape)
            fg3 = load_fg_softmax(p3[cid], m3.shape)
            if fg2 is None or fg3 is None:
                raise SystemExit(
                    f"[error] ensemble dir missing and softmax .npz "
                    f"unavailable for {cid}. Can't synthesize soft-avg.")
            me = ((fg2 + fg3) * 0.5) > 0.5

        if args.gate_mode == "hard":
            abs_t = args.min_fg_voxels
            rel_t = args.min_fg_ratio

            def _empty(n_self: int, n_other: int) -> bool:
                if n_self < abs_t:
                    return True
                if rel_t > 0 and n_other >= abs_t and n_self < rel_t * n_other:
                    return True
                return False

            e2 = _empty(n2, n3)
            e3 = _empty(n3, n2)

            if e2 and not e3:
                final = m3
                rule = "3d (2d empty)"
            elif e3 and not e2:
                final = m2
                rule = "2d (3d empty)"
            else:
                final = me
                rule = "soft-avg"

        else:  # sigmoid
            if fg2 is None:
                fg2 = load_fg_softmax(p2[cid], m2.shape)
            if fg3 is None:
                fg3 = load_fg_softmax(p3[cid], m3.shape)
            if fg2 is None or fg3 is None:
                raise SystemExit(
                    f"[error] gate-mode=sigmoid requires softmax .npz for "
                    f"{cid}. Missing or shape mismatch. Was training run "
                    f"with --npz?")

            v_min = float(args.min_fg_voxels)
            tau = max(1.0, float(args.tau))
            w2 = _sigmoid((n2 - v_min) / tau)
            w3 = _sigmoid((n3 - v_min) / tau)
            if args.use_confidence:
                k = max(1.0, float(args.conf_power))
                if n2 > 0:
                    conf2 = float(fg2[m2].mean())
                    w2 *= conf2 ** k
                if n3 > 0:
                    conf3 = float(fg3[m3].mean())
                    w3 *= conf3 ** k
            total = w2 + w3
            if total < 1e-6:
                final = np.zeros_like(m2)
                rule = "sigmoid (both empty)"
                w2 = w3 = 0.0
            else:
                w2, w3 = w2 / total, w3 / total
                p_final = w2 * fg2 + w3 * fg3
                final = p_final > 0.5
                rule = f"sig w2={w2:.2f} w3={w3:.2f}"

        gt_path = labels_dir / f"{cid}.nii.gz"
        if not gt_path.is_file():
            print(f"[warn] no GT for {cid} at {gt_path}; skipping metrics")
            continue
        gt = np.asarray(nib.load(str(gt_path)).dataobj) > 0

        d2v = dice(m2, gt)
        d3v = dice(m3, gt)
        dev = dice(me, gt)
        dgv = dice(final, gt)
        rows.append((cid, n2, n3, rule, d2v, d3v, dev, dgv))

        nib.save(
            nib.Nifti1Image(final.astype(np.uint8), img2.affine, img2.header),
            str(cases_dir / f"{cid}.nii.gz"),
        )
        pc = {
            "case": cid,
            "n_pred_2d": n2, "n_pred_3d": n3, "rule": rule,
            "dice_2d": d2v, "dice_3d": d3v,
            "dice_ens": dev, "dice_gated": dgv,
        }
        if args.gate_mode == "sigmoid":
            pc["w2"] = float(w2)
            pc["w3"] = float(w3)
            if args.use_confidence:
                pc["conf2"] = conf2
                pc["conf3"] = conf3
        per_case.append(pc)

    mean2 = _mean(r[4] for r in rows)
    mean3 = _mean(r[5] for r in rows)
    meane = _mean(r[6] for r in rows)
    meang = _mean(r[7] for r in rows)

    rule_width = max(15, *(len(r[3]) for r in rows)) if rows else 15
    print(f"{'case':8} {'n2d':>6} {'n3d':>6} {'rule':>{rule_width}} "
          f"{'2d':>6} {'3d':>6} {'ens':>6} {'gated':>6} {'Δens':>7}")
    print("-" * (63 + rule_width))
    for cid, n2, n3, rule, d2v, d3v, dev, dgv in rows:
        flag = "  <--" if dgv - dev >= 0.1 else ("  REG" if dgv - dev <= -0.05 else "")
        print(f"{cid:8} {n2:>6} {n3:>6} {rule:>{rule_width}} "
              f"{d2v:>6.3f} {d3v:>6.3f} {dev:>6.3f} {dgv:>6.3f} "
              f"{dgv-dev:>+7.3f}{flag}")
    print()
    print(f"Means (n={len(rows)}):  "
          f"2d={mean2:.4f}  3d={mean3:.4f}  "
          f"ens={meane:.4f}  GATED={meang:.4f}  Δens={meang-meane:+.4f}")

    summary = {
        "dataset": args.dataset,
        "trainer": args.trainer,
        "trainer_2d": trainer_2d,
        "trainer_3d": trainer_3d,
        "plans_2d": args.plans_2d, "config_2d": args.config_2d,
        "plans_3d": args.plans_3d, "config_3d": args.config_3d,
        "gate_mode": args.gate_mode,
        "min_fg_voxels": args.min_fg_voxels,
        "min_fg_ratio": args.min_fg_ratio,
        "tau": args.tau,
        "use_confidence": args.use_confidence,
        "conf_power": args.conf_power,
        "n_cases": len(rows),
        "mean_dice_2d": mean2,
        "mean_dice_3d": mean3,
        "mean_dice_ens": meane,
        "mean_dice_gated": meang,
        "per_case": per_case,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote] {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
