#!/usr/bin/env bash
# 03 — Correr 3 variantes de gated ensemble usando Exp. 3-B1 como 3D branch.
#
# El 2D branch se mantiene = baseline (nnUNetTrainerALT_os033_250epochs), ya
# que Exp. 3 solo cambia el 3D branch (detector de bordes con oversample=1.0
# + GT dilate 1 vox).
#
# Produce 3 ensembles para comparar contra gated_v2c_sigmoid (0.7822):
#
#   1) gated_exp3b1_hard    -> Semana-1 style (hard gate, v2b params)
#   2) gated_exp3b1_sigmoid -> Semana-2 style (sigmoid, tau=200)
#   3) gated_exp3b1_sigconf -> Semana-2 + confidence tiebreaker
#
# Uso:
#   bash scripts/wake/03_run_gated.sh
#
# Requiere env vars de nnU-Net (las inyecta este script).
set -euo pipefail

REPO=$(pwd)
export nnUNet_raw=$REPO/nnunet_env/nnUNet_raw
export nnUNet_preprocessed=$REPO/nnunet_env/nnUNet_preprocessed
export nnUNet_results=$REPO/nnunet_env/nnUNet_results

# Trainer del 2D branch: baseline. Override con TRAINER_2D=... si cambias.
TRAINER_2D=${TRAINER_2D:-nnUNetTrainerALT_os033_250epochs}
# Trainer del 3D branch: Exp. 3-B1.
TRAINER_3D=${TRAINER_3D:-nnUNetTrainerALT_os1_dilate1_250epochs}

COMMON=(
  --trainer-2d "$TRAINER_2D"
  --trainer-3d "$TRAINER_3D"
  --plans-2d nnUNetPlans
  --plans-3d nnUNetTSMRIPlans
)

echo "=== Check 2D + 3D dirs ==="
D2=$nnUNet_results/Dataset501_ALT_T1/${TRAINER_2D}__nnUNetPlans__2d
D3=$nnUNet_results/Dataset501_ALT_T1/${TRAINER_3D}__nnUNetTSMRIPlans__3d_fullres
for d in "$D2" "$D3"; do
  if [ ! -d "$d" ]; then
    echo "[ABORT] missing: $d"
    exit 1
  fi
done
for f in 0 1 2 3 4; do
  for d in "$D2/fold_$f/validation" "$D3/fold_$f/validation"; do
    if [ ! -d "$d" ]; then
      echo "[ABORT] missing validation dir: $d"
      exit 1
    fi
  done
done
echo "[OK]"
echo

echo "=== (1/3) hard gate (v2b style) ==="
python scripts/ensemble_gated.py "${COMMON[@]}" \
  --gate-mode hard --min-fg-voxels 50 --min-fg-ratio 0.40 \
  --out-name exp3b1_hard

echo
echo "=== (2/3) sigmoid gate (v2c style) ==="
python scripts/ensemble_gated.py "${COMMON[@]}" \
  --gate-mode sigmoid --min-fg-voxels 50 --tau 200 \
  --out-name exp3b1_sigmoid

echo
echo "=== (3/3) sigmoid + confidence ==="
python scripts/ensemble_gated.py "${COMMON[@]}" \
  --gate-mode sigmoid --min-fg-voxels 50 --tau 200 \
  --use-confidence --conf-power 2.0 \
  --out-name exp3b1_sigconf

echo
echo "========================================================"
echo "Ensembles escritos en:"
echo "  $nnUNet_results/Dataset501_ALT_T1/gated_ensemble_exp3b1_hard/"
echo "  $nnUNet_results/Dataset501_ALT_T1/gated_ensemble_exp3b1_sigmoid/"
echo "  $nnUNet_results/Dataset501_ALT_T1/gated_ensemble_exp3b1_sigconf/"
echo
echo "Siguiente paso -> bash scripts/wake/04_compare.sh"
echo "========================================================"
