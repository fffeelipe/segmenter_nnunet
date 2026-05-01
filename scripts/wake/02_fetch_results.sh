#!/usr/bin/env bash
# 02 — Descargar resultados de Exp. 3-B1 desde el remoto al laptop local.
#
# Baja por cada fold:
#   * checkpoint_final.pth    (~130 MB, para inferencia futura)
#   * validation/*.nii.gz     (~10-30 MB/caso, predicciones para ensemble)
#   * validation/*.npz        (softmax, requerido para --gate-mode sigmoid)
#   * validation/summary.json (métricas por fold)
#   * training_log_*.txt      (auditoría)
#   * progress.png            (loss curves)
#
# Total estimado: ~1.5 GB (5 folds × ~300 MB).
#
# Uso:
#   bash scripts/wake/02_fetch_results.sh
set -euo pipefail

REMOTE=${REMOTE:-root@74.48.140.178}
PORT=${PORT:-52571}

REMOTE_BASE=/workspace/nnunet_env/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerALT_os1_dilate1_250epochs__nnUNetTSMRIPlans__3d_fullres
LOCAL_BASE=$(pwd)/nnunet_env/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerALT_os1_dilate1_250epochs__nnUNetTSMRIPlans__3d_fullres

echo "=== Verificando que los 5 folds terminaron ==="
MISSING=$(ssh -p "$PORT" -o BatchMode=yes "$REMOTE" "
for f in 0 1 2 3 4; do
  CK=$REMOTE_BASE/fold_\$f/checkpoint_final.pth
  VAL=$REMOTE_BASE/fold_\$f/validation/summary.json
  if [ ! -f \"\$CK\" ] || [ ! -f \"\$VAL\" ]; then
    echo \"fold_\$f\"
  fi
done
")
if [ -n "$MISSING" ]; then
  echo "[ABORT] folds sin checkpoint_final.pth o summary.json:"
  echo "$MISSING" | sed 's/^/    /'
  echo
  echo "Espera a que terminen o corre scripts/wake/01_check_remote.sh para diagnosticar."
  exit 1
fi
echo "[OK] los 5 folds tienen checkpoint_final.pth + summary.json"
echo

mkdir -p "$LOCAL_BASE"

echo "=== rsync resultados (~1.5 GB) ==="
rsync -avhP \
  -e "ssh -p $PORT" \
  --exclude='checkpoint_best.pth' \
  --exclude='checkpoint_latest.pth' \
  --exclude='debug.json' \
  --include='*/' \
  --include='checkpoint_final.pth' \
  --include='dataset.json' \
  --include='dataset_fingerprint.json' \
  --include='plans.json' \
  --include='progress.png' \
  --include='training_log*.txt' \
  --include='fold_*/validation/**' \
  --exclude='*' \
  "$REMOTE:$REMOTE_BASE/" "$LOCAL_BASE/"

echo
echo "=== verificación local ==="
for f in 0 1 2 3 4; do
  CK=$LOCAL_BASE/fold_$f/checkpoint_final.pth
  SUM=$LOCAL_BASE/fold_$f/validation/summary.json
  NPZ=$(ls "$LOCAL_BASE/fold_$f/validation/"*.npz 2>/dev/null | wc -l | xargs)
  NII=$(ls "$LOCAL_BASE/fold_$f/validation/"*.nii.gz 2>/dev/null | wc -l | xargs)
  SZ=""
  [ -f "$CK" ] && SZ=$(du -h "$CK" | cut -f1)
  echo "fold_$f: ckpt=$SZ nii=$NII npz=$NPZ summary=$([ -f "$SUM" ] && echo yes || echo MISSING)"
done

echo
echo "========================================================"
echo "Siguiente paso -> bash scripts/wake/03_run_gated.sh"
echo "========================================================"
