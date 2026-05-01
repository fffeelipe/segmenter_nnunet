#!/usr/bin/env bash
# 04 — Comparar las 3 variantes de Exp. 3-B1 vs baseline gated_v2c_sigmoid.
#
# El baseline champion es gated_v2c_sigmoid (mean dice = 0.7822).
# Reporta delta mean + lista de casos donde cada variante ganó/perdió.
#
# Uso:
#   bash scripts/wake/04_compare.sh
set -euo pipefail

REPO=$(pwd)
BASELINE=$REPO/nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_v2c_sigmoid/summary.json

if [ ! -f "$BASELINE" ]; then
  echo "[ABORT] baseline not found: $BASELINE"
  exit 1
fi

for variant in exp3b1_hard exp3b1_sigmoid exp3b1_sigconf; do
  CAND=$REPO/nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_${variant}/summary.json
  if [ ! -f "$CAND" ]; then
    echo "[skip] $variant: summary.json not found"
    continue
  fi
  echo
  echo "========================================================"
  echo "  gated_v2c_sigmoid (0.7822)  vs  gated_${variant}"
  echo "========================================================"
  python scripts/compare_gated_summaries.py \
    --baseline "$BASELINE" \
    --candidate "$CAND" \
    --label-baseline "v2c_sig(0.7822)" \
    --label-candidate "exp3b1_${variant##*_}"
done

echo
echo "========================================================"
echo "Lectura rápida del resultado:"
echo
echo "  * Si alguna variante tiene Δ mean > 0 -> gana a baseline."
echo "  * Mira los casos hard-fail (IOG1, IOG10, IOG38, IOG40, IOG45)"
echo "    que eran la motivación de Exp. 3."
echo "  * El mejor ensemble gana automáticamente; si mejora, actualiza"
echo "    ANALYSIS.md con Exp. 3-B1."
echo "========================================================"
