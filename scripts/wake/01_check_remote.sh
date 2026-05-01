#!/usr/bin/env bash
# 01 — Al despertar: chequear estado del entrenamiento Exp. 3-B1 en el remoto.
#
# Reporta:
#   * estado del watchdog
#   * progreso por fold (epochs, Pseudo dice, errores)
#   * GPU util actual
#   * si los 5 folds ya terminaron -> OK para ejecutar 02_fetch_results.sh
#
# Uso:
#   bash scripts/wake/01_check_remote.sh
set -euo pipefail

REMOTE=${REMOTE:-root@74.48.140.178}
PORT=${PORT:-52571}

ssh -p "$PORT" -o BatchMode=yes "$REMOTE" '
set +e
BASE=/workspace/nnunet_env/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerALT_os1_dilate1_250epochs__nnUNetTSMRIPlans__3d_fullres

echo "=== watchdog ==="
tail -15 /workspace/logs/watchdog.log 2>/dev/null || echo "(no watchdog log)"
echo

echo "=== GPU state ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv
echo

echo "=== proc table ==="
for f in 0 1 2 3 4; do
  PIDF=/workspace/logs/pid_fold$f.txt
  if [ -f "$PIDF" ]; then
    PID=$(cat "$PIDF")
    STATE=$(ps -p "$PID" -o pid,etime,stat 2>/dev/null | tail -1 | xargs)
    if [ -n "$STATE" ]; then
      echo "fold_$f: $STATE (alive)"
    else
      echo "fold_$f: pid=$PID DEAD"
    fi
  else
    echo "fold_$f: no pid file (not launched yet)"
  fi
done
echo

echo "=== fold progress (last 2 epochs + val) ==="
for f in 0 1 2 3 4; do
  LOG=/workspace/logs/exp3_b1_fold$f.log
  if [ -f "$LOG" ]; then
    echo "--- fold_$f ---"
    grep -E "Epoch [0-9]|Pseudo dice|Traceback|RuntimeError|out of memory|CUDA error|Training done|Validation|Now performing" "$LOG" | tail -8
  fi
done
echo

echo "=== checkpoints final ==="
for f in 0 1 2 3 4; do
  CK=$BASE/fold_$f/checkpoint_final.pth
  VAL=$BASE/fold_$f/validation/summary.json
  if [ -f "$CK" ]; then
    SIZE=$(du -h "$CK" | cut -f1)
    VALTAG=""
    [ -f "$VAL" ] && VALTAG=" + val/summary.json"
    echo "fold_$f: checkpoint_final.pth ($SIZE)$VALTAG"
  else
    echo "fold_$f: (training in progress, no checkpoint_final yet)"
  fi
done
echo

echo "=== val summary means (if available) ==="
for f in 0 1 2 3 4; do
  SUM=$BASE/fold_$f/validation/summary.json
  if [ -f "$SUM" ]; then
    python3 -c "
import json, sys
d = json.load(open(\"$SUM\"))
fg = d.get(\"foreground_mean\", {})
dice = fg.get(\"Dice\") if isinstance(fg, dict) else None
print(f\"  fold_$f  mean_dice = {dice:.4f}\" if dice else f\"  fold_$f  (summary present, no Dice field)\")
" 2>/dev/null || echo "  fold_$f  (cannot parse summary)"
  fi
done
'

echo
echo "========================================================"
echo "Si todos los 5 folds tienen checkpoint_final.pth + val/summary.json:"
echo "  siguiente paso -> bash scripts/wake/02_fetch_results.sh"
echo
echo "Si algún fold sigue corriendo:"
echo "  espera y re-corre este script en ~30min"
echo "========================================================"
