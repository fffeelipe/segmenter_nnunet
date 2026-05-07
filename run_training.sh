#!/usr/bin/env bash
# ALT nnU-Net v2 end-to-end training script (T1 + T2 by default, T1-only opt-in).
#
# One-shot entrypoint: zip this folder, upload to a vast.ai machine with
# one RTX 4080 (16 GB), and run:
#
#   # Default: train both T1 and T2 with the baseline DA5 trainer
#   EPOCHS=1000 bash run_training.sh
#
#   # Quick smoke test (100 epochs, DA5 baseline)
#   EPOCHS=100  bash run_training.sh
#
#   # Train ONLY T1 (Dataset501) with baseline DA5:
#   T1_ONLY=1 EPOCHS=1000 bash run_training.sh
#
#   # Train ONLY T2 (Dataset502) with the current ALT_OS033 winner recipe
#   # on a 2x-GPU node (fold-parallel across 2 GPUs):
#   T2_ONLY=1 TRAINER=ALT_OS033 NUM_GPUS=2 bash run_training.sh
#
#   # Train ONLY T1 with the ALT minimal-plan trainer (250 epochs,
#   # oversample 0.66, batch_dice=False, stratified splits, plans patched):
#   T1_ONLY=1 TRAINER=ALT bash run_training.sh
#
#   # Train ONLY T1 with the ALT ABLATION trainer (same as ALT but
#   # oversample_foreground_percent=0.33, i.e. DA5 default). Isolates the
#   # effect of the heavy foreground oversampling from the longer
#   # schedule / batch_dice / clipping:
#   T1_ONLY=1 TRAINER=ALT_OS033 bash run_training.sh
#
#   # Exp. 2b — same as ALT_OS033 + intensity-inversion augmentation
#   # (p=0.15) targeting atypical-contrast cases (IOG38, IOG1):
#   T1_ONLY=1 TRAINER=ALT_OS033_INV bash run_training.sh
#   # Sanity fold 2 only (fold contains IOG38 and IOG1):
#   T1_ONLY=1 TRAINER=ALT_OS033_INV FOLDS=2 CONFIGS=2d bash run_training.sh
#
#   # Exp. 2b' (control) — same as ALT_OS033 but double the
#   # apply_probability of the two DA5 Gamma(p_invert=1) wrappers
#   # (0.1 -> 0.2 each). Race against ALT_OS033_INV on the other GPU:
#   T1_ONLY=1 TRAINER=ALT_OS033_INVGAMMA FOLDS=2 CONFIGS=2d bash run_training.sh
#
#   # Exp. 3-B1 — detection-focused trainer: oversample=1.0 +
#   # GT dilate 1 voxel on the training target (Dice denom softer at
#   # the border, small tumors grow enough to survive DS downsampling).
#   # Sanity: fold 2 + fold 0 in parallel on 2 GPUs, 3d_fullres only:
#   T1_ONLY=1 TRAINER=ALT_OS1_DILATE1 FOLDS="2 0" CONFIGS=3d_fullres \
#     NUM_GPUS=2 bash run_training.sh
#
#   # Dual-channel T1+T2 (Dataset503_ALT_T1T2). Builds a 2-channel dataset
#   # from T1/ + T2/ on disk (T2 resampled onto the T1 grid, masks fused),
#   # then trains it without TS-MRI pretraining (TS is single-channel).
#   # Default fusion is union; override with FUSION=intersection|staple.
#   T1T2=1 TRAINER=ALT_OS033 EPOCHS=250 NUM_GPUS=2 bash run_training.sh
#   T1T2=1 FUSION=intersection TRAINER=ALT_OS033 bash run_training.sh
#
# Optional env vars:
#   TRAINER           DA5 | ALT | ALT_OS033 | ALT_OS033_INV | ALT_OS033_INVGAMMA | ALT_OS1_DILATE1   (default DA5)
#                      DA5                -> nnUNetTrainerDA5_${EPOCHS}epochs                (legacy baseline)
#                      ALT                -> nnUNetTrainerALT_${EPOCHS}epochs               (minimal plan, oversample 0.66)
#                      ALT_OS033          -> nnUNetTrainerALT_os033_${EPOCHS}epochs         (ablation, oversample 0.33)
#                      ALT_OS033_INV      -> nnUNetTrainerALT_os033_inv_${EPOCHS}epochs      (Exp. 2b:  os033 + InvertImageTransform(p=0.15))
#                      ALT_OS033_INVGAMMA -> nnUNetTrainerALT_os033_invgamma_${EPOCHS}epochs (Exp. 2b': os033 + DA5 Gamma(p_invert=1) wrappers boosted to apply_prob=0.2)
#                      ALT_OS1_DILATE1    -> nnUNetTrainerALT_os1_dilate1_${EPOCHS}epochs   (Exp. 3-B1: oversample=1.0 + GT dilate 1 vox on train target)
#   EPOCHS            depends on TRAINER
#                      - DA5:                                                                 100 | 1000   (default 1000)
#                      - ALT, ALT_OS033, ALT_OS033_INV, ALT_OS033_INVGAMMA, ALT_OS1_DILATE1:  250 | 500    (default 250)
#   T1_ONLY           1 -> only convert T1/ and train Dataset501_ALT_T1
#   T2_ONLY           1 -> only convert T2/ and train Dataset502_ALT_T2
#   T1T2              1 -> build and train Dataset503_ALT_T1T2 (2 channels,
#                          T2 resampled onto T1 grid, masks fused).
#                          Mutually exclusive with T1_ONLY / T2_ONLY.
#   FUSION            union | intersection | staple  (default union;
#                          only used when T1T2=1 / DATASETS includes 503).
#   CLIP              0 | 1                       (default 0)
#                      1 -> apply percentile intensity clipping (p_lo, p_hi)
#                      after _fix_intensity. Writes to the clipped sibling
#                      datasets so the un-clipped baseline cache stays intact:
#                        T1_ONLY=1 + CLIP=1 -> Dataset505_ALT_T1_clip
#                        T2_ONLY=1 + CLIP=1 -> Dataset506_ALT_T2_clip
#                        T1T2=1   + CLIP=1 -> Dataset504_ALT_T1T2_clip
#   CLIP_P_LO         lower percentile for CLIP=1 (default 0.5)
#   CLIP_P_HI         upper percentile for CLIP=1 (default 99.5)
#   FOLDS             "0 1 2 3 4"               (default all 5 for 5-fold CV)
#   CONFIGS           "3d_fullres 2d"           (default both; find_best picks winner)
#   DATASETS          "501 502"                 (default both unless T1_ONLY=1 /
#                                                T2_ONLY=1 / T1T2=1)
#   STRATIFY_SPLITS   1 -> rewrite splits_final.json stratified by tumor vol.
#                     Default: 1 when TRAINER=ALT, else 0.
#   PATCH_PLANS       1 -> persist batch_dice=False on 2d in nnUNetPlans.json.
#                     Default: 1 when TRAINER=ALT, else 0.
#   TS_TASK_ID        852 (TotalSegmentator MRI task for pretraining)
#   NUM_GPUS          1 | 2                     (default 1)
#                      2 -> 2-slot work-stealing scheduler over the flat
#                      {DATASETS} x {CONFIGS} x {FOLDS} cartesian product.
#                      Each unit runs as its own process pinned to one GPU
#                      via CUDA_VISIBLE_DEVICES; each unit keeps its full
#                      batch_size (DDP with num_gpus=2 would halve batch_size
#                      and hurt 3d_fullres which has batch_size=2 -> 1 per
#                      GPU). Pairs across configs / datasets so a final solo
#                      fold within one config never leaves a GPU idle (a
#                      common 5-fold-per-config waste on the previous
#                      pairs-only scheduler). Requires bash 4.3+ (wait -n).
#   GPU_IDS           GPU indices for NUM_GPUS=2 (default "0 1")
#   PER_PROC_THREADS  cap on OMP/OpenBLAS/MKL threads per process (inherited
#                     by DA worker subprocesses). Default: 1 (scale CPU via
#                     DA_PROCS, not BLAS threads inside each worker). Set to 0
#                     to disable capping so libs autodetect.
#   DA_PROCS          cap on batchgenerators worker processes per training
#                     (maps to nnUNet_n_proc_DA). Default: physical CPU core
#                     count (lscpu; falls back to nproc). Override on NUM_GPUS=2
#                     nodes if two parallel trainings oversubscribe the host.
#   FORCE_T1T2_BUILD  1 -> always rerun scripts/build_t1t2_dataset.py when T1T2=1
#                     (clears nnUNet_raw/Dataset503_ALT_T1T2 and re-aligns all cases).
#                     Default 0: skip that step when dataset.json + imagesTr/*.nii.gz
#                     already exist and .fusion_mode matches FUSION (fast resume).
#   SKIP_COMPLETED_FOLDS  1 (default) -> if fold_* has checkpoint_final.pth, skip
#                     nnUNetv2_train for that fold (training already finished).
#                     Set to 0 to always invoke nnUNetv2_train (still uses --c when
#                     checkpoints exist; delete fold_* to start truly fresh).
#   PYTHON            python3
#   TORCH_CUDA        cu118|cu121|cu124|cu126|cu128 (default cu128; Blackwell/RTX 50 requires cu128)
#   CUDA_VISIBLE_DEVICES  honored when NUM_GPUS=1; ignored when NUM_GPUS=2
#                         (we override it per fold to pin each fold to a GPU)
#
#   Holdout / test inference (optional):
#   HOLDOUT_CASES_FILE   Path to a text file: one patient id per line (IOGxx),
#                        ``#`` comments. Those cases are written only under
#                        nnUNet_raw/Dataset.../holdout/{images,labels}/ and are
#                        excluded from training. Changing this file invalidates
#                        the matching nnUNet_preprocessed cache (same idea as
#                        FUSION stamp for 503). Path may be repo-relative.
#   RUN_HOLDOUT_EVAL     1 -> after find_best, run nnUNetv2_predict on holdout
#                        images (2d + 3d_fullres, all FOLDS) and scripts/ensemble_gated.py
#                        on flat prediction dirs (needs --save_probabilities).
#                        Default 0. Requires holdout/images under each dataset.
#   HOLDOUT_GATE_MODE    hard | sigmoid (default sigmoid, matches gated v2c).
#   HOLDOUT_VMIN         --min-fg-voxels for sigmoid gate (default 1000).
#   HOLDOUT_TAU          --tau for sigmoid (default 20).
#   HOLDOUT_USE_CONF     1 -> add --use-confidence for sigmoid (default 1).
#   HOLDOUT_MIN_RATIO    --min-fg-ratio when HOLDOUT_GATE_MODE=hard (default 0.1).
#   HOLDOUT_GATED_OUT_NAME  suffix folder under nnUNet_results/<ds>/gated_ensemble_<name>
#                        (default holdout_eval).
#
#   Example (T1, stratified train pool + locked test list):
#     HOLDOUT_CASES_FILE=holdout_cases.txt RUN_HOLDOUT_EVAL=1 \\
#       T1_ONLY=1 TRAINER=ALT_OS033 bash run_training.sh
#
#   Note: when ``holdout_cases.txt`` exists at the repo root, both
#   HOLDOUT_CASES_FILE and RUN_HOLDOUT_EVAL default to that file / to 1.
#   Pass ``HOLDOUT_CASES_FILE=`` (explicit empty) to opt out.
#
#   CV-time gated ensemble (v2c sigmoid + confidence):
#   RUN_CV_GATED         1 (default) -> after find_best, run scripts/ensemble_gated.py
#                        on per-fold validation outputs (single-channel datasets only;
#                        multichannel 503 keeps the existing holdout-only gating).
#                        Output dir suffix: gated_ensemble_<CV_GATED_OUT_NAME>/
#                        Set to 0 to skip.
#   CV_GATED_OUT_NAME    suffix folder under nnUNet_results/<ds>/gated_ensemble_<name>
#                        (default v2c_cv).
#
# Training resume: nnUNetv2_train only loads checkpoints when given --c
# (nnunetv2.run.run_training.maybe_load_checkpoint). This script passes --c when
# a fold has checkpoint_latest or checkpoint_best but not yet checkpoint_final;
# folds with checkpoint_final are skipped when SKIP_COMPLETED_FOLDS=1 (default).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

TRAINER_FAMILY="${TRAINER:-DA5}"
case "$TRAINER_FAMILY" in
  DA5)
    EPOCHS="${EPOCHS:-1000}"
    if [[ "$EPOCHS" != "100" && "$EPOCHS" != "1000" ]]; then
      echo "ERROR: with TRAINER=DA5, EPOCHS must be 100 or 1000 (got: $EPOCHS)" >&2
      exit 1
    fi
    TRAINER="nnUNetTrainerDA5_${EPOCHS}epochs"
    ;;
  ALT)
    EPOCHS="${EPOCHS:-250}"
    if [[ "$EPOCHS" != "250" && "$EPOCHS" != "500" ]]; then
      echo "ERROR: with TRAINER=ALT, EPOCHS must be 250 or 500 (got: $EPOCHS)" >&2
      exit 1
    fi
    TRAINER="nnUNetTrainerALT_${EPOCHS}epochs"
    ;;
  ALT_OS033)
    EPOCHS="${EPOCHS:-250}"
    if [[ "$EPOCHS" != "250" && "$EPOCHS" != "500" ]]; then
      echo "ERROR: with TRAINER=ALT_OS033, EPOCHS must be 250 or 500 (got: $EPOCHS)" >&2
      exit 1
    fi
    TRAINER="nnUNetTrainerALT_os033_${EPOCHS}epochs"
    ;;
  ALT_OS033_INV)
    EPOCHS="${EPOCHS:-250}"
    if [[ "$EPOCHS" != "250" ]]; then
      echo "ERROR: TRAINER=ALT_OS033_INV only ships a 250-epoch class (got: $EPOCHS). " \
           "See custom_trainers/nnUNetTrainerALT_inv.py — add a 500-epoch sibling there if needed." >&2
      exit 1
    fi
    TRAINER="nnUNetTrainerALT_os033_inv_${EPOCHS}epochs"
    ;;
  ALT_OS033_INVGAMMA)
    EPOCHS="${EPOCHS:-250}"
    if [[ "$EPOCHS" != "250" ]]; then
      echo "ERROR: TRAINER=ALT_OS033_INVGAMMA only ships a 250-epoch class (got: $EPOCHS). " \
           "See custom_trainers/nnUNetTrainerALT_inv.py — add a 500-epoch sibling there if needed." >&2
      exit 1
    fi
    TRAINER="nnUNetTrainerALT_os033_invgamma_${EPOCHS}epochs"
    ;;
  ALT_OS1_DILATE1)
    EPOCHS="${EPOCHS:-250}"
    if [[ "$EPOCHS" != "250" && "$EPOCHS" != "500" ]]; then
      echo "ERROR: with TRAINER=ALT_OS1_DILATE1, EPOCHS must be 250 or 500 (got: $EPOCHS)" >&2
      exit 1
    fi
    TRAINER="nnUNetTrainerALT_os1_dilate1_${EPOCHS}epochs"
    ;;
  *)
    echo "ERROR: unsupported TRAINER=$TRAINER_FAMILY. Use DA5, ALT, ALT_OS033, ALT_OS033_INV, ALT_OS033_INVGAMMA, or ALT_OS1_DILATE1." >&2
    exit 1
    ;;
esac

T1_ONLY="${T1_ONLY:-0}"
T2_ONLY="${T2_ONLY:-0}"
T1T2="${T1T2:-0}"
_mode_count=$(( (T1_ONLY==1) + (T2_ONLY==1) + (T1T2==1) ))
if (( _mode_count > 1 )); then
  echo "ERROR: T1_ONLY, T2_ONLY and T1T2 are mutually exclusive" >&2
  exit 1
fi
FOLDS="${FOLDS:-0 1 2 3 4}"
CONFIGS="${CONFIGS:-3d_fullres 2d}"

# Percentile-clipping A/B switch. When CLIP=1, the builders apply
# _clip_percentiles(p_lo, p_hi) after _fix_intensity and write to the clipped
# dataset ids (504/505/506) so the un-clipped baseline cache stays intact.
CLIP="${CLIP:-0}"
CLIP_P_LO="${CLIP_P_LO:-0.5}"
CLIP_P_HI="${CLIP_P_HI:-99.5}"
CLIP_BUILD_ARGS=()
if [[ "$CLIP" == "1" ]]; then
  CLIP_BUILD_ARGS=(--percentile-clip --clip-p-lo "$CLIP_P_LO" --clip-p-hi "$CLIP_P_HI")
fi

if [[ "$T1_ONLY" == "1" ]]; then
  if [[ "$CLIP" == "1" ]]; then
    DATASETS="${DATASETS:-505}"
  else
    DATASETS="${DATASETS:-501}"
  fi
elif [[ "$T2_ONLY" == "1" ]]; then
  if [[ "$CLIP" == "1" ]]; then
    DATASETS="${DATASETS:-506}"
  else
    DATASETS="${DATASETS:-502}"
  fi
elif [[ "$T1T2" == "1" ]]; then
  if [[ "$CLIP" == "1" ]]; then
    DATASETS="${DATASETS:-504}"
  else
    DATASETS="${DATASETS:-503}"
  fi
else
  if [[ "$CLIP" == "1" ]]; then
    DATASETS="${DATASETS:-505 506}"
  else
    DATASETS="${DATASETS:-501 502}"
  fi
fi
# Multichannel datasets cannot inherit TS-MRI plans (TS is single-channel).
# Training always uses default nnUNetPlans and no pretrained weights for those.
# 504 = percentile-clipped sibling of 503.
MULTICHANNEL_DATASETS_RE="^(503|504)$"
# Fusion mode for the T1+T2 dataset builder. Only used when dataset 503 is in DATASETS.
FUSION="${FUSION:-union}"
TS_TASK_ID="${TS_TASK_ID:-852}"
PYTHON="${PYTHON:-python3}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"

# Default to repo-root holdout_cases.txt when present. Use ${VAR-default}
# (no colon) so that an explicit `HOLDOUT_CASES_FILE=` opts out cleanly.
if [[ -z "${HOLDOUT_CASES_FILE+x}" && -f "$here/holdout_cases.txt" ]]; then
  HOLDOUT_CASES_FILE="holdout_cases.txt"
else
  HOLDOUT_CASES_FILE="${HOLDOUT_CASES_FILE-}"
fi
HOLDOUT_RESOLVED=""
HOLDOUT_EXCLUDE_ARGS=()
if [[ -n "$HOLDOUT_CASES_FILE" ]]; then
  if [[ "$HOLDOUT_CASES_FILE" != /* ]]; then
    HOLDOUT_RESOLVED="$here/$HOLDOUT_CASES_FILE"
  else
    HOLDOUT_RESOLVED="$HOLDOUT_CASES_FILE"
  fi
  if [[ ! -f "$HOLDOUT_RESOLVED" ]]; then
    echo "ERROR: HOLDOUT_CASES_FILE is not a file: $HOLDOUT_RESOLVED" >&2
    exit 1
  fi
  HOLDOUT_EXCLUDE_ARGS=(--exclude-cases-file "$HOLDOUT_RESOLVED")
fi
if [[ -n "$HOLDOUT_RESOLVED" ]]; then
  RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-1}"
else
  RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-0}"
fi
HOLDOUT_GATE_MODE="${HOLDOUT_GATE_MODE:-sigmoid}"
HOLDOUT_VMIN="${HOLDOUT_VMIN:-1000}"
HOLDOUT_TAU="${HOLDOUT_TAU:-20}"
HOLDOUT_USE_CONF="${HOLDOUT_USE_CONF:-1}"
HOLDOUT_MIN_RATIO="${HOLDOUT_MIN_RATIO:-0.1}"
HOLDOUT_GATED_OUT_NAME="${HOLDOUT_GATED_OUT_NAME:-holdout_eval}"

# v2c gated ensemble on per-fold CV outputs (default on).
RUN_CV_GATED="${RUN_CV_GATED:-1}"
CV_GATED_OUT_NAME="${CV_GATED_OUT_NAME:-v2c_cv}"

NUM_GPUS="${NUM_GPUS:-1}"
if [[ "$NUM_GPUS" != "1" && "$NUM_GPUS" != "2" ]]; then
  echo "ERROR: NUM_GPUS must be 1 or 2 (got: $NUM_GPUS)" >&2
  exit 1
fi
GPU_IDS_ARR=(${GPU_IDS:-0 1})
if [[ "$NUM_GPUS" == "2" && ${#GPU_IDS_ARR[@]} -lt 2 ]]; then
  echo "ERROR: NUM_GPUS=2 requires GPU_IDS with 2 ids (got: '${GPU_IDS:-}')" >&2
  exit 1
fi

# Per-process thread / worker budgets.
#
# nnU-Net uses batchgenerators which spawns `nnUNet_n_proc_DA` *processes*
# for data augmentation. Each worker is a fresh Python process that inherits
# env vars; high OPENBLAS_NUM_THREADS per worker multiplies across workers
# and oversubscribes the host. Default: OMP/BLAS = 2 thread per process,
# DA_PROCS = physical core count (not SMT). Override DA_PROCS when NUM_GPUS=2
# if two parallel trainings need a lower per-job worker count.
NCPU=$(nproc 2>/dev/null || echo 16)
NCPU_PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
(( NCPU_PHYS < 1 )) && NCPU_PHYS=$NCPU
_DEF_T=2
_DEF_DA=$NCPU_PHYS
PER_PROC_THREADS="${PER_PROC_THREADS:-$_DEF_T}"   # OMP / OpenBLAS / MKL cap
DA_PROCS="${DA_PROCS:-$_DEF_DA}"                  # batchgenerators workers

# torch.compile: default OFF. On fresh sm_120 (Blackwell / RTX 5080)
# Inductor adds 100s+ per epoch of compile time for the first few epochs
# and still yields <50% GPU util in steady-state for nnU-Net's dynamic
# patch shapes. Set NNUNET_COMPILE=1 to re-enable if you've benchmarked
# it on your specific node and it helps.
NNUNET_COMPILE="${NNUNET_COMPILE:-0}"
if [[ "$NNUNET_COMPILE" == "1" ]]; then
  _NNUNET_COMPILE_ENV="t"
else
  _NNUNET_COMPILE_ENV="f"
fi

case "$TRAINER_FAMILY" in
  ALT|ALT_OS033|ALT_OS033_INV|ALT_OS033_INVGAMMA|ALT_OS1_DILATE1)
    STRATIFY_SPLITS="${STRATIFY_SPLITS:-1}"
    PATCH_PLANS="${PATCH_PLANS:-1}"
    ;;
  *)
    STRATIFY_SPLITS="${STRATIFY_SPLITS:-0}"
    PATCH_PLANS="${PATCH_PLANS:-0}"
    ;;
esac

echo "=========================================================="
echo " ALT nnU-Net v2 training"
echo "   Trainer family: $TRAINER_FAMILY"
echo "   Trainer:        $TRAINER"
echo "   Epochs:         $EPOCHS"
echo "   T1_ONLY:        $T1_ONLY"
echo "   T2_ONLY:        $T2_ONLY"
echo "   T1T2:           $T1T2  (fusion=$FUSION)"
echo "   CLIP:           $CLIP  (p_lo=$CLIP_P_LO, p_hi=$CLIP_P_HI)"
echo "   Folds:          $FOLDS"
echo "   Configs:        $CONFIGS"
echo "   Datasets:       $DATASETS"
echo "   Stratify splits:$STRATIFY_SPLITS"
echo "   Patch 2d plans: $PATCH_PLANS"
echo "   TS MRI task:    $TS_TASK_ID"
echo "   NUM_GPUS:       $NUM_GPUS  (gpu ids: ${GPU_IDS_ARR[*]})"
echo "   CPU threads:    nproc=$NCPU phys=$NCPU_PHYS -> per-proc=$PER_PROC_THREADS, DA workers=$DA_PROCS"
echo "   torch.compile:  $_NNUNET_COMPILE_ENV  (NNUNET_COMPILE=$NNUNET_COMPILE)"
echo "   CV gated (v2c): $RUN_CV_GATED  (out_name=$CV_GATED_OUT_NAME)"
if [[ -n "$HOLDOUT_RESOLVED" ]]; then
  echo "   Holdout file:   $HOLDOUT_RESOLVED"
  echo "   Run holdout eval: $RUN_HOLDOUT_EVAL"
fi
echo "=========================================================="

# Try to bump the process/thread limit. In most containers this is capped
# and cannot be raised from userland, but it's harmless to try and often
# works on bare-metal / vast.ai nodes where the user is inside an
# unprivileged namespace.
( ulimit -u 65535 2>/dev/null || true )
( ulimit -n 65535 2>/dev/null || true )

# 1. Environment layout
export NNUNET_ROOT="$here/nnunet_env"
export nnUNet_raw="$NNUNET_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_ROOT/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# 2. Python venv + deps (skip if a marker file says deps are installed)
VENV="$here/.venv"
DEPS_MARKER="$VENV/.deps_installed"
if [[ ! -d "$VENV" ]]; then
  echo "[setup] creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

if [[ ! -f "$DEPS_MARKER" ]]; then
  echo "[setup] installing torch ($TORCH_CUDA) + nnU-Net + deps"
  case "$TORCH_CUDA" in
    cu128|cu126)
      # PyTorch >= 2.7 has Blackwell (sm_120) kernels in cu128.
      pip install --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
        "torch>=2.7" torchvision
      ;;
    cu124|cu121|cu118)
      # Older toolkits, no Blackwell support; pin to 2.6.x.
      pip install --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
        "torch>=2.4,<2.7" torchvision
      ;;
    *)
      echo "ERROR: unsupported TORCH_CUDA=$TORCH_CUDA. Use cu118|cu121|cu124|cu126|cu128." >&2
      exit 1
      ;;
  esac
  pip install -r requirements.txt
  python custom_trainers/install_trainers.py
  touch "$DEPS_MARKER"
else
  echo "[setup] deps already installed (remove $DEPS_MARKER to force reinstall)"
  python custom_trainers/install_trainers.py
fi

# Quick GPU/torch sanity check so a wrong wheel fails loudly here, not 30
# minutes into preprocessing.
python - <<'PY'
import torch
print(f"[setup] torch {torch.__version__}, CUDA {torch.version.cuda}, "
      f"cuDNN {torch.backends.cudnn.version()}")
if not torch.cuda.is_available():
    print("[setup] WARNING: torch.cuda.is_available() is False")
else:
    archs = torch.cuda.get_arch_list()
    print(f"[setup] torch arch list: {archs}")
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        sm = f"sm_{cap[0]}{cap[1]}"
        name = torch.cuda.get_device_name(i)
        ok = sm in archs
        print(f"[setup] GPU{i}: {name} {sm}  -> kernels available: {ok}")
        if not ok:
            print(f"[setup] WARNING: torch was not built for {sm}. "
                  f"Re-run with TORCH_CUDA=cu128 (Blackwell sm_120) "
                  f"after deleting $DEPS_MARKER.")
PY

# 3. Convert raw T1/T2 folders into nnU-Net datasets
# Skip when the raw dataset.json is already present AND we don't have the
# source T1/ folder (happens when migrating a preprocessed workspace to a
# fresh node: nnUNet_raw is rsync'd in, but the original T1/IOG* dirs
# stay on the source machine).
if [[ "$T1T2" == "1" ]]; then
  if [[ "$CLIP" == "1" ]]; then
    T1T2_DS_NAME="Dataset504_ALT_T1T2_clip"
  else
    T1T2_DS_NAME="Dataset503_ALT_T1T2"
  fi
  RAW_MARKER="$nnUNet_raw/$T1T2_DS_NAME/dataset.json"
  T1T2_RAW_DIR="$nnUNet_raw/$T1T2_DS_NAME"
  SRC_DIR="$here/T1"  # build_t1t2_dataset.py requires BOTH T1/ and T2/
  FORCE_T1T2_BUILD="${FORCE_T1T2_BUILD:-0}"
  T1T2_HAS_IMAGES=0
  if [[ -d "$T1T2_RAW_DIR/imagesTr" ]]; then
    shopt -s nullglob
    _t1t2_glob=("$T1T2_RAW_DIR/imagesTr"/*.nii.gz)
    shopt -u nullglob
    ((${#_t1t2_glob[@]} > 0)) && T1T2_HAS_IMAGES=1
  fi
  _t1t2_fusion_ok() {
    if [[ ! -f "$T1T2_RAW_DIR/.fusion_mode" ]]; then
      echo "[data] $T1T2_RAW_DIR/.fusion_mode missing -> writing FUSION=$FUSION (confirm this matches how labels were built)"
      echo "$FUSION" > "$T1T2_RAW_DIR/.fusion_mode"
      return 0
    fi
    if [[ "$(cat "$T1T2_RAW_DIR/.fusion_mode")" != "$FUSION" ]]; then
      echo "ERROR: FUSION=$FUSION but $T1T2_RAW_DIR/.fusion_mode says $(cat "$T1T2_RAW_DIR/.fusion_mode")." >&2
      echo "       Raw labels were built with a different fusion mode. Restore T1/ and T2/ and re-run the builder," >&2
      echo "       or delete $T1T2_RAW_DIR and rsync a consistent raw tree." >&2
      exit 1
    fi
    return 0
  }
  if [[ -f "$RAW_MARKER" && ! -d "$SRC_DIR" ]]; then
    echo "[data] nnUNet_raw already populated and T1/ source missing -> skipping build_t1t2_dataset.py"
    _t1t2_fusion_ok
  elif [[ "$FORCE_T1T2_BUILD" != "1" && -f "$RAW_MARKER" && "$T1T2_HAS_IMAGES" == "1" ]]; then
    _t1t2_fusion_ok
    echo "[data] Dataset503_ALT_T1T2 already built (imagesTr + dataset.json present, fusion=$FUSION) -> skipping build_t1t2_dataset.py"
    echo "[data] hint: set FORCE_T1T2_BUILD=1 after changing T1/ T2/ or FUSION to force a full rebuild."
  else
    if [[ "$FORCE_T1T2_BUILD" == "1" ]]; then
      echo "[data] FORCE_T1T2_BUILD=1 -> rebuilding 2-channel $T1T2_DS_NAME (fusion=$FUSION, clip=$CLIP)"
    else
      echo "[data] building 2-channel $T1T2_DS_NAME (fusion=$FUSION, clip=$CLIP)"
    fi
    python scripts/build_t1t2_dataset.py --fusion "$FUSION" \
      "${CLIP_BUILD_ARGS[@]}" "${HOLDOUT_EXCLUDE_ARGS[@]}"
    echo "$FUSION" > "$T1T2_RAW_DIR/.fusion_mode"
  fi
else
  CONVERT_ARGS=()
  if [[ "$T1_ONLY" == "1" ]]; then
    CONVERT_ARGS+=("--t1-only")
  elif [[ "$T2_ONLY" == "1" ]]; then
    CONVERT_ARGS+=("--t2-only")
  fi
  if [[ "$T2_ONLY" == "1" ]]; then
    if [[ "$CLIP" == "1" ]]; then
      RAW_MARKER="$nnUNet_raw/Dataset506_ALT_T2_clip/dataset.json"
    else
      RAW_MARKER="$nnUNet_raw/Dataset502_ALT_T2/dataset.json"
    fi
    SRC_DIR="$here/T2"
  else
    if [[ "$CLIP" == "1" ]]; then
      RAW_MARKER="$nnUNet_raw/Dataset505_ALT_T1_clip/dataset.json"
    else
      RAW_MARKER="$nnUNet_raw/Dataset501_ALT_T1/dataset.json"
    fi
    SRC_DIR="$here/T1"
  fi
  if [[ -f "$RAW_MARKER" && ! -d "$SRC_DIR" ]]; then
    echo "[data] nnUNet_raw already populated and $(basename "$SRC_DIR")/ source missing -> skipping convert_to_nnunet.py"
  else
    echo "[data] converting raw patient folders to nnU-Net raw datasets (clip=$CLIP)"
    python scripts/convert_to_nnunet.py "${CONVERT_ARGS[@]}" \
      "${CLIP_BUILD_ARGS[@]}" "${HOLDOUT_EXCLUDE_ARGS[@]}"
  fi
fi

# 4. Fetch TotalSegmentator MRI pretraining weights + plans
# Skip for T1T2 runs that only train multichannel datasets (TS is single-channel).
NEEDS_TS=0
for D in $DATASETS; do
  if ! [[ "$D" =~ $MULTICHANNEL_DATASETS_RE ]]; then
    NEEDS_TS=1
    break
  fi
done
if [[ "$NEEDS_TS" == "1" ]]; then
  if [[ -f "$nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_manifest.json" ]]; then
    echo "[pretrain] TotalSegMRI manifest present -> skipping fetch"
  else
    echo "[pretrain] fetching TotalSegmentator MRI weights (task $TS_TASK_ID)"
    python scripts/fetch_totalseg_mri.py --task "$TS_TASK_ID"
  fi
else
  echo "[pretrain] all target datasets are multichannel -> skipping TS-MRI fetch"
fi

# 5. Plan + preprocess (default plans for 2d + TS-compatible plans for 3d_fullres
# on single-channel datasets). Multichannel datasets (e.g. 503) only get
# nnUNetPlans; see prepare_pretrain_plans.py.
# Skip if plans + preprocessed shards are already on disk (migration case).
# Resolve `Dataset<id>_*` folder name by globbing nnUNet_raw. Centralises the
# dataset-id -> folder-name mapping so we don't enumerate 501/502/503/504/...
_ds_folder_name() {
  local D="$1" base="$2"
  shopt -s nullglob
  local m=("$base"/Dataset"${D}"_*)
  shopt -u nullglob
  if ((${#m[@]} != 1)); then
    return 1
  fi
  basename "${m[0]}"
}

PREPARE_ARGS=(--datasets $DATASETS)
PREP_DIR=""
PREP_NEEDS_TS=0
for D in $DATASETS; do
  if [[ "$D" =~ $MULTICHANNEL_DATASETS_RE ]]; then
    _ds_needs_ts=0
  else
    _ds_needs_ts=1
    PREP_NEEDS_TS=1
  fi
  if _name=$(_ds_folder_name "$D" "$nnUNet_raw"); then
    PREP_DIR="$nnUNet_preprocessed/$_name"
  fi
done
# Multichannel (503/504): if fusion mode changed since last preprocess, drop
# cache so nnUNetPlans.json and gt_segmentations are regenerated.
if [[ "$T1T2" == "1" && -d "$PREP_DIR" ]]; then
  _prep_fusion_stamp="$PREP_DIR/.fusion_mode"
  if [[ ! -f "$_prep_fusion_stamp" ]]; then
    echo "[plan] $PREP_DIR missing .fusion_mode -> removing preprocessed cache (fusion=$FUSION)"
    rm -rf "$PREP_DIR"
  elif [[ "$(cat "$_prep_fusion_stamp")" != "$FUSION" ]]; then
    echo "[plan] fusion changed ($(cat "$_prep_fusion_stamp") -> $FUSION) -> removing $PREP_DIR"
    rm -rf "$PREP_DIR"
  fi
fi
# If the holdout list file changes (or stamp missing while holdout is enabled),
# drop preprocessed cache for every dataset we train so splits/cases match raw.
if [[ -n "$HOLDOUT_RESOLVED" ]]; then
  _holdout_hash="$(sha256sum "$HOLDOUT_RESOLVED" | awk '{print $1}')"
  for D in $DATASETS; do
    if _name=$(_ds_folder_name "$D" "$nnUNet_raw"); then
      _hd_prep="$nnUNet_preprocessed/$_name"
    else
      continue
    fi
    if [[ -d "$_hd_prep" ]]; then
      if [[ ! -f "$_hd_prep/.holdout_stamp" ]] || [[ "$(cat "$_hd_prep/.holdout_stamp")" != "$_holdout_hash" ]]; then
        echo "[plan] holdout list new/changed vs $_hd_prep/.holdout_stamp -> removing $_hd_prep"
        rm -rf "$_hd_prep"
      fi
    fi
  done
fi
PREP_HAS_DEFAULT=0
if [[ -f "$PREP_DIR/nnUNetPlans.json" && -d "$PREP_DIR/nnUNetPlans_2d" ]]; then
  PREP_HAS_DEFAULT=1
fi
PREP_HAS_TS=0
if [[ -f "$PREP_DIR/nnUNetTSMRIPlans.json" ]]; then
  PREP_HAS_TS=1
fi
if [[ "$PREP_HAS_DEFAULT" == "1" && ( "$PREP_NEEDS_TS" == "0" || "$PREP_HAS_TS" == "1" ) ]]; then
  echo "[plan] plans + preprocessed shards already present -> skipping prepare_pretrain_plans.py"
else
  echo "[plan] planning + preprocessing datasets"
  python scripts/prepare_pretrain_plans.py "${PREPARE_ARGS[@]}"
  if [[ "$T1T2" == "1" && -d "$PREP_DIR" && -f "$nnUNet_raw/$T1T2_DS_NAME/.fusion_mode" ]]; then
    cp "$nnUNet_raw/$T1T2_DS_NAME/.fusion_mode" "$PREP_DIR/.fusion_mode"
    echo "[plan] stamped $PREP_DIR/.fusion_mode <- $(cat "$PREP_DIR/.fusion_mode")"
  fi
  if [[ -n "$HOLDOUT_RESOLVED" ]]; then
    _holdout_hash="$(sha256sum "$HOLDOUT_RESOLVED" | awk '{print $1}')"
    for D in $DATASETS; do
      if _name=$(_ds_folder_name "$D" "$nnUNet_raw"); then
        _dstamp="$nnUNet_preprocessed/$_name"
      else
        continue
      fi
      if [[ -d "$_dstamp" ]]; then
        echo "$_holdout_hash" > "$_dstamp/.holdout_stamp"
        echo "[plan] wrote $_dstamp/.holdout_stamp"
      fi
    done
  fi
fi

# 5b. (optional) stratify splits by tumor volume
if [[ "$STRATIFY_SPLITS" == "1" ]]; then
  for D in $DATASETS; do
    echo "[splits] writing stratified splits_final.json for dataset $D"
    python scripts/make_stratified_splits.py --dataset "$D"
  done
fi

# 5c. (optional) persist batch_dice=False on 2d in nnUNetPlans.json
if [[ "$PATCH_PLANS" == "1" ]]; then
  for D in $DATASETS; do
    echo "[plans] patching nnUNetPlans.json for dataset $D (batch_dice=False on 2d)"
    python scripts/patch_plans.py --dataset "$D" --plans nnUNetPlans
  done
fi

# 6. Extract pretrained checkpoint path from the manifest (only when any
# target dataset is single-channel, i.e. compatible with TS-MRI weights).
TS_CKPT=""
if [[ "$NEEDS_TS" == "1" ]]; then
  MANIFEST="$nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_manifest.json"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: missing pretrain manifest $MANIFEST" >&2
    exit 1
  fi
  TS_CKPT="$(python -c "import json,sys; print(json.load(open('$MANIFEST'))['default_checkpoint'])")"
  echo "[pretrain] using pretrained checkpoint: $TS_CKPT"
else
  echo "[pretrain] all target datasets are multichannel -> no TS checkpoint loaded"
fi

# 7. Train: for each dataset, each config, each fold
#
# With NUM_GPUS=1 we just loop sequentially. With NUM_GPUS=2 we launch
# two folds in parallel, each pinned to a distinct GPU via
# CUDA_VISIBLE_DEVICES, and wait on both before moving on. Per-fold stdout
# is prefixed with [gpu<id>] so interleaved output stays readable; the
# authoritative training logs still land in $nnUNet_results/.../fold_*/.
SKIP_COMPLETED_FOLDS="${SKIP_COMPLETED_FOLDS:-1}"

# Echo fold output dir (nnUNetTrainer.output_folder); return 1 if dataset path is ambiguous.
nnunet_fold_base() {
  local D="$1" PLANS="$2" C="$3" F="$4"
  shopt -s nullglob
  local matches=("$nnUNet_raw"/Dataset"${D}"_*)
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 ]]; then
    return 1
  fi
  echo "$nnUNet_results/$(basename "${matches[0]}")/${TRAINER}__${PLANS}__${C}/fold_${F}"
}

train_one() {
  local D="$1" C="$2" F="$3" PLANS="$4" GPU_ID="$5"
  shift 5
  local -a pretrain_args=("$@")
  local -a ckpt_args=()
  local fold_base=""
  if fold_base=$(nnunet_fold_base "$D" "$PLANS" "$C" "$F"); then
    :
  else
    fold_base=""
  fi

  if [[ -n "$fold_base" && -f "$fold_base/checkpoint_final.pth" ]]; then
    if [[ "$SKIP_COMPLETED_FOLDS" == "1" ]]; then
      echo "[train] fold_${F} already complete (checkpoint_final.pth) -> skipping nnUNetv2_train"
      echo "---- train: dataset=$D config=$C fold=$F plans=$PLANS trainer=$TRAINER gpu=$GPU_ID ----"
      return 0
    fi
    echo "[train] fold_${F} has checkpoint_final.pth but SKIP_COMPLETED_FOLDS=0 -> nnUNetv2_train --c"
    ckpt_args=(--c)
  elif [[ -n "$fold_base" && ( -f "$fold_base/checkpoint_latest.pth" || -f "$fold_base/checkpoint_best.pth" ) ]]; then
    echo "[train] fold_${F} has partial checkpoint -> nnUNetv2_train --c (cannot combine with -pretrained_weights)"
    ckpt_args=(--c)
  else
    ckpt_args=("${pretrain_args[@]}")
  fi
  echo "---- train: dataset=$D config=$C fold=$F plans=$PLANS trainer=$TRAINER gpu=$GPU_ID ----"
  # Cap BLAS / OpenMP threads and batchgenerators workers per process. If
  # PER_PROC_THREADS is 0 (single-GPU run), skip the caps and let the libs
  # autodetect.
  local -a envs=( "CUDA_VISIBLE_DEVICES=$GPU_ID" )
  if (( PER_PROC_THREADS > 0 )); then
    envs+=(
      "OMP_NUM_THREADS=$PER_PROC_THREADS"
      "OPENBLAS_NUM_THREADS=$PER_PROC_THREADS"
      "MKL_NUM_THREADS=$PER_PROC_THREADS"
      "NUMEXPR_NUM_THREADS=$PER_PROC_THREADS"
      "VECLIB_MAXIMUM_THREADS=$PER_PROC_THREADS"
    )
  fi
  if (( DA_PROCS > 0 )); then
    envs+=( "nnUNet_n_proc_DA=$DA_PROCS" )
  fi
  envs+=( "nnUNet_compile=$_NNUNET_COMPILE_ENV" )
  env "${envs[@]}" \
    nnUNetv2_train "$D" "$C" "$F" \
      -tr "$TRAINER" \
      -p "$PLANS" \
      "${ckpt_args[@]}" \
      --npz 2>&1 | sed -u "s/^/[gpu${GPU_ID} d=${D} c=${C} f=${F}] /"
  return "${PIPESTATUS[0]}"
}

# Build a flat queue of (dataset, config, fold, plans, pretrain_key) units across
# the full {DATASETS} x {CONFIGS} x {FOLDS} cartesian product. The queue lets us
# pair across configs / datasets so the scheduler stays full when |FOLDS| is
# odd: with NUM_GPUS=2 and the default 5 folds x 2 configs x 1 dataset = 10
# units, the pairing is always perfect (no GPU sits idle for a final solo fold).
JOBS=()
for D in $DATASETS; do
  D_MULTICHANNEL=0
  if [[ "$D" =~ $MULTICHANNEL_DATASETS_RE ]]; then
    D_MULTICHANNEL=1
  fi
  for C in $CONFIGS; do
    if [[ "$C" == "3d_fullres" && "$D_MULTICHANNEL" == "0" ]]; then
      JPLANS="nnUNetTSMRIPlans"
      JPK="ts"
    else
      JPLANS="nnUNetPlans"
      JPK="none"
    fi
    for F in $FOLDS; do
      JOBS+=("${D}|${C}|${F}|${JPLANS}|${JPK}")
    done
  done
done
echo "[sched] queued ${#JOBS[@]} training units across $DATASETS x $CONFIGS x folds=[$FOLDS]"

if [[ "$NUM_GPUS" == "1" ]]; then
  for job in "${JOBS[@]}"; do
    IFS='|' read -r D C F PLANS PK <<<"$job"
    if [[ "$PK" == "ts" ]]; then
      PRETRAIN_ARGS=(-pretrained_weights "$TS_CKPT")
    else
      PRETRAIN_ARGS=()
    fi
    echo ""
    train_one "$D" "$C" "$F" "$PLANS" "${GPU_IDS_ARR[0]}" "${PRETRAIN_ARGS[@]}"
  done
else
  # 2-slot work-stealing scheduler keyed by GPU id. Whenever both GPUs are
  # busy we wait for any one to finish (`wait -n`, bash 4.3+) and dispatch
  # the next queued unit on the freed GPU.
  declare -A SLOT_PID SLOT_DESC

  _drain_one() {
    # Block until ANY background slot finishes; reap every slot whose pid is
    # no longer alive; return non-zero if any reaped slot exited non-zero.
    set +e
    wait -n
    set -e
    local gpu p rc fail=0
    for gpu in "${!SLOT_PID[@]}"; do
      p="${SLOT_PID[$gpu]}"
      if ! kill -0 "$p" 2>/dev/null; then
        set +e; wait "$p"; rc=$?; set -e
        local desc="${SLOT_DESC[$gpu]}"
        unset "SLOT_PID[$gpu]"
        unset "SLOT_DESC[$gpu]"
        if (( rc != 0 )); then
          echo "ERROR: training failed: $desc (rc=$rc)" >&2
          fail=1
        fi
      fi
    done
    return $fail
  }

  _drain_remaining_and_die() {
    while (( ${#SLOT_PID[@]} > 0 )); do
      _drain_one || true
    done
    exit 1
  }

  _pick_free_gpu() {
    local i gpu
    for (( i=0; i<NUM_GPUS; i++ )); do
      gpu="${GPU_IDS_ARR[$i]}"
      if [[ -z "${SLOT_PID[$gpu]:-}" ]]; then
        echo "$gpu"
        return 0
      fi
    done
    return 1
  }

  for job in "${JOBS[@]}"; do
    IFS='|' read -r D C F PLANS PK <<<"$job"
    if [[ "$PK" == "ts" ]]; then
      PRETRAIN_ARGS=(-pretrained_weights "$TS_CKPT")
    else
      PRETRAIN_ARGS=()
    fi

    # Block until a GPU is free.
    while ! gpu=$(_pick_free_gpu); do
      _drain_one || _drain_remaining_and_die
    done

    echo ""
    train_one "$D" "$C" "$F" "$PLANS" "$gpu" "${PRETRAIN_ARGS[@]}" &
    SLOT_PID[$gpu]=$!
    SLOT_DESC[$gpu]="gpu${gpu} d=${D} c=${C} f=${F}"
  done

  # Drain whatever is still running.
  while (( ${#SLOT_PID[@]} > 0 )); do
    _drain_one || _drain_remaining_and_die
  done
fi

# Helper used by the find_best loop, the CV gated block, and the holdout block.
nnunet_one_raw_dataset_dir() {
  local D="$1"
  shopt -s nullglob
  local m=("$nnUNet_raw"/Dataset"${D}"_*)
  shopt -u nullglob
  if ((${#m[@]} != 1)); then
    return 1
  fi
  printf '%s' "${m[0]}"
}

# 8. Pick the best configuration (and emit inference commands) per dataset
for D in $DATASETS; do
  echo ""
  echo "---- find_best_configuration: dataset=$D ----"
  if [[ "$D" =~ $MULTICHANNEL_DATASETS_RE ]]; then
    FB_PLANS=(nnUNetPlans)
  else
    FB_PLANS=(nnUNetPlans nnUNetTSMRIPlans)
  fi
  nnUNetv2_find_best_configuration "$D" \
    -tr "$TRAINER" \
    -p "${FB_PLANS[@]}" \
    -c $CONFIGS \
    -f $FOLDS || true
done

# 8b. v2c gated ensemble on per-fold CV outputs (sigmoid + confidence).
# Single-channel datasets only: multichannel 503 keeps the holdout-only gating
# below since its 3D plans are nnUNetPlans (no TS-MRI), and the per-fold ensemble
# folder layout differs.
if [[ "$RUN_CV_GATED" == "1" ]]; then
  echo ""
  echo "---- v2c gated ensemble on CV (RUN_CV_GATED=1) ----"
  for D in $DATASETS; do
    if [[ "$D" =~ $MULTICHANNEL_DATASETS_RE ]]; then
      echo "[cv-gated] skip dataset $D (multichannel; gated path runs at holdout time only)"
      continue
    fi
    raw_ds="$(nnunet_one_raw_dataset_dir "$D")" || {
      echo "[cv-gated] WARN: skip $D: could not resolve nnUNet_raw/Dataset${D}_*"
      continue
    }
    ds_name="$(basename "$raw_ds")"
    echo "[cv-gated] dataset=$D ($ds_name) -> gated_ensemble_${CV_GATED_OUT_NAME}/"
    python scripts/ensemble_gated.py \
      --dataset "$ds_name" \
      --trainer "$TRAINER" \
      --plans-2d nnUNetPlans \
      --plans-3d nnUNetTSMRIPlans \
      --config-2d 2d \
      --config-3d 3d_fullres \
      --folds "$FOLDS" \
      --gate-mode sigmoid \
      --tau 20 \
      --min-fg-voxels 1000 \
      --use-confidence \
      --out-name "$CV_GATED_OUT_NAME" || \
      echo "[cv-gated] WARN: ensemble_gated.py failed for $ds_name (continuing)"
  done
fi

# 9. Optional holdout: nnUNetv2_predict (2d + 3d_fullres) + gated ensemble on flat dirs.
if [[ "$RUN_HOLDOUT_EVAL" == "1" ]]; then
  echo ""
  echo "---- holdout inference + gated eval (RUN_HOLDOUT_EVAL=1) ----"
  _PRED_FOLD_ARGS=(-f)
  for _pf in $FOLDS; do
    _PRED_FOLD_ARGS+=("$_pf")
  done
  for D in $DATASETS; do
    raw_ds="$(nnunet_one_raw_dataset_dir "$D")" || {
      echo "[holdout] WARN: skip $D: could not resolve nnUNet_raw/Dataset${D}_*"
      continue
    }
    hi="$raw_ds/holdout/images"
    hl="$raw_ds/holdout/labels"
    if [[ ! -d "$hi" ]]; then
      echo "[holdout] skip dataset $D: missing $hi"
      continue
    fi
    shopt -s nullglob
    _hn=("$hi"/*.nii.gz)
    shopt -u nullglob
    if ((${#_hn[@]} == 0)); then
      echo "[holdout] skip dataset $D: no *.nii.gz in $hi (set HOLDOUT_CASES_FILE and rebuild raw)."
      continue
    fi
    ds_name="$(basename "$raw_ds")"
    D_MULTICHANNEL=0
    if [[ "$D" =~ $MULTICHANNEL_DATASETS_RE ]]; then
      D_MULTICHANNEL=1
    fi
    if [[ "$D_MULTICHANNEL" == "1" ]]; then
      PLANS_3D_PRED="nnUNetPlans"
    else
      PLANS_3D_PRED="nnUNetTSMRIPlans"
    fi
    out_2d="$nnUNet_results/$ds_name/holdout_pred_2d"
    out_3d="$nnUNet_results/$ds_name/holdout_pred_3d"
    rm -rf "$out_2d" "$out_3d"
    mkdir -p "$out_2d" "$out_3d"
    gpu_pred="${GPU_IDS_ARR[0]}"
    _pred_env=( "CUDA_VISIBLE_DEVICES=$gpu_pred" )
    if (( PER_PROC_THREADS > 0 )); then
      _pred_env+=(
        "OMP_NUM_THREADS=$PER_PROC_THREADS"
        "OPENBLAS_NUM_THREADS=$PER_PROC_THREADS"
        "MKL_NUM_THREADS=$PER_PROC_THREADS"
        "NUMEXPR_NUM_THREADS=$PER_PROC_THREADS"
        "VECLIB_MAXIMUM_THREADS=$PER_PROC_THREADS"
      )
    fi
    _pred_env+=( "nnUNet_compile=$_NNUNET_COMPILE_ENV" )
    echo "[holdout] dataset=$D ($ds_name) predict 2d -> $out_2d"
    env "${_pred_env[@]}" nnUNetv2_predict \
      -d "$D" -i "$hi" -o "$out_2d" \
      -tr "$TRAINER" -c 2d -p nnUNetPlans \
      "${_PRED_FOLD_ARGS[@]}" --save_probabilities
    echo "[holdout] dataset=$D ($ds_name) predict 3d_fullres ($PLANS_3D_PRED) -> $out_3d"
    env "${_pred_env[@]}" nnUNetv2_predict \
      -d "$D" -i "$hi" -o "$out_3d" \
      -tr "$TRAINER" -c 3d_fullres -p "$PLANS_3D_PRED" \
      "${_PRED_FOLD_ARGS[@]}" --save_probabilities
    echo "[holdout] gated ensemble -> $nnUNet_results/$ds_name/gated_ensemble_${HOLDOUT_GATED_OUT_NAME}/"
    _gate_cmd=(
      python scripts/ensemble_gated.py
      --dataset "$ds_name"
      --trainer "$TRAINER"
      --pred-dir-2d "$out_2d"
      --pred-dir-3d "$out_3d"
      --labels-dir "$hl"
      --out-name "$HOLDOUT_GATED_OUT_NAME"
      --gate-mode "$HOLDOUT_GATE_MODE"
      --min-fg-voxels "$HOLDOUT_VMIN"
      --min-fg-ratio "$HOLDOUT_MIN_RATIO"
    )
    if [[ "$HOLDOUT_GATE_MODE" == "sigmoid" ]]; then
      _gate_cmd+=(--tau "$HOLDOUT_TAU")
      if [[ "$HOLDOUT_USE_CONF" == "1" ]]; then
        _gate_cmd+=(--use-confidence)
      fi
    fi
    "${_gate_cmd[@]}"
  done
fi

echo ""
echo "=========================================================="
echo " Training complete."
echo " Results under: $nnUNet_results"
echo " Use the inference command printed above by"
echo " nnUNetv2_find_best_configuration, or run manually, e.g.:"
if [[ "$T1T2" == "1" ]]; then
echo ""
echo "   # 2-channel: imagesTs must contain <case>_0000.nii.gz (T1) and"
echo "   # <case>_0001.nii.gz (T2 already resampled onto the T1 grid)."
echo "   nnUNetv2_predict -i <T1T2_images_dir> -o <out_dir> \\"
echo "     -d 503 -c 3d_fullres -tr $TRAINER -p nnUNetPlans \\"
echo "     -f $FOLDS"
else
echo ""
echo "   nnUNetv2_predict -i <T1_images_dir> -o <out_dir> \\"
echo "     -d 501 -c 3d_fullres -tr $TRAINER -p nnUNetTSMRIPlans \\"
echo "     -f $FOLDS"
if [[ "$T1_ONLY" != "1" ]]; then
echo ""
echo "   nnUNetv2_predict -i <T2_images_dir> -o <out_dir> \\"
echo "     -d 502 -c 3d_fullres -tr $TRAINER -p nnUNetTSMRIPlans \\"
echo "     -f $FOLDS"
fi
fi
echo "=========================================================="
