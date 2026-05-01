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
#                      2 -> train two folds in parallel, one per GPU (fold-level
#                      parallelism via CUDA_VISIBLE_DEVICES). Recommended for
#                      2-GPU nodes because each fold keeps its full batch_size
#                      (DDP with num_gpus=2 would halve batch_size and hurt
#                      3d_fullres which has batch_size=2 -> 1 per GPU).
#   GPU_IDS           GPU indices for NUM_GPUS=2 (default "0 1")
#   PER_PROC_THREADS  cap on OMP/OpenBLAS/MKL threads per process (inherited
#                     by DA worker subprocesses). Default: 1 when NUM_GPUS=2
#                     (avoids oversubscription: DA workers multiply this by
#                     DA_PROCS). Set to 0 to disable capping on single-GPU
#                     runs so libs autodetect.
#   DA_PROCS          cap on batchgenerators worker processes per training
#                     (maps to nnUNet_n_proc_DA). This is the main CPU
#                     parallelism knob. Default on NUM_GPUS=2:
#                     min(16, (phys_cores - 2*NUM_GPUS) / NUM_GPUS)
#                     e.g. 14 on a 32-phys-core node, 10 on 24-phys-core,
#                     6 on 16-phys-core. Uses physical cores (not SMT threads)
#                     because batchgenerators workers are CPU-heavy numpy/scipy
#                     and two workers per physical core oversubscribe.
#   PYTHON            python3
#   TORCH_CUDA        cu118|cu121|cu124|cu126|cu128 (default cu128; Blackwell/RTX 50 requires cu128)
#   CUDA_VISIBLE_DEVICES  honored when NUM_GPUS=1; ignored when NUM_GPUS=2
#                         (we override it per fold to pin each fold to a GPU)
#
# Idempotent: nnU-Net auto-resumes from checkpoint_latest.pth when a fold
# is re-run, so it's safe to restart this script after interruption.

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
    if [[ "$EPOCHS" != "250" && "$EPOCHS" != "500" ]]; then
      echo "ERROR: with TRAINER=ALT_OS033_INV, EPOCHS must be 250 or 500 (got: $EPOCHS)" >&2
      exit 1
    fi
    TRAINER="nnUNetTrainerALT_os033_inv_${EPOCHS}epochs"
    ;;
  ALT_OS033_INVGAMMA)
    EPOCHS="${EPOCHS:-250}"
    if [[ "$EPOCHS" != "250" && "$EPOCHS" != "500" ]]; then
      echo "ERROR: with TRAINER=ALT_OS033_INVGAMMA, EPOCHS must be 250 or 500 (got: $EPOCHS)" >&2
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
if [[ "$T1_ONLY" == "1" ]]; then
  DATASETS="${DATASETS:-501}"
elif [[ "$T2_ONLY" == "1" ]]; then
  DATASETS="${DATASETS:-502}"
elif [[ "$T1T2" == "1" ]]; then
  DATASETS="${DATASETS:-503}"
else
  DATASETS="${DATASETS:-501 502}"
fi
# Multichannel datasets (e.g. Dataset503_ALT_T1T2) cannot inherit TS-MRI plans
# because TS is single-channel. Training always uses default nnUNetPlans and
# no pretrained weights for those.
MULTICHANNEL_DATASETS_RE="^(503)$"
# Fusion mode for the T1+T2 dataset builder. Only used when dataset 503 is in DATASETS.
FUSION="${FUSION:-union}"
TS_TASK_ID="${TS_TASK_ID:-852}"
PYTHON="${PYTHON:-python3}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"

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
# for data augmentation. Each augmentation worker is a fresh Python
# process that inherits env vars, so if we set OPENBLAS_NUM_THREADS=8
# that's actually 8 threads per worker (24 workers -> 192 BLAS threads
# per training). With 2 parallel trainings this oversubscribes even a
# 64-core node.
#
# Best practice:
#   - Set BLAS/OMP threads = 1 for ALL processes to avoid oversubscription.
#     Augmentation work is process-parallel, not thread-parallel, so this
#     does NOT reduce throughput.
#   - Set DA_PROCS high so augmentation scales with cores.
#   - Reserve ~4 cores for the main trainer / torch intra-op / IO.
#
# Heuristic defaults scale with *physical* cores and `NUM_GPUS`; override via env.
#
# We deliberately ignore SMT/hyperthreads here: batchgenerators workers are
# CPU-heavy numpy/scipy code, and two workers sharing one physical core via
# SMT fight over the same FPU/L2 rather than doubling throughput. Using
# `nproc` (= SMT threads) as the budget over-provisions on any SMT host and
# causes context-switch thrash when NUM_GPUS=2 runs two trainings in
# parallel.
NCPU=$(nproc 2>/dev/null || echo 16)
NCPU_PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
(( NCPU_PHYS < 1 )) && NCPU_PHYS=$NCPU
if [[ "$NUM_GPUS" == "2" ]]; then
  # Workers per training ~ (phys_cores - reserved_main) / NUM_GPUS, capped at 16.
  # Cap is lower than before (was 24 based on SMT threads) because diagnostics
  # on 32-phys-core nodes showed workers sitting at ~50% CPU with DA_PROCS=24
  # -> they were idling, and the extra processes added coordination overhead.
  _RESERVED=$(( 2 * NUM_GPUS ))
  _AVAIL=$(( NCPU_PHYS - _RESERVED ))
  (( _AVAIL < NUM_GPUS * 4 )) && _AVAIL=$(( NUM_GPUS * 4 ))
  _PER=$(( _AVAIL / NUM_GPUS ))
  # Empirically on 2x RTX 5080 + 32-phys-core Threadripper, DA_PROCS=14/proc
  # (28 workers total) oversubscribed and GPU util collapsed to <10%.
  # Cap at 10/proc (20 total, leaves 12 phys cores for main trainer + torch
  # intra-op + OS / FS IO). Lower cap also helps when torch.compile is off
  # (data pipeline is less bursty, doesn't need as many feeder procs).
  (( _PER > 10 )) && _PER=10
  (( _PER < 4 )) && _PER=4
  _DEF_T=1                # 1 thread per process; scale via DA_PROCS instead
  _DEF_DA=$_PER
else
  # Single-GPU. Autodetect is unsafe inside containers: OpenBLAS sees the
  # host's vCPU count (e.g. 64) and tries to spawn that many threads per
  # DA worker process, which instantly exhausts RLIMIT_NPROC and crashes
  # nnU-Net with "can't start new thread". Cap BLAS/OMP at 1 and pick a
  # conservative DA worker count (~half of phys cores, minus the trainer,
  # hard-capped at 8) so single-GPU runs work on vast.ai / Docker out of
  # the box. Override with DA_PROCS / PER_PROC_THREADS if you know the
  # host allows more.
  _AVAIL1=$(( NCPU_PHYS - 2 ))
  (( _AVAIL1 < 4 )) && _AVAIL1=4
  _PER1=$(( _AVAIL1 / 2 ))
  (( _PER1 > 8 )) && _PER1=8
  (( _PER1 < 4 )) && _PER1=4
  _DEF_T=1
  _DEF_DA=$_PER1
fi
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
echo "   Folds:          $FOLDS"
echo "   Configs:        $CONFIGS"
echo "   Datasets:       $DATASETS"
echo "   Stratify splits:$STRATIFY_SPLITS"
echo "   Patch 2d plans: $PATCH_PLANS"
echo "   TS MRI task:    $TS_TASK_ID"
echo "   NUM_GPUS:       $NUM_GPUS  (gpu ids: ${GPU_IDS_ARR[*]})"
echo "   CPU threads:    nproc=$NCPU phys=$NCPU_PHYS -> per-proc=$PER_PROC_THREADS, DA workers=$DA_PROCS"
echo "   torch.compile:  $_NNUNET_COMPILE_ENV  (NNUNET_COMPILE=$NNUNET_COMPILE)"
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
  RAW_MARKER="$nnUNet_raw/Dataset503_ALT_T1T2/dataset.json"
  T1T2_RAW_DIR="$nnUNet_raw/Dataset503_ALT_T1T2"
  SRC_DIR="$here/T1"  # build_t1t2_dataset.py requires BOTH T1/ and T2/
  if [[ -f "$RAW_MARKER" && ! -d "$SRC_DIR" ]]; then
    echo "[data] nnUNet_raw already populated and T1/ source missing -> skipping build_t1t2_dataset.py"
    if [[ ! -f "$T1T2_RAW_DIR/.fusion_mode" ]]; then
      echo "$FUSION" > "$T1T2_RAW_DIR/.fusion_mode"
      echo "[data] wrote $T1T2_RAW_DIR/.fusion_mode=$FUSION (first run; verify raw labels match)"
    elif [[ "$(cat "$T1T2_RAW_DIR/.fusion_mode")" != "$FUSION" ]]; then
      echo "ERROR: FUSION=$FUSION but $T1T2_RAW_DIR/.fusion_mode says $(cat "$T1T2_RAW_DIR/.fusion_mode")." >&2
      echo "       Raw labels were built with a different fusion mode. Restore T1/ and T2/ and re-run the builder," >&2
      echo "       or delete $T1T2_RAW_DIR and rsync a consistent raw tree." >&2
      exit 1
    fi
  else
    echo "[data] building 2-channel Dataset503_ALT_T1T2 (fusion=$FUSION)"
    python scripts/build_t1t2_dataset.py --fusion "$FUSION"
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
    RAW_MARKER="$nnUNet_raw/Dataset502_ALT_T2/dataset.json"
    SRC_DIR="$here/T2"
  else
    RAW_MARKER="$nnUNet_raw/Dataset501_ALT_T1/dataset.json"
    SRC_DIR="$here/T1"
  fi
  if [[ -f "$RAW_MARKER" && ! -d "$SRC_DIR" ]]; then
    echo "[data] nnUNet_raw already populated and $(basename "$SRC_DIR")/ source missing -> skipping convert_to_nnunet.py"
  else
    echo "[data] converting raw patient folders to nnU-Net raw datasets"
    python scripts/convert_to_nnunet.py "${CONVERT_ARGS[@]}"
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
PREPARE_ARGS=()
if [[ "$T1_ONLY" == "1" ]]; then
  PREPARE_ARGS+=("--t1-only")
elif [[ "$T2_ONLY" == "1" ]]; then
  PREPARE_ARGS+=("--t2-only")
elif [[ "$T1T2" == "1" ]]; then
  PREPARE_ARGS+=("--t1t2-only")
fi
if [[ "$T2_ONLY" == "1" ]]; then
  PREP_DIR="$nnUNet_preprocessed/Dataset502_ALT_T2"
  PREP_NEEDS_TS=1
elif [[ "$T1T2" == "1" ]]; then
  PREP_DIR="$nnUNet_preprocessed/Dataset503_ALT_T1T2"
  PREP_NEEDS_TS=0
else
  PREP_DIR="$nnUNet_preprocessed/Dataset501_ALT_T1"
  PREP_NEEDS_TS=1
fi
# Multichannel 503: if fusion mode changed since last preprocess, drop cache so
# nnUNetPlans.json and gt_segmentations are regenerated from the new labels.
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
  if [[ "$T1T2" == "1" && -d "$PREP_DIR" && -f "$nnUNet_raw/Dataset503_ALT_T1T2/.fusion_mode" ]]; then
    cp "$nnUNet_raw/Dataset503_ALT_T1T2/.fusion_mode" "$PREP_DIR/.fusion_mode"
    echo "[plan] stamped $PREP_DIR/.fusion_mode <- $(cat "$PREP_DIR/.fusion_mode")"
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
train_one() {
  local D="$1" C="$2" F="$3" PLANS="$4" GPU_ID="$5"
  shift 5
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
      "$@" \
      --npz 2>&1 | sed -u "s/^/[gpu${GPU_ID} d=${D} c=${C} f=${F}] /"
  return "${PIPESTATUS[0]}"
}

for D in $DATASETS; do
  D_MULTICHANNEL=0
  if [[ "$D" =~ $MULTICHANNEL_DATASETS_RE ]]; then
    D_MULTICHANNEL=1
  fi
  for C in $CONFIGS; do
    if [[ "$C" == "3d_fullres" && "$D_MULTICHANNEL" == "0" ]]; then
      PLANS="nnUNetTSMRIPlans"
      PRETRAIN_ARGS=(-pretrained_weights "$TS_CKPT")
    else
      # 2d config, or any config on a multichannel dataset.
      PLANS="nnUNetPlans"
      PRETRAIN_ARGS=()
    fi

    FOLDS_ARR=($FOLDS)
    if [[ "$NUM_GPUS" == "1" ]]; then
      for F in "${FOLDS_ARR[@]}"; do
        echo ""
        train_one "$D" "$C" "$F" "$PLANS" "${GPU_IDS_ARR[0]}" "${PRETRAIN_ARGS[@]}"
      done
    else
      # Pair folds two at a time, one per GPU.
      for (( i=0; i<${#FOLDS_ARR[@]}; i+=2 )); do
        F1="${FOLDS_ARR[i]}"
        F2="${FOLDS_ARR[i+1]:-}"
        echo ""
        train_one "$D" "$C" "$F1" "$PLANS" "${GPU_IDS_ARR[0]}" "${PRETRAIN_ARGS[@]}" &
        PID1=$!
        if [[ -n "$F2" ]]; then
          train_one "$D" "$C" "$F2" "$PLANS" "${GPU_IDS_ARR[1]}" "${PRETRAIN_ARGS[@]}" &
          PID2=$!
          set +e
          wait "$PID1"; RET1=$?
          wait "$PID2"; RET2=$?
          set -e
          if (( RET1 != 0 || RET2 != 0 )); then
            echo "ERROR: parallel training failed (gpu${GPU_IDS_ARR[0]} fold $F1 -> $RET1, gpu${GPU_IDS_ARR[1]} fold $F2 -> $RET2)" >&2
            exit 1
          fi
        else
          # Odd fold left over, run alone on the first GPU.
          set +e
          wait "$PID1"; RET1=$?
          set -e
          if (( RET1 != 0 )); then
            echo "ERROR: training failed (gpu${GPU_IDS_ARR[0]} fold $F1 -> $RET1)" >&2
            exit 1
          fi
        fi
      done
    fi
  done
done

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
