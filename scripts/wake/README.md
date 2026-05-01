# Wake-up runbook — Exp. 3-B1 evaluation

Scripts secuenciales para correr al despertar y evaluar Exp. 3-B1 (detección-segmentación cascade, Variante B1: oversample=1.0 + GT dilate 1 vox).

## Plan remoto que corre mientras duermes

- Nodo: `root@74.48.140.178 -p 52571` (2× RTX 5080)
- **Fase 1 (~5.6 h):** 4 folds entrenan en paralelo (2 por GPU)
  - GPU 0: fold 2 + fold 1
  - GPU 1: fold 0 + fold 3
- **Fase 2 (~2.1 h):** watchdog detecta fin de fase 1 y lanza **fold 4 con DDP 2-GPU**
- Flags del entrenamiento: `-pretrained_weights` (TotalSegMRI) + `--npz` (softmax para gated sigmoid)
- Trainer: `nnUNetTrainerALT_os1_dilate1_250epochs`
- Plans: `nnUNetTSMRIPlans` (bs=4, patch 96×96×128)
- **ETA total**: ~7.7 h desde las 19:25 UTC del lanzamiento

## Secuencia al despertar

```bash
cd /Users/luis.realpe/Documents/felipe/segment-med

# 1) Chequea estado del remote (ver epochs, dice, watchdog, 5 folds done?)
bash scripts/wake/01_check_remote.sh

# 2) Si los 5 folds terminaron -> descarga resultados (~1.5 GB)
bash scripts/wake/02_fetch_results.sh

# 3) Corre 3 variantes de gated ensemble (hard / sigmoid / sigmoid+conf)
bash scripts/wake/03_run_gated.sh

# 4) Compara cada variante vs gated_v2c_sigmoid (baseline 0.7822)
bash scripts/wake/04_compare.sh
```

## Si algún fold falló

Si `01_check_remote.sh` muestra un fold muerto/crasheado:

```bash
ssh -p 52571 root@74.48.140.178 'tail -50 /workspace/logs/exp3_b1_fold{0,1,2,3,4}.log'
```

Revisa el traceback y restart ese fold con `-c` (continua desde `checkpoint_latest.pth`):

```bash
ssh -p 52571 root@74.48.140.178 '
source /workspace/.venv/bin/activate
export nnUNet_raw=/workspace/nnunet_env/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnunet_env/nnUNet_preprocessed
export nnUNet_results=/workspace/nnunet_env/nnUNet_results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 nohup nnUNetv2_train 501 3d_fullres FOLD_ID \
  -tr nnUNetTrainerALT_os1_dilate1_250epochs \
  -p nnUNetTSMRIPlans \
  -pretrained_weights /workspace/nnunet_env/nnUNet_results/Dataset850_TotalSegMRI/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth \
  --npz -c > /workspace/logs/exp3_b1_fold_FOLD_ID.log 2>&1 &'
```

## Criterio de éxito

- **Objetivo**: superar `gated_v2c_sigmoid = 0.7822`
- **Hard-fail cases a monitorear**: IOG1, IOG10, IOG38, IOG40, IOG45 (tenían dice < 0.5 en baseline v2c)
- **Mejor variante** automáticamente elegida por delta de mean dice

Si alguna variante mejora, actualizar `ANALYSIS.md` con el resultado de Exp. 3-B1.
