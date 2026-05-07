        ## `reports/` — regeneration notes (next sessions)

This folder contains the *human-facing* CSV used throughout `ANALYSIS.md`:

- `per_case_baseline_and_mixed.csv`

### What this CSV is

`per_case_baseline_and_mixed.csv` is a per-patient Dice table (one row per `IOGxx`),
combining:

- **DA5 baselines** (Dataset501, 2D and 3D)
- **Ensembles/gates** (Dataset501: soft-avg + gated v2b/v2c)
- **Fusion experiments** (Dataset503 T1+T2: `intersection`, and latest `union_v2`)

### Regenerate the CSV

From the repo root:

```bash
python scripts/update_per_case_baseline_and_mixed.py
```

It overwrites:

- `reports/per_case_baseline_and_mixed.csv`

### Column → source mapping (defaults)

The script is path-opinionated; it matches the layout described in `ANALYSIS.md`.

- `da5_2d`
  - `nnunet_env_base/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerDA5_100epochs__nnUNetPlans__2d/fold_{0..4}/validation/summary.json`
- `da5_3d`
  - `nnunet_env_base/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerDA5_100epochs__nnUNetTSMRIPlans__3d_fullres/fold_{0..4}/validation/summary.json`
- `softavg_2d3d_tta`
  - extracted from `dice_ens` in:
  - `nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_v2c_sigmoid/summary.json`
- `gated_v2b`
  - extracted from `dice_gated` in:
  - `nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_v2b_hard/summary.json`
- `gated_v2c`
  - extracted from `dice_gated` in:
  - `nnunet_env/nnUNet_results/Dataset501_ALT_T1/gated_ensemble_v2c_sigmoid/summary.json`
- `fusion_intersection_503_3d`
  - `nnunet_env_intersection/nnUNet_results/Dataset503_ALT_T1T2/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessed/summary.json`
- `fusion_union_v2_503_3d` (latest run)
  - fold 0:
    - `nnUNet_results_union_V2/Dataset503_ALT_T1T2/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0/postprocessed/summary.json`
  - folds 1–4:
    - `nnUNet_results_union_V2/Dataset503_ALT_T1T2/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres/crossval_results_folds_1_2_3_4/postprocessed/summary.json`
  - **Important**: the script averages `*_T1` and `*_T2` entries into a single patient row.

### Sanity checks / common pitfalls

- If you moved results folders or renamed trainers, update the paths in
  `scripts/update_per_case_baseline_and_mixed.py` (function `_default_inputs`).
- Dataset503 summaries may include `*_T1/_T2` entries (two masks per patient). The
  CSV is per-patient, so those are averaged.

