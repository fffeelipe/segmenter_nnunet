"""Custom trainers tuned for Dataset501_ALT_T1 (small FN-dominated lesions).

Two families live here:

``nnUNetTrainerALT_{250,500}epochs`` (the "minimal plan" trainer)
    Differences vs. ``nnUNetTrainerDA5`` (the baseline):
    1. ``num_epochs`` extended to 250 or 500 (the 100-epoch run was still
       improving; pseudo-Dice EMA ~0.82 on fold 0).
    2. ``oversample_foreground_percent = 0.66`` — double the fraction of
       foreground-centered patches per batch (DA5 default: 0.33). Aimed
       at rescuing tiny tumors (IOG45, IOG29, IOG38, IOG40).
    3. ``batch_dice = False`` forced through the configuration manager so
       per-sample Dice is used in the composite loss (relevant for the
       2d plans; 3d_fullres TS-MRI plans already default to False).

``nnUNetTrainerALT_os033_{250,500}epochs`` (ablation trainer)
    Same as above but keeps ``oversample_foreground_percent = 0.33``
    (DA5 default). Used to isolate the effect of the heavy foreground
    oversampling from the longer training schedule. If this ablation
    closes the gap vs. the DA5 baseline on regressed cases (IOG36, IOG4,
    IOG24, IOG47 without the intensity-fix artifact), then the 0.66
    oversample is the variable to tune down.

Keeps everything else from DA5 (heavy spatial / intensity augmentation).
"""
from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)


class _ALTBase(nnUNetTrainerDA5):
    """Base class for the minimal-plan trainer family.

    Subclasses set ``num_epochs``; this base:
    - Forces ``oversample_foreground_percent = 0.66``.
    - Forces ``batch_dice = False`` in the configuration manager.
    """

    oversample_percent = 0.66

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = self.oversample_percent
        cm_cfg = getattr(self.configuration_manager, "configuration", None)
        if isinstance(cm_cfg, dict):
            cm_cfg["batch_dice"] = False


class _ALTOs033Base(_ALTBase):
    """Ablation base: same as ``_ALTBase`` but with default DA5 oversample (0.33)."""

    oversample_percent = 0.33


class nnUNetTrainerALT_250epochs(_ALTBase):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerALT_500epochs(_ALTBase):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainerALT_os033_250epochs(_ALTOs033Base):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerALT_os033_500epochs(_ALTOs033Base):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
