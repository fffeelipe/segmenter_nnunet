"""DA5 trainer variants with configurable epoch counts.

These subclasses only override ``num_epochs`` on top of
``nnUNetTrainerDA5`` (the heaviest-augmentation trainer shipped with
nnU-Net v2). Pattern matches ``nnUNetTrainerDA5_10epochs`` from the
upstream file ``nnUNetTrainerDA5.py``.
"""
from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)


class nnUNetTrainerDA5_100epochs(nnUNetTrainerDA5):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5_1000epochs(nnUNetTrainerDA5):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
