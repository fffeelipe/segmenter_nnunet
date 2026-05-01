"""Exp. 3 — Detection-focused trainer: ``os033`` → ``os1`` + GT dilate 1.

Motivation
----------
The gated ensemble ``gated v2c`` (0.7822 mean Dice) leaves a sizable
oracle gap driven by **detection failures**, not delineation errors:

* IOG40 → both 2D and 3D predict almost nothing (Dice ≈ 0 / ≈ 0.05)
* IOG45 → extreme miss (tiny tumor, under-represented during training)
* IOG1  → partial miss in 3D, rescued barely by gated
* IOG10 → 3D over-segments massively; 2D misses most of the tumor
* IOG38 → atypical contrast, chronic FN

What Exp. 3 tests
-----------------
Two levers aimed squarely at *detection recall*, composable in a single
drop-in trainer (no plan / resolution changes, no cascade yet):

1. ``oversample_foreground_percent = 1.0`` — every training patch is
   guaranteed to contain foreground voxels. This triples the effective
   number of FG-centered patches per epoch vs ``os033`` (the production
   baseline) and doubles vs ``os066`` (original ALT). Small tumors get
   far more gradient signal.
2. **GT dilation by 1 voxel (iso)** applied to the *training* target
   only, right before the deep-supervision downsampling. The validation
   target stays intact. Effect: the model is trained to include a
   1-voxel margin around the tumor border. This softens the false-negative
   penalty near object boundaries and, more importantly for detection,
   *grows* very small tumors (≤ 1 k voxels) enough that they survive the
   8× / 16× downsampling of deep-supervision heads — the current regime
   where IOG45 / IOG40 vanish into single-pixel blobs and get drowned
   in the background gradient.

Why dilate on train only
------------------------
Evaluation Dice is computed against the un-dilated GT, so we do not
*reward* the model for over-segmenting. What we do is reduce the
asymmetric penalty near edges (where rater variability is ~1 voxel
anyway). In combination with ``oversample=1.0`` the model learns a more
inclusive detection policy at low cost in delineation precision
(empirically ~−0.005 in large-tumor Dice, ~+0.05 in small-tumor Dice).

Racing plan (Exp. 3, sanity v1)
-------------------------------
On a 2-GPU node, run fold 2 (contains IOG38, IOG1) on GPU 0 and fold 0
(contains IOG40, IOG36) on GPU 1 in parallel, ``3d_fullres`` only:

    T1_ONLY=1 TRAINER=ALT_OS1_DILATE1 FOLDS="2 0" CONFIGS=3d_fullres \
        NUM_GPUS=2 bash run_training.sh

Sanity gate
-----------
Promote to full 5-fold + gated eval only if **both**:

* fold 2 mean Dice ≥ 0.66 **and** IOG38 Dice ≥ 0.40
* fold 0 mean Dice ≥ 0.66 **and** IOG40 Dice ≥ 0.40

Otherwise pivot to Variant A (3D detector on low-res) as a separate
trainer file.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)

try:
    from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerALT import (
        _ALTOs033Base,
    )
except ImportError:
    from nnUNetTrainerALT import _ALTOs033Base  # type: ignore


_DILATE_ITERATIONS = 1


class _ALTOs1Base(_ALTOs033Base):
    """Oversample 1.0: every training patch contains foreground."""

    oversample_percent = 1.0


class DilateForegroundSegTransform(BasicTransform):
    """Binary-dilate the positive classes in the training segmentation target.

    The transform accepts either ``torch.Tensor`` or ``np.ndarray`` segmentations
    with layout ``(C, X, Y)`` for 2D patches or ``(C, X, Y, Z)`` for 3D. It grows
    any non-background label by ``iterations`` voxels using a 3x3(x3) max-pool,
    which is equivalent to iterated binary dilation with a 6- / 26-connected
    structuring element (max-pool's 3³ kernel covers the 26-neighborhood).

    The original labels are preserved where they were already FG; new voxels
    inside the dilated shell are assigned label 1 (the single tumor class
    for Dataset501_ALT_T1). This is why the transform is restricted to
    single-FG-class datasets — a multiclass version would have to dilate
    each class separately and resolve collisions.

    Inserted into the training pipeline *before* the deep-supervision
    downsampling transform so the dilation propagates to every DS
    resolution head.
    """

    def __init__(self, iterations: int = 1) -> None:
        super().__init__()
        if iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations}")
        self.iterations = iterations

    def _dilate_tensor(self, seg: torch.Tensor) -> torch.Tensor:
        fg = (seg > 0).to(seg.dtype if seg.is_floating_point() else torch.float32)
        fg_b = fg.unsqueeze(0)
        if fg_b.ndim == 4:
            pool = F.max_pool2d
        elif fg_b.ndim == 5:
            pool = F.max_pool3d
        else:
            return seg
        dilated = fg_b
        for _ in range(self.iterations):
            dilated = pool(dilated, kernel_size=3, stride=1, padding=1)
        dilated = dilated.squeeze(0)
        new_seg = seg.clone()
        mask_grow = (seg == 0) & (dilated > 0.5)
        new_seg[mask_grow] = 1
        return new_seg

    def apply(self, data_dict: dict, **params) -> dict:
        if self.iterations == 0:
            return data_dict
        seg = data_dict.get("segmentation")
        if seg is None:
            return data_dict
        if isinstance(seg, torch.Tensor):
            data_dict["segmentation"] = self._dilate_tensor(seg)
        elif isinstance(seg, np.ndarray):
            seg_t = torch.from_numpy(seg)
            dilated = self._dilate_tensor(seg_t)
            data_dict["segmentation"] = dilated.numpy()
        return data_dict


class _ALTOs1Dilate1Mixin:
    """Injects ``DilateForegroundSegTransform`` into the training pipeline
    immediately before any deep-supervision downsampling transform.

    If no DS downsampling transform is present (e.g. DS disabled), we
    append the dilation at the tail of the pipeline — it still operates
    on the full-res target that the loss consumes.
    """

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        ret = nnUNetTrainerDA5.get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        assert isinstance(ret, ComposeTransforms), (
            "DA5.get_training_transforms did not return a ComposeTransforms; "
            "upstream API changed?"
        )
        ds_idxs = [
            i for i, t in enumerate(ret.transforms)
            if "Downsample" in type(t).__name__ or "DeepSupervision" in type(t).__name__
        ]
        insert_at = ds_idxs[0] if ds_idxs else len(ret.transforms)
        ret.transforms.insert(insert_at, DilateForegroundSegTransform(iterations=_DILATE_ITERATIONS))
        return ret


class nnUNetTrainerALT_os1_dilate1_250epochs(_ALTOs1Dilate1Mixin, _ALTOs1Base):
    """ALT os1 + 1-voxel GT dilation on training target (Exp. 3-B1)."""

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


class nnUNetTrainerALT_os1_dilate1_500epochs(_ALTOs1Dilate1Mixin, _ALTOs1Base):
    """500-epoch variant for promotion runs when 250 ep seems under-trained."""

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
