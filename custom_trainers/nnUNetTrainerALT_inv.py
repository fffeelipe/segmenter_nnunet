"""Exp. 2b / 2b' trainers: ALT baseline + stronger inverted-contrast
supervision (two competing hypotheses).

Motivation
----------
Dataset501_ALT_T1 has a handful of atypical-contrast cases where the
tumor is *darker* than surrounding tissue (IOG38 with measured SNR
−0.06, and likely IOG1). The production trainer
(``nnUNetTrainerALT_os033_250epochs``) inherits from
``nnUNetTrainerDA5``, which already includes two
``GammaTransform(p_invert_image=1)`` blocks at p=0.1 each (~20 %
combined). Despite that, IOG38 stays at Dice 0.232 (2D) / 0.000 (3D)
and IOG1 at ~0.12 (gated v2c) — the gamma-with-inversion branch of DA5
is apparently not reaching enough batches to teach the model a true
contrast flip.

Two falsifiable variants race in parallel on separate GPUs:

``nnUNetTrainerALT_os033_inv_250epochs`` (**Exp. 2b**)
    Inserts an explicit ``InvertImageTransform(p=0.15)`` right after
    the ``SpatialTransform`` in the DA5 pipeline. The inversion used by
    ``batchgeneratorsv2`` is *mean-preserving* (``x -> 2*mean − x``),
    so it does not shift the z-score normalisation that nnU-Net applies
    upstream. Tests the hypothesis that the model needs a *linear* flip,
    not a gamma-on-flipped-image.

``nnUNetTrainerALT_os033_invgamma_250epochs`` (**Exp. 2b'**, control)
    Leaves the pipeline structure intact and only doubles the
    ``apply_probability`` of the two pre-existing
    ``RandomTransform(GammaTransform(p_invert_image=1))`` wrappers
    (0.1 -> 0.2 each, ~20 % -> ~40 %). Tests the simpler hypothesis
    that the existing gamma-flip branch works, it's just underexposed.

Racing both answers *whether* inverted-contrast supervision matters
**and** *whether the linear form matters*, in a single GPU-day of sanity.

Promotion criterion (see ANALYSIS.md §8.2, Exp. 2b)
---------------------------------------------------
Fold 2 sanity check (contains IOG38 and IOG1) must satisfy **both**:

* IOG38 Dice ≥ 0.45
* fold 2 mean Dice ≥ 0.63 (must not regress the os033 nopretrain
  baseline, which scored 0.6266 on fold 2)

If both pass on either variant, promote that variant to 5-fold training
and re-evaluate with the sigmoid gate (``gated v2c`` hyperparameters).
If both pass on both variants, pick the one with higher IOG38 Dice.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.inversion import InvertImageTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)

try:
    from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerALT import (
        _ALTOs033Base,
    )
except ImportError:
    # When this file is run from the workspace (before install_trainers.py
    # copied it into the nnunetv2 package), import the local module.
    from nnUNetTrainerALT import _ALTOs033Base  # type: ignore


_INVERT_PROB = 0.15

# Exp. 2b' control: target apply_probability for the two DA5 GammaTransforms
# that have ``p_invert_image=1``. DA5 defaults these to 0.1 each (~20 % of
# batches with a gamma-on-inverted-image). Doubling to 0.2 each (~40 %) is
# the cheapest way to push more inverted-contrast supervision through the
# existing pipeline without adding a new transform.
_GAMMA_INV_APPLY_PROB = 0.2


class _ALTOs033InvMixin:
    """Mixin that injects an ``InvertImageTransform`` right after the
    ``SpatialTransform`` in the DA5 training pipeline.

    The inversion is *mean-preserving* (``x -> 2*mean − x``), so it does
    not invalidate the z-score normalisation nnU-Net applies during
    preprocessing. We keep ``p_synchronize_channels=1`` and
    ``p_per_channel=1`` (single-modality dataset; always invert the
    whole image when the transform fires).
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
        sp_idx = np.where(
            [isinstance(t, SpatialTransform) for t in ret.transforms]
        )[0]
        assert len(sp_idx) == 1, (
            f"Expected exactly one SpatialTransform in DA5 pipeline, "
            f"found {len(sp_idx)}."
        )
        invert = InvertImageTransform(
            p_invert_image=_INVERT_PROB,
            p_synchronize_channels=1.0,
            p_per_channel=1.0,
        )
        ret.transforms.insert(int(sp_idx[0]) + 1, invert)
        return ret


class nnUNetTrainerALT_os033_inv_250epochs(_ALTOs033InvMixin, _ALTOs033Base):
    """ALT os033 + intensity-inversion augmentation (Exp. 2b)."""

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


class _ALTOs033InvGammaMixin:
    """Exp. 2b' (control): instead of inserting ``InvertImageTransform``,
    double the ``apply_probability`` of the two pre-existing
    ``GammaTransform(p_invert_image=1)`` wrappers in the DA5 pipeline
    (0.1 -> 0.2 each, ~20 % -> ~40 % combined).

    Rationale: DA5 already trains the model to tolerate inverted contrast
    via gamma-on-flipped-image, but at only ~20 % of batches. If the
    cheapest lever — just raising that rate — already rescues IOG38, then
    Exp. 2b's extra ``InvertImageTransform`` becomes unnecessary. This
    mixin lets 2b and 2b' race on separate GPUs to falsify the simpler
    hypothesis first.

    Side-effect-free: we do not add or remove transforms, only mutate the
    ``apply_probability`` attribute of the two matching ``RandomTransform``
    wrappers.
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
        assert isinstance(ret, ComposeTransforms)

        patched = 0
        for t in ret.transforms:
            if not isinstance(t, RandomTransform):
                continue
            inner = getattr(t, "transform", None)
            if isinstance(inner, GammaTransform) and float(inner.p_invert_image) == 1.0:
                t.apply_probability = _GAMMA_INV_APPLY_PROB
                patched += 1
        assert patched == 2, (
            f"Expected exactly 2 RandomTransform(Gamma, p_invert=1) in DA5 "
            f"pipeline, patched {patched}. Upstream DA5 changed?"
        )
        return ret


class nnUNetTrainerALT_os033_invgamma_250epochs(
    _ALTOs033InvGammaMixin, _ALTOs033Base
):
    """ALT os033 + DA5 gamma-inversion probability doubled (Exp. 2b')."""

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
