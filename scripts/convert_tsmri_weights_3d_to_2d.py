#!/usr/bin/env python3
"""Convert a TotalSegmentator-MRI 3D nnU-Net checkpoint into a 2D pretrained
checkpoint compatible with ``nnUNetv2_train ... -pretrained_weights <path>``.

Strategy (inflation-inverse):

- ``Conv3d``/``ConvTranspose3d`` weights of shape ``(out, in, kz, ky, kx)``
  are collapsed to ``(out, in, ky, kx)`` by averaging over the z-axis
  (``mean(dim=2)``). This is the standard "inverse inflation" used when
  transferring 3D video/volumetric kernels to 2D models.
- 1D norm parameters (``InstanceNorm*.weight``/``.bias``, any running
  stats) are copied verbatim when feature widths coincide.
- Segmentation heads (``.seg_layers.*``) are never transferred: nnU-Net
  skips them on load, and the label spaces differ (TS-MRI task 852 has
  51 classes, Dataset501_ALT_T1 has 2).

Architecture alignment (what actually transfers):

    Source 3d_fullres (TotalSegMRI task 852)     Target 2d (Dataset501)
        n_stages = 6                                n_stages = 7
        features = [32, 64, 128, 256, 320, 320]     features = [32, 64, 128, 256, 512, 512, 512]

    * Encoder stages 0..3 match by features (32, 64, 128, 256) and
      transfer directly.
    * Stages 4, 5 (+ target's extra stage 6) differ in width and are
      filled from the target network's own freshly-initialized weights.
    * Decoder: the source has 5 decoder stages, the target has 6. We
      align them by pyramid depth (skip the deepest extra target stage):
      ``target.decoder.stages.i  <-  source.decoder.stages.(i - 1)``
      for ``i >= 1`` with matching feature widths. In practice this
      transfers decoder stages operating at 256/128/64/32 features and
      the three transposed convs that live between them.

nnU-Net's ``load_pretrained_weights`` asserts that every non-seg_layer
key of the target network is present in the checkpoint *with the same
shape*. Therefore the output checkpoint contains **every** target key
(except seg_layers): converted values where possible, target's own
Kaiming-initialized tensors otherwise.

Partial-transfer modes (to mitigate negative transfer on the small
target dataset):

- ``--max-transfer-encoder-stage K``: shallow transfer. Only tensors
  living at pyramid levels ``0..K`` are pulled from the source; deeper
  stages fall back to the target's own init. Pyramid level is defined
  by the encoder stage index; decoder stages and transpconvs are
  mapped to the encoder level they read/write.

- ``--mix-ratio A``: soft warm-start. Every successfully-transferred
  tensor is blended with the target's init as ``A * pretrain + (1-A) *
  init`` (A=1.0 = full transfer, the default; A=0.0 = no transfer;
  A=0.5 = balanced). Applied *after* the stage gate, so both flags can
  be combined.

Usage
-----

    python scripts/convert_tsmri_weights_3d_to_2d.py \\
        --manifest      $nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_manifest.json \\
        --target-plans  $nnUNet_preprocessed/Dataset501_ALT_T1/nnUNetPlans.json \\
        --target-dataset-json $nnUNet_preprocessed/Dataset501_ALT_T1/dataset.json \\
        --target-config 2d \\
        --output        $nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_2d.pth

Then:

    nnUNetv2_train 501 2d $F \\
        -tr nnUNetTrainerALT_os033_250epochs \\
        -p  nnUNetPlans \\
        -pretrained_weights $nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_2d.pth \\
        --npz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import torch


SEG_LAYER_MARKER = ".seg_layers."
DECODER_STAGE_RE = re.compile(r"^(?P<prefix>.*decoder\.stages\.)(?P<idx>\d+)(?P<suffix>\..*)$")
DECODER_TRANSP_RE = re.compile(r"^(?P<prefix>.*decoder\.transpconvs\.)(?P<idx>\d+)(?P<suffix>\..*)$")

ENCODER_STAGE_RE = re.compile(r"(?:^|\.)encoder\.stages\.(\d+)\.")
DECODER_STAGE_ONLY_RE = re.compile(r"^decoder\.stages\.(\d+)\.")
DECODER_TRANSP_ONLY_RE = re.compile(r"^decoder\.transpconvs\.(\d+)\.")


def _env_msg() -> str:
    return (
        "This script must run inside an environment where nnunetv2 and "
        "dynamic_network_architectures are installed (same env used to "
        "train the 2D model)."
    )


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_target_network(target_plans: Path, target_dataset_json: Path,
                          target_config: str, seed: int,
                          deep_supervision: bool):
    """Instantiate the target 2D network from plans.json and dataset.json.

    Uses nnU-Net's own factory so the resulting state_dict key naming and
    Kaiming init perfectly match what ``nnUNetv2_train`` will build.
    """
    try:
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    except Exception as exc:  # pragma: no cover - only hit without nnU-Net installed
        raise SystemExit(f"{_env_msg()}\nImport error: {exc}") from exc

    plans_dict = _load_json(target_plans)
    dataset_json = _load_json(target_dataset_json)

    plans_manager = PlansManager(plans_dict)
    cfg_manager = plans_manager.get_configuration(target_config)
    label_manager = plans_manager.get_label_manager(dataset_json)

    channel_names = dataset_json.get("channel_names") or dataset_json.get("modality", {})
    if not channel_names:
        raise SystemExit("dataset.json is missing 'channel_names'")
    num_input_channels = len(channel_names)
    num_output_channels = label_manager.num_segmentation_heads

    torch.manual_seed(seed)
    net = get_network_from_plans(
        cfg_manager.network_arch_class_name,
        cfg_manager.network_arch_init_kwargs,
        cfg_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=deep_supervision,
    )
    return net, cfg_manager, num_input_channels, num_output_channels


def _resolve_source_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.resolve()
    if args.manifest is None:
        raise SystemExit("provide --manifest or --checkpoint")

    manifest = _load_json(args.manifest)
    if args.fold == "default":
        ckpt = manifest.get("default_checkpoint")
        if ckpt is None:
            raise SystemExit("manifest has no 'default_checkpoint' entry")
    else:
        by_fold = manifest.get("checkpoints_by_fold", {})
        if args.fold not in by_fold:
            raise SystemExit(
                f"fold {args.fold!r} not in manifest.checkpoints_by_fold "
                f"(available: {sorted(by_fold)})"
            )
        ckpt = by_fold[args.fold]
    return Path(ckpt).resolve()


def _load_source_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    if not ckpt_path.is_file():
        raise SystemExit(f"source checkpoint not found: {ckpt_path}")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(saved, dict) and "network_weights" in saved:
        return saved["network_weights"]
    if isinstance(saved, dict) and "state_dict" in saved:
        return saved["state_dict"]
    if isinstance(saved, dict) and all(isinstance(v, torch.Tensor) for v in saved.values()):
        return saved
    raise SystemExit(
        f"could not locate network weights in {ckpt_path}; expected keys "
        f"'network_weights' or 'state_dict'"
    )


def _reduce_to_target(src: torch.Tensor, target_shape: torch.Size) -> Optional[torch.Tensor]:
    """Try to adapt a source tensor to the target shape.

    Supported reductions:
      * identical shape          -> direct copy
      * 5D -> 4D (Conv3d/ConvT3d)-> mean over dim=2 (the z axis of the kernel)

    Returns ``None`` if the shapes are not compatible (different feature
    widths, for example).
    """
    tgt_shape = tuple(target_shape)
    if tuple(src.shape) == tgt_shape:
        return src.detach().clone()
    if src.dim() == 5 and len(tgt_shape) == 4:
        reduced = src.mean(dim=2)
        if tuple(reduced.shape) == tgt_shape:
            return reduced.detach().clone().contiguous()
    return None


def _n_stages(cfg_manager) -> int:
    kw = cfg_manager.network_arch_init_kwargs
    return int(kw["n_stages"])


def _key_pyramid_level(key: str, n_stages: int) -> Optional[int]:
    """Return the encoder pyramid level that a target key belongs to, or
    ``None`` if the key is not tied to a pyramid stage (e.g. seg layers
    that live outside the encoder/decoder hierarchy).

    Mapping (for an ``n_stages``-deep PlainConvUNet):
      * ``encoder.stages.<i>.*``            -> level i
      * ``decoder.encoder.stages.<i>.*``    -> level i (reference dup)
      * ``decoder.stages.<j>.*``            -> level ``n_stages - 2 - j``
      * ``decoder.transpconvs.<j>.*``       -> level ``n_stages - 2 - j``
    """
    m = ENCODER_STAGE_RE.search(key)
    if m:
        return int(m.group(1))
    m = DECODER_STAGE_ONLY_RE.match(key)
    if m:
        return (n_stages - 2) - int(m.group(1))
    m = DECODER_TRANSP_ONLY_RE.match(key)
    if m:
        return (n_stages - 2) - int(m.group(1))
    return None


def _remap_decoder_index(key: str, offset: int) -> Optional[str]:
    """If ``key`` names a decoder stage / transpconv, shift its index by
    ``-offset`` and return the remapped key. Returns None if the key isn't
    a decoder stage/transpconv or the shifted index is negative.
    """
    for regex in (DECODER_STAGE_RE, DECODER_TRANSP_RE):
        m = regex.match(key)
        if m is None:
            continue
        src_idx = int(m.group("idx")) - offset
        if src_idx < 0:
            return None
        return f"{m.group('prefix')}{src_idx}{m.group('suffix')}"
    return None


def _convert(target_sd: dict[str, torch.Tensor],
             source_sd: dict[str, torch.Tensor],
             decoder_offset: int,
             n_stages_tgt: int,
             max_transfer_level: Optional[int],
             mix_ratio: float,
             verbose: bool) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return (output_state_dict, stats)."""
    out: dict[str, torch.Tensor] = {}
    copied_direct: list[str] = []
    converted_z_mean: list[str] = []
    copied_via_decoder_align: list[str] = []
    filled_from_target_init: list[str] = []
    blocked_by_stage_gate: list[str] = []
    mixed_with_init: list[str] = []
    skipped_seg: list[str] = []
    shape_mismatch: list[tuple[str, tuple, tuple]] = []

    gate_active = max_transfer_level is not None
    do_mix = mix_ratio < 1.0

    for key, tgt in target_sd.items():
        if SEG_LAYER_MARKER in key:
            skipped_seg.append(key)
            continue

        used: Optional[torch.Tensor] = None
        label: Optional[str] = None

        # Shallow partial-transfer gate: refuse source-derived values for
        # pyramid levels deeper than `max_transfer_level`.
        lvl = _key_pyramid_level(key, n_stages_tgt) if gate_active else None
        gate_blocks = gate_active and lvl is not None and lvl > max_transfer_level

        if not gate_blocks and key in source_sd:
            reduced = _reduce_to_target(source_sd[key], tgt.shape)
            if reduced is not None:
                used = reduced
                label = "converted_z_mean" if reduced.shape != source_sd[key].shape else "copied_direct"
            else:
                shape_mismatch.append((key, tuple(source_sd[key].shape), tuple(tgt.shape)))

        if used is None and not gate_blocks and decoder_offset > 0:
            remapped = _remap_decoder_index(key, decoder_offset)
            if remapped is not None and remapped in source_sd:
                reduced = _reduce_to_target(source_sd[remapped], tgt.shape)
                if reduced is not None:
                    used = reduced
                    label = "copied_via_decoder_align"
                else:
                    shape_mismatch.append((f"{key} <- {remapped}", tuple(source_sd[remapped].shape), tuple(tgt.shape)))

        # Soft warm-start: blend the transferred tensor with the target's
        # fresh init. `mix_ratio == 0` is equivalent to "no transfer" for
        # this key.
        if used is not None and do_mix:
            if mix_ratio <= 0.0:
                used = None
            else:
                init_v = tgt.detach().clone().to(used.dtype)
                used = (mix_ratio * used) + ((1.0 - mix_ratio) * init_v)
                used = used.contiguous()
                mixed_with_init.append(key)

        if used is None:
            used = tgt.detach().clone()
            label = "filled_from_target_init"
            if gate_blocks:
                blocked_by_stage_gate.append(key)

        out[key] = used
        if label == "copied_direct":
            copied_direct.append(key)
        elif label == "converted_z_mean":
            converted_z_mean.append(key)
        elif label == "copied_via_decoder_align":
            copied_via_decoder_align.append(key)
        else:
            filled_from_target_init.append(key)

    stats: dict[str, Any] = {
        "n_target_keys": len(target_sd),
        "n_seg_layers_skipped": len(skipped_seg),
        "n_copied_direct": len(copied_direct),
        "n_converted_z_mean": len(converted_z_mean),
        "n_copied_via_decoder_align": len(copied_via_decoder_align),
        "n_filled_from_target_init": len(filled_from_target_init),
        "n_blocked_by_stage_gate": len(blocked_by_stage_gate),
        "n_mixed_with_init": len(mixed_with_init),
        "n_shape_mismatch": len(shape_mismatch),
        "max_transfer_encoder_stage": max_transfer_level,
        "mix_ratio": mix_ratio,
        "copied_direct": copied_direct,
        "converted_z_mean": converted_z_mean,
        "copied_via_decoder_align": copied_via_decoder_align,
        "filled_from_target_init": filled_from_target_init,
        "blocked_by_stage_gate": blocked_by_stage_gate,
        "mixed_with_init": mixed_with_init,
        "skipped_seg_layers": skipped_seg,
        "shape_mismatch_examples": shape_mismatch[:20],
    }
    if verbose:
        for group in ("copied_direct", "converted_z_mean", "copied_via_decoder_align"):
            for k in stats[group]:
                tag = " +mixed" if k in mixed_with_init else ""
                print(f"  [{group}{tag}] {k}  shape={tuple(out[k].shape)}")
        for k in filled_from_target_init:
            tag = " (stage-gated)" if k in blocked_by_stage_gate else ""
            print(f"  [filled_from_target_init{tag}] {k}  shape={tuple(out[k].shape)}")
    return out, stats


def _print_stats_header(stats: dict[str, Any]) -> None:
    total = stats["n_target_keys"] - stats["n_seg_layers_skipped"]
    pretrained = (stats["n_copied_direct"]
                  + stats["n_converted_z_mean"]
                  + stats["n_copied_via_decoder_align"])
    print("\n[convert] conversion summary")
    if stats.get("max_transfer_encoder_stage") is not None:
        print(f"  shallow transfer: max_transfer_encoder_stage = {stats['max_transfer_encoder_stage']}")
    if stats.get("mix_ratio", 1.0) < 1.0:
        print(f"  soft warm-start:  mix_ratio = {stats['mix_ratio']:.3f}")
    print(f"  target keys (excluding seg_layers): {total}")
    print(f"    - copied direct (shape match)     : {stats['n_copied_direct']}")
    print(f"    - converted Conv3d -> Conv2d      : {stats['n_converted_z_mean']}")
    print(f"    - copied via decoder alignment    : {stats['n_copied_via_decoder_align']}")
    print(f"    - filled from target random init  : {stats['n_filled_from_target_init']}")
    if stats.get("n_blocked_by_stage_gate", 0):
        print(f"      (of which {stats['n_blocked_by_stage_gate']} blocked by stage gate)")
    if stats.get("n_mixed_with_init", 0):
        print(f"    - mixed with init (soft warm)    : {stats['n_mixed_with_init']}")
    if stats["n_shape_mismatch"]:
        print(f"  shape mismatches encountered (fell back to init):")
        for k, ss, ts in stats["shape_mismatch_examples"]:
            print(f"    - {k}: src {ss} vs tgt {ts}")
    coverage = 100.0 * pretrained / max(total, 1)
    print(f"  pretrained coverage: {pretrained}/{total} ({coverage:.1f} %)")


def _sanity_forward(net: torch.nn.Module, target_config: str,
                    n_input_channels: int, cfg_manager) -> None:
    """Run a dummy forward pass on the converted network to catch any
    silent shape regressions. Uses a small patch to keep memory low.
    """
    patch_size = list(cfg_manager.patch_size)
    if target_config == "2d":
        dummy = torch.zeros((1, n_input_channels, patch_size[-2], patch_size[-1]))
    else:
        dummy = torch.zeros((1, n_input_channels, *patch_size))
    net.eval()
    with torch.no_grad():
        out = net(dummy)
    if isinstance(out, (list, tuple)):
        shapes = [tuple(o.shape) for o in out]
    else:
        shapes = [tuple(out.shape)]
    print(f"[convert] sanity forward OK. output shapes: {shapes}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--manifest", type=Path,
        help="Path to pretrain_manifest.json produced by fetch_totalseg_mri.py. "
             "Uses the 'default_checkpoint' entry unless --fold is given.",
    )
    src.add_argument(
        "--checkpoint", type=Path,
        help="Direct path to a TS-MRI 3d_fullres checkpoint (overrides --manifest).",
    )
    p.add_argument(
        "--fold", type=str, default="default",
        help="Manifest key: 'default', 'fold_0', ..., 'fold_4'. Ignored if --checkpoint is given.",
    )
    p.add_argument("--target-plans", type=Path, required=True,
                   help="Path to target dataset's nnUNetPlans.json (the 2D plan).")
    p.add_argument("--target-dataset-json", type=Path, required=True,
                   help="Path to target dataset's dataset.json (for label/channel info).")
    p.add_argument("--target-config", type=str, default="2d",
                   help="Target configuration name inside plans.json (default: 2d).")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the converted pretrained checkpoint (.pth).")
    p.add_argument("--no-decoder-align", action="store_true",
                   help="Disable pyramid-depth alignment for the decoder (only name-matching is used).")
    p.add_argument("--max-transfer-encoder-stage", type=int, default=None, metavar="K",
                   help="Shallow partial transfer: only copy weights living at pyramid "
                        "levels 0..K. Decoder stages/transpconvs that write to level >K "
                        "are blocked too (they fall back to target init). Default: no "
                        "restriction. Typical values: 1 (only the lowest features, "
                        "aggressive shallow), 2 (conservative shallow), 3 (matches the "
                        "encoder widths that overlap with TS-MRI so the default of "
                        "None already behaves like this for encoder; use explicit K to "
                        "also prune the decoder counterpart).")
    p.add_argument("--mix-ratio", type=float, default=1.0, metavar="A",
                   help="Soft warm-start blending factor: out = A * pretrain + (1-A) * "
                        "target_init. Applied to every successfully-transferred tensor. "
                        "A=1.0 (default) = full pretrain, A=0.0 = no pretrain, A=0.5 = "
                        "balanced. Combinable with --max-transfer-encoder-stage.")
    p.add_argument("--no-sanity-forward", action="store_true",
                   help="Skip the post-conversion dummy forward pass.")
    p.add_argument("--seed", type=int, default=12345,
                   help="Seed used to initialize the target network (for reproducible fallback inits).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and report the conversion, but do not write the output file.")
    p.add_argument("--verbose", action="store_true",
                   help="Print one line per key indicating how it was handled.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if not (0.0 <= args.mix_ratio <= 1.0):
        raise SystemExit(f"--mix-ratio must be in [0, 1] (got {args.mix_ratio})")
    if args.max_transfer_encoder_stage is not None and args.max_transfer_encoder_stage < 0:
        raise SystemExit("--max-transfer-encoder-stage must be >= 0")

    ckpt_path = _resolve_source_checkpoint(args)
    print(f"[convert] source checkpoint: {ckpt_path}")
    source_sd = _load_source_state_dict(ckpt_path)
    print(f"[convert] source state_dict: {len(source_sd)} tensors")

    net, cfg_manager, n_in, n_out = _build_target_network(
        target_plans=args.target_plans,
        target_dataset_json=args.target_dataset_json,
        target_config=args.target_config,
        seed=args.seed,
        deep_supervision=True,
    )
    target_sd = net.state_dict()
    print(
        f"[convert] target network: {cfg_manager.network_arch_class_name} "
        f"({args.target_config}), {len(target_sd)} tensors, "
        f"in={n_in} out={n_out}"
    )

    n_stages_tgt = _n_stages(cfg_manager)
    source_arch = _extract_source_n_stages(source_sd)
    if source_arch is None:
        print("[convert] warning: could not infer source n_stages from state_dict; "
              "decoder alignment disabled.")
        decoder_offset = 0
    else:
        decoder_offset = 0 if args.no_decoder_align else max(0, n_stages_tgt - source_arch)
    print(
        f"[convert] stages: target={n_stages_tgt} "
        f"source={source_arch if source_arch is not None else '?'} "
        f"-> decoder_offset={decoder_offset}"
    )

    out_sd, stats = _convert(
        target_sd=target_sd,
        source_sd=source_sd,
        decoder_offset=decoder_offset,
        n_stages_tgt=n_stages_tgt,
        max_transfer_level=args.max_transfer_encoder_stage,
        mix_ratio=args.mix_ratio,
        verbose=args.verbose,
    )
    _print_stats_header(stats)

    if not args.no_sanity_forward:
        missing, unexpected = net.load_state_dict(out_sd, strict=False)
        missing = [k for k in missing if SEG_LAYER_MARKER not in k]
        if missing:
            raise SystemExit(
                f"sanity check failed: target keys missing from converted "
                f"state_dict (not seg_layers): {missing[:8]}"
            )
        if unexpected:
            raise SystemExit(f"sanity check failed: unexpected keys: {unexpected[:8]}")
        _sanity_forward(net, args.target_config, n_in, cfg_manager)

    if args.dry_run:
        print("[convert] dry-run: not writing output.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "network_weights": out_sd,
        "conversion": {
            "source_checkpoint": str(ckpt_path),
            "target_plans": str(args.target_plans),
            "target_dataset_json": str(args.target_dataset_json),
            "target_config": args.target_config,
            "decoder_offset": decoder_offset,
            "max_transfer_encoder_stage": args.max_transfer_encoder_stage,
            "mix_ratio": args.mix_ratio,
            "seed": args.seed,
            "stats": {k: v for k, v in stats.items() if not isinstance(v, list) and not isinstance(v, tuple)},
        },
    }
    torch.save(payload, args.output)
    print(f"[convert] wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


def _extract_source_n_stages(source_sd: dict[str, torch.Tensor]) -> Optional[int]:
    """Infer the source encoder depth from its state_dict keys."""
    max_idx = -1
    enc_re = re.compile(r"^encoder\.stages\.(\d+)\.")
    for key in source_sd:
        m = enc_re.match(key)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    if max_idx < 0:
        return None
    return max_idx + 1


if __name__ == "__main__":
    sys.exit(main())
