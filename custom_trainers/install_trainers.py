#!/usr/bin/env python3
"""Install all custom nnU-Net v2 trainer modules into the package's discovery path.

nnU-Net v2 discovers trainers with ``recursive_find_python_class`` walking
the physical directory ``nnunetv2/training/nnUNetTrainer``. So the simplest
reliable way to make our custom trainers visible to the CLI
(``nnUNetv2_train -tr ...``) is to copy each trainer module into the
``variants/data_augmentation/`` folder right after ``pip install nnunetv2``.

This script:
- Locates the installed ``nnunetv2`` package.
- Copies every ``nnUNetTrainer*.py`` file in this directory (except
  this installer itself) into
  ``<nnunetv2>/training/nnUNetTrainer/variants/data_augmentation/``.
- Is idempotent (overwrites; safe to re-run).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    try:
        import nnunetv2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "nnunetv2 is not importable. Install it (pip install nnunetv2) before running this script."
        ) from e

    variants_dir = (
        Path(nnunetv2.__file__).resolve().parent
        / "training"
        / "nnUNetTrainer"
        / "variants"
        / "data_augmentation"
    )
    if not variants_dir.is_dir():
        raise SystemExit(f"Expected nnU-Net variants dir does not exist: {variants_dir}")

    here = Path(__file__).resolve().parent
    trainer_files = sorted(
        p for p in here.glob("nnUNetTrainer*.py") if p.name != Path(__file__).name
    )
    if not trainer_files:
        raise SystemExit(f"No custom trainer files found in {here}")

    for src in trainer_files:
        dst = variants_dir / src.name
        shutil.copy2(src, dst)
        print(f"[install-trainers] installed -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
