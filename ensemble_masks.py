#!/usr/bin/env python3
"""Ensemble multiple sets of predicted masks and save the averaged result per patient."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import nibabel as nib
import numpy as np


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average multiple sets of predicted masks (per patient) and save the ensembled result."
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories containing predicted .nii.gz masks. Must contain the same filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write ensembled masks.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of segmentation classes (including background).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".nii.gz",
        help="Filename suffix to look for (default: .nii.gz).",
    )
    return parser.parse_args(argv)


def list_files(directory: Path, suffix: str) -> set[str]:
    return {p.name for p in directory.glob(f"*{suffix}") if p.is_file()}


def one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((num_classes,) + mask.shape, dtype=np.float32)
    mask_int = mask.astype(np.int64)
    valid = (mask_int >= 0) & (mask_int < num_classes)
    for c in range(num_classes):
        oh[c] = (mask_int == c) & valid
    return oh


def load_as_probs(path: Path, num_classes: int) -> np.ndarray:
    data = nib.load(str(path)).get_fdata()
    if data.ndim == 4:
        if data.shape[0] != num_classes:
            raise ValueError(f"{path} has channel dimension {data.shape[0]} but expected num_classes={num_classes}")
        return data.astype(np.float32)
    if data.ndim == 3:
        return one_hot(data, num_classes)
    raise ValueError(f"Unsupported ndim {data.ndim} for {path}")


def ensemble_patient(paths: Iterable[Path], num_classes: int) -> tuple[np.ndarray, nib.Nifti1Image]:
    paths = list(paths)
    probs: List[np.ndarray] = []
    ref_img: nib.Nifti1Image | None = None
    for p in paths:
        img = nib.load(str(p))
        if ref_img is None:
            ref_img = img
        probs.append(load_as_probs(p, num_classes))
    assert ref_img is not None, "No reference image found."
    stacked = np.stack(probs, axis=0)  # (N, C, H, W, D)
    avg = stacked.mean(axis=0)
    pred = np.argmax(avg, axis=0).astype(np.uint8)
    return pred, ref_img


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    input_dirs = [d.resolve() for d in args.input_dirs]
    for d in input_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Input dir not found: {d}")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find common filenames across all input dirs
    common_files = list_files(input_dirs[0], args.suffix)
    for d in input_dirs[1:]:
        common_files &= list_files(d, args.suffix)
    if not common_files:
        raise RuntimeError("No common files found across input directories.")
    print(f"Found {len(common_files)} common files.")

    for name in sorted(common_files):
        paths = [d / name for d in input_dirs]
        pred, ref_img = ensemble_patient(paths, args.num_classes)
        out_path = out_dir / name
        nib.Nifti1Image(pred, ref_img.affine, ref_img.header).to_filename(out_path)
        print(f"Saved ensemble: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
