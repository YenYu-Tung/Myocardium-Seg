#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import os
from pathlib import Path
from typing import Iterable, List, Sequence
from sklearn.model_selection import KFold


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare dataset and train locally.")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=base_dir / "workspace",
        help="Path to the workspace.",
    )
    parser.add_argument(
        "--train-image-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Directories that contain patient???.nii.gz volumes. "
             "Defaults to every folder that matches 41_training_image_* inside the project root.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=base_dir / "41_training_label",
        help="Directory that contains patient???_gt.nii.gz label files.",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="chgh",
        help="Dataset nickname.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="AICUP_training_local",
        help="Name of the Ray Tune experiment.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="swinunetr",
        help="Backbone defined inside workspace/networks.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of patients used for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0,
        help="Fraction of patients used for the held-out evaluation split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=13,
        help="Random seed used when shuffling the patient list.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=1,
        help="Number of folds for cross-validation. Set to 5 for 5-fold CV.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Index of the fold to use (0-based) when num-folds > 1.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Epoch to start from when a checkpoint is available.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=800,
        help="Maximum number of epochs to train.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Validation frequency in epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Effective batch size passed to tune.py.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate passed to tune.py.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="AdamW",
        help="Optimizer name (matches torch.optim or custom entries like AdaBelief).",
    )
    parser.add_argument(
        "--lr-schedule",
        dest="lr_schedule",
        type=str,
        default="LinearWarmupCosineAnnealingLR",
        help="Learning rate scheduler name (matches workspace/optimizers/lr_scheduler.py).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay passed to tune.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from.",
    )
    parser.add_argument(
        "--ssl-checkpoint",
        type=Path,
        default=None,
        help="Optional self-supervised checkpoint for warm starting the model.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only prepare the dataset/json file but do not invoke tune.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed instead of running it.",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Enable DataLoader pin_memory (default: enabled).",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pin_memory to reduce shared memory pressure.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=None,
        help="Directory where checkpoints should be stored. Defaults to <workspace>/checkpoints.",
    )
    parser.add_argument(
        "--log-output-dir",
        type=Path,
        default=None,
        help="Directory where TensorBoard logs should be stored. Defaults to <workspace>/logs.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=None,
        help="Directory where evaluation files should be stored. Defaults to <workspace>/evals.",
    )
    parser.add_argument(
        "--strong-aug",
        action="store_true",
        help="Enable additional augmentation ops (affine, noise, coarse dropout).",
    )
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-class weights (length = out_channels) to emphasize specific segments.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=80,
        help="Warmup epochs for schedulers that support it.",
    )
    parser.add_argument(
        "--warmup-start-lr",
        dest="warmup_start_lr",
        type=float,
        default=0.0,
        help="Starting learning rate during warmup.",
    )
    parser.add_argument(
        "--eta-min",
        dest="eta_min",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine schedulers.",
    )
    parser.add_argument(
        "--attn-drop-rate",
        dest="attn_drop_rate",
        type=float,
        default=0.0,
        help="Attention dropout rate for transformer models (e.g., SwinUNETR).",
    )
    parser.add_argument(
        "--dropout-path-rate",
        dest="dropout_path_rate",
        type=float,
        default=0.0,
        help="Drop path (stochastic depth) rate for transformer backbones.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        dest="grad_clip_norm",
        type=float,
        default=0.0,
        help="Gradient clipping L2 norm (0 disables clipping).",
    )
    parser.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        default=True,
        help="Enable torch.cuda.amp mixed precision (default: on).",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable mixed precision.",
    )
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        help="Maintain exponential moving average of weights.",
    )
    parser.add_argument(
        "--ema-decay",
        dest="ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate when --use-ema is set.",
    )
    return parser.parse_args()


def find_image_dirs(project_root: Path, overrides: Sequence[Path] | None) -> List[Path]:
    if overrides:
        return [d.resolve() for d in overrides]
    return sorted(d.resolve() for d in project_root.glob("41_training_image_*") if d.is_dir())


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os_link = getattr(Path, "hardlink_to", None)
        if os_link:
            dst.hardlink_to(src)
        else:
            os.link(src, dst)  
    except OSError:
        shutil.copy2(src, dst)


def stage_dataset(image_dirs: Iterable[Path], label_dir: Path, target_dir: Path) -> List[str]:
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory {label_dir} does not exist.")
    image_map = {}
    for folder in image_dirs:
        if not folder.is_dir():
            raise FileNotFoundError(f"Image directory {folder} does not exist.")
        for img in folder.glob("*.nii.gz"):
            image_map[img.name] = img
    if not image_map:
        raise RuntimeError("No training images were found.")

    patients: List[str] = []
    for label_file in sorted(label_dir.glob("*_gt.nii.gz")):
        patient_id = label_file.name.replace("_gt.nii.gz", "")
        image_name = f"{patient_id}.nii.gz"
        if image_name not in image_map:
            raise RuntimeError(f"Missing image volume for label {label_file.name}")
        link_file(image_map[image_name], target_dir / image_name)
        link_file(label_file, target_dir / label_file.name)
        patients.append(patient_id)

    if not patients:
        raise RuntimeError(f"No labels were found inside {label_dir}")
    return patients


def split_patients(
    patients: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, List[dict[str, str]]]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("Validation and test ratios must sum to < 1.")
    rng = random.Random(seed)
    ordered = list(patients)
    rng.shuffle(ordered)
    total = len(ordered)
    val_count = max(1 if val_ratio > 0 else 0, math.floor(total * val_ratio))
    test_count = max(1 if test_ratio > 0 else 0, math.floor(total * test_ratio))
    if val_count + test_count >= total:
        raise ValueError("Not enough patients to maintain the requested validation/test splits.")
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Not enough patients remain for training with the chosen splits.")
    train = ordered[:train_count]
    val = ordered[train_count:train_count + val_count]
    test = ordered[train_count + val_count:]

    def build(entries: Sequence[str]) -> List[dict[str, str]]:
        return [{"image": f"{pid}.nii.gz", "label": f"{pid}_gt.nii.gz"} for pid in sorted(entries)]

    return {"train": build(train), "val": build(val), "test": build(test)}


def split_patients_kfold(
    patients: Sequence[str],
    num_folds: int,
    fold_index: int,
    test_ratio: float,
    seed: int,
) -> dict[str, List[dict[str, str]]]:
    """Split patients with K-fold CV; optional held-out test set."""
    if num_folds < 2:
        raise ValueError("num_folds must be >=2 to enable K-fold cross validation.")
    rng = random.Random(seed)
    ordered = list(patients)
    rng.shuffle(ordered)

    total = len(ordered)
    test_count = max(1 if test_ratio > 0 else 0, math.floor(total * test_ratio))
    if test_count >= total:
        raise ValueError("Not enough patients left for cross-validation after reserving test set.")
    test = ordered[-test_count:] if test_count else []
    fold_pool = ordered[:-test_count] if test_count else ordered

    if len(fold_pool) < num_folds:
        raise ValueError(f"Need at least {num_folds} patients for {num_folds}-fold CV, got {len(fold_pool)}.")
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(fold_pool))
    if not (0 <= fold_index < num_folds):
        raise ValueError(f"fold_index must be in [0, {num_folds-1}] for {num_folds}-fold CV.")
    train_idx, val_idx = folds[fold_index]
    train = [fold_pool[i] for i in train_idx]
    val = [fold_pool[i] for i in val_idx]

    def build(entries: Sequence[str]) -> List[dict[str, str]]:
        return [{"image": f"{pid}.nii.gz", "label": f"{pid}_gt.nii.gz"} for pid in sorted(entries)]

    return {"train": build(train), "val": build(val), "test": build(test)}


def save_json(split: dict[str, List[dict[str, str]]], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=4)


def build_training_command(
    args: argparse.Namespace,
    workspace_dir: Path,
    data_dir: Path,
    data_dicts_json: Path,
    model_dir: Path,
    log_dir: Path,
    eval_dir: Path,
) -> List[str]:
    root_exp_dir = workspace_dir / "exps" / "exps" / args.model_name / args.data_name / "tune_results"
    root_exp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(workspace_dir / "entrypoints" / "tune.py"),
        "--tune_mode", "train",
        "--exp_name", args.exp_name,
        "--data_name", args.data_name,
        "--data_dir", str(data_dir),
        "--root_exp_dir", str(root_exp_dir),
        "--model_name", args.model_name,
        "--model_dir", str(model_dir),
        "--log_dir", str(log_dir),
        "--eval_dir", str(eval_dir),
        "--start_epoch", str(args.start_epoch),
        "--val_every", str(args.val_every),
        "--max_epoch", str(args.max_epoch),
        "--lrschedule", args.lr_schedule,
        "--warmup_epochs", str(args.warmup_epochs),
        "--warmup_start_lr", str(args.warmup_start_lr),
        "--eta_min", str(args.eta_min),
        "--data_dicts_json", str(data_dicts_json),
        "--num_fold", str(args.num_folds),
        "--fold", str(args.fold_index),
        "--batch_size", str(args.batch_size),
        "--workers", str(args.workers),
        "--optim", args.optim,
        "--lr", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--out_channels", "4",
        "--patch_size", "2",
        "--feature_size", "48",
        "--drop_rate", "0.1",
        "--kernel_size", "7",
        "--exp_rate", "4",
        "--norm_name", "layer",
        "--attn_drop_rate", str(args.attn_drop_rate),
        "--dropout_path_rate", str(args.dropout_path_rate),
        "--grad_clip_norm", str(args.grad_clip_norm),
        "--a_min", "-42",
        "--a_max", "423",
        "--space_x", "0.7",
        "--space_y", "0.7",
        "--space_z", "1",
        "--roi_x", "128",
        "--roi_y", "128",
        "--roi_z", "96", 
        "--depths", "3", "3", "9", "3",
        "--use_init_weights",
        "--infer_post_process",
    ]
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.strong_aug:
        cmd.append("--strong_aug")
    if args.use_amp:
        cmd.append("--use_amp")
    if args.use_ema:
        cmd.extend(["--use_ema", "--ema_decay", str(args.ema_decay)])
    if args.class_weights:
        cmd.extend(["--class_weights", *map(str, args.class_weights)])

    if args.checkpoint:
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    if args.ssl_checkpoint:
        cmd.extend(["--ssl_checkpoint", str(args.ssl_checkpoint)])
    return cmd


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    image_dirs = find_image_dirs(project_root, args.train_image_dirs)
    data_dir = args.workspace_dir / "dataset" / args.data_name
    patients = stage_dataset(image_dirs, args.label_dir, data_dir)

    exp_name = args.exp_name
    if args.num_folds and args.num_folds > 1:
        exp_name = f"{args.exp_name}_fold{args.fold_index + 1}of{args.num_folds}"
        split = split_patients_kfold(
            patients,
            num_folds=args.num_folds,
            fold_index=args.fold_index,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
    else:
        split = split_patients(
            patients,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
    data_dicts_json = args.workspace_dir / "exps" / "data_dicts" / args.data_name / f"{exp_name}.json"
    save_json(split, data_dicts_json)
    print(f"Wrote dataset split to {data_dicts_json}")
    print(f"Staged {len(patients)} patients in {data_dir}")

    if args.skip_training:
        return
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    if args.ssl_checkpoint:
        args.ssl_checkpoint = args.ssl_checkpoint.resolve()

    model_output_dir = (args.model_output_dir or (args.workspace_dir / "checkpoints")).resolve()
    log_output_dir = (args.log_output_dir or (args.workspace_dir / "logs")).resolve()
    eval_output_dir = (args.eval_output_dir or (args.workspace_dir / "evals")).resolve()
    for path in (model_output_dir, log_output_dir, eval_output_dir):
        path.mkdir(parents=True, exist_ok=True)
    args.exp_name = exp_name

    cmd = build_training_command(
        args,
        args.workspace_dir,
        data_dir,
        data_dicts_json,
        model_output_dir,
        log_output_dir,
        eval_output_dir,
    )
    print("Launching training command:\n  " + " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
