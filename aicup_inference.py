#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_workspace = base_dir / "workspace"
    parser = argparse.ArgumentParser(description="Run local inference using local checkpoints.")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=default_workspace,
        help="Path to the workspace repository.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="AICUP_training_local",
        help="Name of the experiment directory produced during training.",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="chgh",
        help="Dataset nickname used by workspace.",
    )
    parser.add_argument(
        "--data-dicts-json",
        type=Path,
        default=None,
        help="Optional data split JSON (train/val/test). When provided, inference focuses on the 'test' list.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="swinunetr",
        help="Backbone defined inside workspace/networks.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint (.pth). If omitted the script searches for the latest best_model.pth "
             "within the experiment directory.",
    )
    parser.add_argument(
        "--root-exp-dir",
        type=Path,
        default=None,
        help="Override the Ray experiment root. Defaults to <workspace>/exps/exps/<model>/<data>/tune_results.",
    )
    parser.add_argument(
        "--image-roots",
        type=Path,
        nargs="*",
        default=None,
        help="Folders that contain *.nii.gz volumes for inference. "
             "Defaults to 41_testing_image_* inside the project root.",
    )
    parser.add_argument(
        "--image-files",
        type=Path,
        nargs="*",
        default=None,
        help="Additional individual volumes to include.",
    )
    parser.add_argument(
        "--image-pattern",
        type=str,
        default="*.nii.gz",
        help="Glob used to discover inference volumes inside each root folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "inference_outputs",
        help="Directory where predicted masks (.nii.gz) will be written.",
    )
    parser.add_argument(
        "--zip-output",
        type=Path,
        default=None,
        help="Optional path (with or without .zip suffix) where the predictions directory will be archived.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Run inference in evaluation mode (no outputs saved, requires labels/data_dicts_json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands without executing inference.",
    )
    parser.add_argument(
        "--infer-post-process",
        dest="infer_post_process",
        action="store_true",
        help="Enable the --infer_post_process flag (keep largest component).",
    )
    parser.add_argument(
        "--no-infer-post-process",
        dest="infer_post_process",
        action="store_false",
        help="Disable post-processing (default: enabled).",
    )
    parser.set_defaults(infer_post_process=True)
    parser.add_argument(
        "--out-channels",
        type=int,
        default=4,
        help="Number of segmentation classes passed to infer.py.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=2,
        help="Patch size argument for the network.",
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=48,
        help="Feature size argument for the network.",
    )
    parser.add_argument(
        "--drop-rate",
        type=float,
        default=0.1,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[3, 3, 9, 3],
        help="Depth configuration for the model backbone.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=7,
        help="Kernel size.",
    )
    parser.add_argument(
        "--exp-rate",
        type=int,
        default=4,
        help="Expansion rate.",
    )
    parser.add_argument(
        "--norm-name",
        type=str,
        default="layer",
        help="Normalization type.",
    )
    parser.add_argument(
        "--a-min",
        type=float,
        default=-42.0,
        help="ScaleIntensityRanged a_min.",
    )
    parser.add_argument(
        "--a-max",
        type=float,
        default=423.0,
        help="ScaleIntensityRanged a_max.",
    )
    parser.add_argument(
        "--space-x",
        type=float,
        default=0.7,
        help="Spacing along x.",
    )
    parser.add_argument(
        "--space-y",
        type=float,
        default=0.7,
        help="Spacing along y.",
    )
    parser.add_argument(
        "--space-z",
        type=float,
        default=1.0,
        help="Spacing along z.",
    )
    parser.add_argument(
        "--roi-x",
        type=int,
        default=128,
        help="Sliding window ROI along x.",
    )
    parser.add_argument(
        "--roi-y",
        type=int,
        default=128,
        help="Sliding window ROI along y.",
    )
    parser.add_argument(
        "--roi-z",
        type=int,
        default=96,
        help="Sliding window ROI along z.",
    )
    parser.add_argument(
        "--infer-overlap",
        type=float,
        default=0.25,
        help="Sliding window overlap passed to infer.py.",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=4,
        help="Sliding window batch size passed to infer.py.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=None,
        help="Directory to search for checkpoints in addition to the Ray experiment folder (defaults to <workspace>/checkpoints).",
    )
    return parser.parse_args()


def find_image_paths(project_root: Path, image_roots: Sequence[Path] | None, image_files: Sequence[Path] | None,
                     pattern: str) -> List[Path]:
    paths: List[Path] = []
    if image_roots:
        roots = list(image_roots)
    else:
        roots = sorted(p for p in project_root.glob("41_testing_image_*") if p.is_dir())
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Inference image directory {root} does not exist.")
        paths.extend(sorted(root.glob(pattern)))
    if image_files:
        for file_path in image_files:
            file_path = file_path.resolve()
            if not file_path.is_file():
                raise FileNotFoundError(f"Inference volume {file_path} does not exist.")
            paths.append(file_path)
    unique = sorted(dict.fromkeys(paths))
    return unique


def find_checkpoint(workspace_dir: Path, args: argparse.Namespace) -> Path:
    if args.checkpoint:
        ckpt = args.checkpoint.resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt} not found.")
        return ckpt
    root_exp_dir = args.root_exp_dir or workspace_dir / "exps" / "exps" / args.model_name / args.data_name / "tune_results"
    exp_dir = root_exp_dir / args.exp_name
    search_roots: List[Path] = []
    if exp_dir.is_dir():
        search_roots.append(exp_dir)
    model_output_dir = (args.model_output_dir or (workspace_dir / "checkpoints")).resolve()
    if model_output_dir.is_dir():
        search_roots.append(model_output_dir)
    candidates: List[Path] = []
    for base in search_roots:
        candidates.extend(base.glob("**/best_model.pth"))
    resolved = {}
    for candidate in candidates:
        try:
            resolved[candidate.resolve()] = candidate.stat().st_mtime
        except FileNotFoundError:
            continue
    if not resolved:
        raise FileNotFoundError(
            f"No best_model.pth files were found under {exp_dir} or {model_output_dir}. Provide --checkpoint explicitly."
        )
    best_path = max(resolved.items(), key=lambda kv: kv[1])[0]
    return best_path


def build_command(args: argparse.Namespace, workspace_dir: Path, checkpoint: Path, image_path: Path,
                  output_dir: Path, label_path: Path | None, test_mode: bool) -> List[str]:
    infer_py = workspace_dir / "entrypoints" / "infer.py"
    data_dir = workspace_dir / "dataset" / args.data_name
    cmd = [
        sys.executable,
        str(infer_py),
        "--model_name", args.model_name,
        "--data_name", args.data_name,
        "--data_dir", str(data_dir),
        "--infer_dir", str(output_dir),
        "--checkpoint", str(checkpoint),
        "--img_pth", str(image_path),
        "--out_channels", str(args.out_channels),
        "--patch_size", str(args.patch_size),
        "--feature_size", str(args.feature_size),
        "--drop_rate", str(args.drop_rate),
        "--kernel_size", str(args.kernel_size),
        "--exp_rate", str(args.exp_rate),
        "--norm_name", args.norm_name,
        "--a_min", str(args.a_min),
        "--a_max", str(args.a_max),
        "--space_x", str(args.space_x),
        "--space_y", str(args.space_y),
        "--space_z", str(args.space_z),
        "--roi_x", str(args.roi_x),
        "--roi_y", str(args.roi_y),
        "--roi_z", str(args.roi_z),
        "--infer_overlap", str(args.infer_overlap),
        "--sw_batch_size", str(args.sw_batch_size),
    ]
    if label_path:
        cmd.extend(["--lbl_pth", str(label_path)])
    if args.depths:
        cmd.extend(["--depths", *map(str, args.depths)])
    if args.infer_post_process:
        cmd.append("--infer_post_process")
    if test_mode:
        cmd.append("--test_mode")
    return cmd


def archive_outputs(output_dir: Path, archive_path: Path) -> Path:
    archive_path = archive_path.with_suffix(".zip") if archive_path.suffix.lower() != ".zip" else archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_base = archive_path
    if archive_path.suffix.lower() == ".zip":
        archive_base = archive_path.with_suffix("")
    shutil.make_archive(
        base_name=str(archive_base),
        format="zip",
        root_dir=str(output_dir),
        base_dir=".",
    )
    return archive_path


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    workspace_dir = args.workspace_dir.resolve()

    data_dicts_json = args.data_dicts_json.resolve() if args.data_dicts_json else None
    targets: List[tuple[Path, Path | None]] = []
    if data_dicts_json:
        if not data_dicts_json.exists():
            raise FileNotFoundError(f"data dict json {data_dicts_json} not found")
        spec = json.loads(data_dicts_json.read_text(encoding="utf-8"))
        test_entries = spec.get("test") or []
        for entry in test_entries:
            img = (workspace_dir / "dataset" / args.data_name / entry["image"]).resolve()
            lbl = entry.get("label")
            label_path = (workspace_dir / "dataset" / args.data_name / lbl).resolve() if lbl else None
            targets.append((img, label_path))
    else:
        image_paths = find_image_paths(project_root, args.image_roots, args.image_files, args.image_pattern)
        targets = [(path, None) for path in image_paths]
    if not targets:
        raise RuntimeError("No inference volumes were found. Use --image-roots/--image-files to specify inputs.")
    if args.limit:
        targets = targets[: args.limit]

    checkpoint = find_checkpoint(workspace_dir, args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dicts_json = args.data_dicts_json.resolve() if args.data_dicts_json else None

    if args.metrics_only and any(label_path is None for _, label_path in targets):
        raise RuntimeError("Metrics-only mode requires labels. Provide --data-dicts-json with label entries.")

    for idx, (image_path, label_path) in enumerate(targets, 1):
        cmd = build_command(args, workspace_dir, checkpoint, image_path, output_dir, label_path, args.metrics_only)
        print(f"[{idx}/{len(targets)}] Running inference for {image_path.name}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    if args.zip_output and not args.dry_run and not args.metrics_only:
        archive_path = archive_outputs(output_dir, args.zip_output.resolve())
        print(f"Archived predictions to {archive_path}")


if __name__ == "__main__":
    main()
