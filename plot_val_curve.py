#!/usr/bin/env python3
"""Plot validation accuracy vs epoch from one or multiple Ray Tune trials."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_progress(run_dir: Path, metric: str) -> pd.DataFrame:
    progress_csv = run_dir / "progress.csv"
    if progress_csv.exists():
        df = pd.read_csv(progress_csv)
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in progress.csv columns: {list(df.columns)}")
        # prefer 'training_iteration' if present, otherwise try 'epoch' or index
        if "training_iteration" in df.columns:
            df = df[["training_iteration", metric]].rename(columns={"training_iteration": "epoch"})
        elif "epoch" in df.columns:
            df = df[["epoch", metric]]
        else:
            df = df.reset_index().rename(columns={"index": "epoch"})
        return df.dropna(subset=[metric])

    result_json = run_dir / "result.json"
    if result_json.exists():
        rows = []
        with result_json.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if metric in obj:
                    epoch = obj.get("training_iteration", obj.get("epoch", len(rows)))
                    rows.append({"epoch": epoch, metric: obj[metric]})
        if rows:
            return pd.DataFrame(rows)
    raise FileNotFoundError(f"Neither progress.csv nor usable result.json found under {run_dir}")


def plot_curve(dfs: dict[str, pd.DataFrame], metric: str, out_path: Path, title: str, y_label: str):
    plt.figure(figsize=(8, 4))
    for label, df in dfs.items():
        plt.plot(
            df["epoch"], 
            df[metric] * 100, 
            marker="None", 
            linewidth=2,      
            label=label
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="lower right")
    legend = plt.legend(loc="lower right")
    legend.texts[-1].set_fontweight('bold')   
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot validation accuracy vs epoch from a Ray Tune trial folder.")
    parser.add_argument("--run-dirs", required=True, type=Path, nargs="+", help="One or more Ray Tune trial directories (each contains progress.csv/result.json).")
    parser.add_argument("--metric", default="val_bst_acc", help="Metric name to plot (default: val_bst_acc).")
    parser.add_argument("--metric-display", default="Validation Accuracy (%)", help="Y-axis label to show on the plot.")
    parser.add_argument("--epoch-scale", type=float, default=5.0, help="Multiply epoch axis by this factor (default: 5).")
    parser.add_argument("--labels", nargs="+", default=None, help="Custom curve labels matching the number of run-dirs.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (png). Defaults to <run-dir>/<metric>_curve.png")
    parser.add_argument("--title", type=str, default="Performance Evaluation Across Training Epochs", help="Custom plot title.")
    args = parser.parse_args()

    dfs = {}
    run_dirs = [rd.resolve() for rd in args.run_dirs]
    custom_labels = args.labels or [rd.name for rd in run_dirs]
    if len(custom_labels) != len(run_dirs):
        raise ValueError("Number of labels must match number of run-dirs.")

    for rd, label in zip(run_dirs, custom_labels):
        df = load_progress(rd, args.metric)
        df["epoch"] = df["epoch"] * args.epoch_scale
        dfs[label] = df
    # default output: use first run_dir name
    out_path = args.output or run_dirs[0] / f"{args.metric}_curve.png"
    title = args.title or f"validation accuracy vs epoch"
    plot_curve(dfs, args.metric, out_path, title, args.metric_display)


if __name__ == "__main__":
    main()



