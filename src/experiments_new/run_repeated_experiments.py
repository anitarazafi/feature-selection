import pandas as pd
import numpy as np
import json
import yaml
import time
from pathlib import Path

from src.utils.paths import BASE_DIR, CONFIG_DIR
from src.experiments_new.run_baseline import train_baseline
from src.experiments_new.run_traditional_fs import run_traditional_fs
from src.experiments_new.run_optimization_fs import run_optimization_fs
from src.experiments_new.run_xai_fs import run_xai_fs

SEEDS = [42, 123, 456, 789, 1024, 2023, 2024, 3333, 5555, 9999]


def run_all_experiments(dataset_name, k_traditional=None, k_optimization=None, k_xai=None):
    """
    Run baseline + 3 FS paradigms across multiple seeds.
    Each paradigm uses its own optimal k value.
    Collect all results and compute mean ± std.
    """
    print(f"\n{'#'*60}")
    print(f"# REPEATED EXPERIMENTS: {len(SEEDS)} seeds")
    print(f"# Seeds: {SEEDS}")
    print(f"# Dataset: {dataset_name}")
    print(f"# k values: traditional={k_traditional}, optimization={k_optimization}, xai={k_xai}")
    print(f"{'#'*60}\n")

    all_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'*'*60}")
        print(f"* RUN {i+1}/{len(SEEDS)} — seed={seed}")
        print(f"{'*'*60}")

        # Baseline
        df = train_baseline(dataset_name, seed=seed)
        all_results.append(df)

        # Traditional FS
        df = run_traditional_fs(dataset_name, seed=seed, k=k_traditional)
        all_results.append(df)

        # Optimization FS
        df = run_optimization_fs(dataset_name, seed=seed, k=k_optimization)
        all_results.append(df)

        # XAI FS
        df = run_xai_fs(dataset_name, seed=seed, k=k_xai)
        all_results.append(df)

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save combined results
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(tables_dir / "all_seeds_combined.csv", index=False)
    print(f"\nSaved combined results to {tables_dir / 'all_seeds_combined.csv'}")

    # Compute summary statistics
    summary_df = compute_summary(combined_df)
    summary_df.to_csv(tables_dir / "summary_mean_std.csv", index=False)
    print(f"Saved summary to {tables_dir / 'summary_mean_std.csv'}")

    return combined_df, summary_df


def compute_summary(df):
    """Compute mean ± std for each (fs_method, model) combination."""
    metrics = ["test_accuracy", "test_precision", "test_recall",
               "test_f1_score", "test_auc", "train_time"]

    grouped = df.groupby(["fs_method", "model"])

    rows = []
    for (fs_method, model), group in grouped:
        row = {"fs_method": fs_method, "model": model,
               "n_features": int(group["n_features"].iloc[0])}
        for metric in metrics:
            values = group[metric].values
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"]  = np.std(values)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "ibtracs.ALL.list.v04r01"

    # Set paradigm-specific k values from sweep results
    k_traditional  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    k_optimization = int(sys.argv[3]) if len(sys.argv) > 3 else None
    k_xai          = int(sys.argv[4]) if len(sys.argv) > 4 else None

    combined_df, summary_df = run_all_experiments(
        dataset_name,
        k_traditional=k_traditional,
        k_optimization=k_optimization,
        k_xai=k_xai
    )
    print("\nDone! All repeated experiments complete.")