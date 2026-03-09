import pandas as pd
from pathlib import Path
import yaml
from src.utils.paths import BASE_DIR, CONFIG_DIR


def build_results_table(dataset_name):
    """
    Combine all results into a single comprehensive table.
    """
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg         = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"] / "tables"

    # ── Load all results ──────────────────────────────────────────
    dfs = []
    for name in ["baseline", "traditional", "optimization", "xai"]:
        path = results_dir / f"{name}.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: {name}.csv not found, skipping")

    all_results = pd.concat(dfs, ignore_index=True)

    # ── Keep only test metrics ────────────────────────────────────
    report_cols = [
        "model", "fs_method", "n_features",
        "test_auc", "test_f1_score", "test_precision",
        "test_recall", "test_accuracy",
        "fs_time", "train_time"
    ]
    # fs_time not in baseline — fill with 0
    if "fs_time" not in all_results.columns:
        all_results["fs_time"] = 0.0
    all_results["fs_time"] = all_results["fs_time"].fillna(0.0)

    report = all_results[report_cols].copy()

    # ── Sort: model → method → n_features ────────────────────────
    report = report.sort_values(
        ["model", "fs_method", "n_features"]
    ).reset_index(drop=True)

    # ── Save full table ───────────────────────────────────────────
    report.to_csv(results_dir / "all_results.csv", index=False)
    print(f"Saved full table to {results_dir / 'all_results.csv'}")

    # ── Pivot table: best result per method per model ─────────────
    best = (
        report
        .sort_values("test_auc", ascending=False)
        .groupby(["model", "fs_method"])
        .first()
        .reset_index()
    )
    best.to_csv(results_dir / "best_per_method.csv", index=False)
    print(f"Saved best-per-method table to {results_dir / 'best_per_method.csv'}")

    # ── Summary pivot: method × model → AUC ──────────────────────
    pivot_auc = best.pivot(
        index="fs_method",
        columns="model",
        values="test_auc"
    ).round(4)

    pivot_f1 = best.pivot(
        index="fs_method",
        columns="model",
        values="test_f1_score"
    ).round(4)

    pivot_auc.to_csv(results_dir / "pivot_auc.csv")
    pivot_f1.to_csv(results_dir / "pivot_f1.csv")
    print(f"Saved AUC pivot to {results_dir / 'pivot_auc.csv'}")
    print(f"Saved F1 pivot  to {results_dir / 'pivot_f1.csv'}")

    # ── Print summary to console ──────────────────────────────────
    print(f"\n{'='*60}")
    print("AUC SUMMARY (best per method per model)")
    print(f"{'='*60}")
    print(pivot_auc.to_string())

    print(f"\n{'='*60}")
    print("F1 SUMMARY (best per method per model)")
    print(f"{'='*60}")
    print(pivot_f1.to_string())

    # ── Highlight best method per model ──────────────────────────
    print(f"\n{'='*60}")
    print("BEST METHOD PER MODEL (by AUC)")
    print(f"{'='*60}")
    for model_name in best["model"].unique():
        model_df = best[best["model"] == model_name]
        top      = model_df.loc[model_df["test_auc"].idxmax()]
        print(f"  {model_name:<20} → {top['fs_method']:<25} "
              f"AUC={top['test_auc']:.4f}  "
              f"F1={top['test_f1_score']:.4f}  "
              f"k={int(top['n_features'])}")

    # ── Baseline comparison ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("IMPROVEMENT OVER BASELINE (AUC)")
    print(f"{'='*60}")
    baseline_auc = (
        best[best["fs_method"] == "baseline"]
        .set_index("model")["test_auc"]
    )
    for model_name in best["model"].unique():
        model_df = best[
            (best["model"] == model_name) &
            (best["fs_method"] != "baseline")
        ]
        if model_df.empty:
            continue
        top      = model_df.loc[model_df["test_auc"].idxmax()]
        baseline = baseline_auc.get(model_name, None)
        if baseline:
            delta = top["test_auc"] - baseline
            print(f"  {model_name:<20} → best: {top['fs_method']:<25} "
                  f"Δ AUC={delta:+.4f}")

    return report, best, pivot_auc, pivot_f1