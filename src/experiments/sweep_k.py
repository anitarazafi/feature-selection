import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
import time
import json
from pathlib import Path

import yaml
from src.utils.paths import BASE_DIR, CONFIG_DIR
from src.utils.data_io import load_splits


def sweep_k(dataset_name):
    """
    Sweep k values using Mutual Information + XGBoost on the validation set.
    Produces a CSV of results and a plot of Val AUC / Val F1 vs k
    to help choose the optimal number of features for all FS methods.
    """
    print(f"\n{'='*60}")
    print(f" Validation Sweep: Choosing optimal k for {dataset_name}")
    print(f"{'='*60}")

    # ── Load config and data ──────────────────────────────────────
    configs = CONFIG_DIR / "datasets.yaml"
    with open(configs, "r") as f:
        all_configs = yaml.safe_load(f)

    if dataset_name not in all_configs:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in config. "
            f"Available: {list(all_configs.keys())}"
        )

    cfg = all_configs[dataset_name]
    splits_dir = BASE_DIR / cfg["paths"]["splits"]
    results_dir = BASE_DIR / cfg["paths"]["results"]

    splits = load_splits(splits_dir)
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]

    n_total = X_train.shape[1]
    print(f"Total features: {n_total}")

    # ── Define k range ────────────────────────────────────────────
    k_values = [3, 5, 7, 10, 13, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    # Remove any k >= total features
    k_values = [k for k in k_values if k < n_total]
    # Add total features as the last point (equivalent to baseline)
    k_values.append(n_total)
    print(f"k values to sweep: {k_values}\n")

    # ── Compute MI scores once ────────────────────────────────────
    print("Computing Mutual Information scores...")
    mi_start = time.time()
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_time = time.time() - mi_start
    print(f"MI computation time: {mi_time:.2f}s")

    # Rank features by MI score (descending)
    mi_ranking = np.argsort(mi_scores)[::-1]

    # ── Sweep k values ────────────────────────────────────────────
    results = []

    for k in k_values:
        print(f"  k={k:>3d} ... ", end="", flush=True)

        # Select top-k features
        top_k_idx = mi_ranking[:k]
        X_train_k = X_train.iloc[:, top_k_idx]
        X_val_k   = X_val.iloc[:, top_k_idx]

        # Train XGBoost
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )

        start = time.time()
        model.fit(X_train_k, y_train)
        train_time = time.time() - start

        # Evaluate on validation set
        y_val_pred  = model.predict(X_val_k)
        y_val_proba = model.predict_proba(X_val_k)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_f1  = f1_score(y_val, y_val_pred)

        print(f"Val AUC={val_auc:.4f}  Val F1={val_f1:.4f}  ({train_time:.1f}s)")

        results.append({
            "k": k,
            "val_auc": round(val_auc, 4),
            "val_f1":  round(val_f1, 4),
            "train_time": round(train_time, 2),
        })

    results_df = pd.DataFrame(results)

    # ── Save results ──────────────────────────────────────────────
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "k_sweep.csv", index=False)
    print(f"\nSaved sweep results to {tables_dir / 'k_sweep.csv'}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_auc = "#2563eb"
    color_f1  = "#dc2626"

    ax1.set_xlabel("Number of Features (k)", fontsize=12)
    ax1.set_ylabel("Validation AUC", color=color_auc, fontsize=12)
    ax1.plot(results_df["k"], results_df["val_auc"],
             "o-", color=color_auc, linewidth=2, markersize=6, label="Val AUC")
    ax1.tick_params(axis="y", labelcolor=color_auc)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation F1", color=color_f1, fontsize=12)
    ax2.plot(results_df["k"], results_df["val_f1"],
             "s--", color=color_f1, linewidth=2, markersize=6, label="Val F1")
    ax2.tick_params(axis="y", labelcolor=color_f1)

    # Mark the best k for each metric
    best_auc_row = results_df.loc[results_df["val_auc"].idxmax()]
    best_f1_row  = results_df.loc[results_df["val_f1"].idxmax()]

    ax1.axvline(x=best_auc_row["k"], color=color_auc, linestyle=":", alpha=0.5)
    ax2.axvline(x=best_f1_row["k"],  color=color_f1,  linestyle=":", alpha=0.5)

    ax1.annotate(f'Best AUC: k={int(best_auc_row["k"])}',
                 xy=(best_auc_row["k"], best_auc_row["val_auc"]),
                 xytext=(10, 10), textcoords="offset points",
                 fontsize=10, color=color_auc,
                 arrowprops=dict(arrowstyle="->", color=color_auc))

    ax2.annotate(f'Best F1: k={int(best_f1_row["k"])}',
                 xy=(best_f1_row["k"], best_f1_row["val_f1"]),
                 xytext=(10, -15), textcoords="offset points",
                 fontsize=10, color=color_f1,
                 arrowprops=dict(arrowstyle="->", color=color_f1))

    fig.suptitle("Validation Performance vs Number of Features (MI + XGBoost)",
                 fontsize=13, fontweight="bold")
    ax1.set_xticks(results_df["k"].tolist())
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "k_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plots_dir / 'k_sweep.png'}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"Best k by Val AUC: k={int(best_auc_row['k'])}  (AUC={best_auc_row['val_auc']:.4f})")
    print(f"Best k by Val F1:  k={int(best_f1_row['k'])}  (F1={best_f1_row['val_f1']:.4f})")
    print(f"{'─'*40}\n")

    return results_df