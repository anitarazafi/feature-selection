import pandas as pd
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

from src.utils.paths import BASE_DIR, CONFIG_DIR
from src.utils.data_io import load_splits


def evaluate_topk(X_train, X_val, y_train, y_val, features, seed=42):
    """Train XGBoost on given features and return val AUC and F1."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X_train[features], y_train)
    y_pred  = model.predict(X_val[features])
    y_proba = model.predict_proba(X_val[features])[:, 1]
    return roc_auc_score(y_val, y_proba), f1_score(y_val, y_pred)


def sweep_k_traditional(dataset_name, seed=42):
    """
    Sweep k values for the Traditional FS paradigm.
    For each k: evaluate MI top-k and RFE top-k individually with XGBoost.
    Pick the k where both methods perform well.
    """
    print(f"\n{'='*60}")
    print(f" Validation Sweep: Choosing optimal k for Traditional FS")
    print(f" Dataset: {dataset_name}, seed={seed}")
    print(f"{'='*60}")

    # ── Load config and data ──────────────────────────────────────
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)

    if dataset_name not in all_configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.")

    cfg = all_configs[dataset_name]
    splits_dir = BASE_DIR / cfg["paths"]["splits"]
    results_dir = BASE_DIR / cfg["paths"]["results"]

    splits = load_splits(splits_dir)
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]

    n_total = X_train.shape[1]
    feature_names = list(X_train.columns)
    print(f"Total features: {n_total}")

    # ── Define k range ────────────────────────────────────────────
    k_values = [3, 5, 7, 10, 13, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    k_values = [k for k in k_values if k < n_total]
    print(f"k values to sweep: {k_values}\n")

    # ── Compute MI scores once ────────────────────────────────────
    print("Computing Mutual Information scores...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)
    mi_ranking = np.argsort(mi_scores)[::-1]
    print("MI scores computed.\n")

    # ── Pre-compute RFE rankings for all k values ─────────────────
    print("Pre-computing RFE rankings for all k values...")
    rfe_estimator = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1
    )
    rfe_results = {}
    for k in k_values:
        print(f"  RFE k={k:>3d} ... ", end="", flush=True)
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=k, step=5)
        rfe.fit(X_train, y_train)
        rfe_results[k] = X_train.columns[rfe.get_support()].tolist()
        print("done")
    print()

    # ── Sweep k values ────────────────────────────────────────────
    results = []

    for k in k_values:
        print(f"  k={k:>3d} ... ", end="", flush=True)

        # MI: top-k features
        mi_features = [feature_names[i] for i in mi_ranking[:k]]

        # RFE: top-k features
        rfe_features = rfe_results[k]

        # Evaluate MI top-k
        mi_auc, mi_f1 = evaluate_topk(X_train, X_val, y_train, y_val, mi_features, seed)

        # Evaluate RFE top-k
        rfe_auc, rfe_f1 = evaluate_topk(X_train, X_val, y_train, y_val, rfe_features, seed)

        # Average of both (for choosing best k)
        avg_auc = (mi_auc + rfe_auc) / 2
        avg_f1  = (mi_f1 + rfe_f1) / 2

        print(f"MI: AUC={mi_auc:.4f} F1={mi_f1:.4f}  |  "
              f"RFE: AUC={rfe_auc:.4f} F1={rfe_f1:.4f}  |  "
              f"Avg: AUC={avg_auc:.4f} F1={avg_f1:.4f}")

        results.append({
            "k": k,
            "mi_auc": round(mi_auc, 4),
            "mi_f1": round(mi_f1, 4),
            "rfe_auc": round(rfe_auc, 4),
            "rfe_f1": round(rfe_f1, 4),
            "avg_auc": round(avg_auc, 4),
            "avg_f1": round(avg_f1, 4),
        })

    results_df = pd.DataFrame(results)

    # ── Save results ──────────────────────────────────────────────
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "k_sweep_traditional.csv", index=False)
    print(f"\nSaved sweep results to {tables_dir / 'k_sweep_traditional.csv'}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC plot
    ax1.plot(results_df["k"], results_df["mi_auc"],
             "o-", color="#2563eb", linewidth=2, markersize=5, label="MI")
    ax1.plot(results_df["k"], results_df["rfe_auc"],
             "s-", color="#16a34a", linewidth=2, markersize=5, label="RFE")
    ax1.plot(results_df["k"], results_df["avg_auc"],
             "D--", color="#9333ea", linewidth=2, markersize=5, label="Average")
    ax1.set_xlabel("k (features per method)", fontsize=12)
    ax1.set_ylabel("Validation AUC", fontsize=12)
    ax1.set_title("Validation AUC vs k", fontsize=13)
    ax1.set_xticks(results_df["k"].tolist())
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    best_avg_auc_row = results_df.loc[results_df["avg_auc"].idxmax()]
    ax1.axvline(x=best_avg_auc_row["k"], color="#9333ea", linestyle=":", alpha=0.5)
    ax1.annotate(f'Best avg: k={int(best_avg_auc_row["k"])}',
                 xy=(best_avg_auc_row["k"], best_avg_auc_row["avg_auc"]),
                 xytext=(10, -15), textcoords="offset points",
                 fontsize=10, color="#9333ea",
                 arrowprops=dict(arrowstyle="->", color="#9333ea"))

    # F1 plot
    ax2.plot(results_df["k"], results_df["mi_f1"],
             "o-", color="#2563eb", linewidth=2, markersize=5, label="MI")
    ax2.plot(results_df["k"], results_df["rfe_f1"],
             "s-", color="#16a34a", linewidth=2, markersize=5, label="RFE")
    ax2.plot(results_df["k"], results_df["avg_f1"],
             "D--", color="#9333ea", linewidth=2, markersize=5, label="Average")
    ax2.set_xlabel("k (features per method)", fontsize=12)
    ax2.set_ylabel("Validation F1", fontsize=12)
    ax2.set_title("Validation F1 vs k", fontsize=13)
    ax2.set_xticks(results_df["k"].tolist())
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    best_avg_f1_row = results_df.loc[results_df["avg_f1"].idxmax()]
    ax2.axvline(x=best_avg_f1_row["k"], color="#9333ea", linestyle=":", alpha=0.5)
    ax2.annotate(f'Best avg: k={int(best_avg_f1_row["k"])}',
                 xy=(best_avg_f1_row["k"], best_avg_f1_row["avg_f1"]),
                 xytext=(10, -15), textcoords="offset points",
                 fontsize=10, color="#9333ea",
                 arrowprops=dict(arrowstyle="->", color="#9333ea"))

    fig.suptitle("Traditional FS: k Selection (MI and RFE evaluated individually with XGBoost)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "k_sweep_traditional.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plots_dir / 'k_sweep_traditional.png'}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Best k by Avg AUC: k={int(best_avg_auc_row['k'])}  "
          f"(MI={best_avg_auc_row['mi_auc']:.4f}, RFE={best_avg_auc_row['rfe_auc']:.4f}, "
          f"Avg={best_avg_auc_row['avg_auc']:.4f})")
    print(f"Best k by Avg F1:  k={int(best_avg_f1_row['k'])}  "
          f"(MI={best_avg_f1_row['mi_f1']:.4f}, RFE={best_avg_f1_row['rfe_f1']:.4f}, "
          f"Avg={best_avg_f1_row['avg_f1']:.4f})")
    print(f"{'─'*50}\n")

    return results_df