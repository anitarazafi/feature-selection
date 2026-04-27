import pandas as pd
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
import pyswarms as ps

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


def evaluate_feature_subset(X_train, X_val, y_train, y_val, selected_cols, model):
    """Fit model on selected columns, return AUC on validation set."""
    if len(selected_cols) == 0:
        return 0.0
    model.fit(X_train[selected_cols], y_train)
    y_proba = model.predict_proba(X_val[selected_cols])[:, 1]
    return roc_auc_score(y_val, y_proba)


def pso_objective(position, X_train, X_val, y_train, y_val, feature_names, model, target_k):
    n_particles = position.shape[0]
    costs = np.zeros(n_particles)
    for i in range(n_particles):
        top_k_idx = np.argsort(position[i])[-target_k:]
        selected_cols = [feature_names[j] for j in top_k_idx]
        auc = evaluate_feature_subset(X_train, X_val, y_train, y_val, selected_cols, model)
        costs[i] = 1 - auc
    return costs


def run_pso_for_k(X_train, X_val, y_train, y_val, feature_names, k, seed=42,
                  n_particles=30, iterations=20):
    """Run PSO for a specific k and return selected features."""
    np.random.seed(seed)
    n_features = len(feature_names)

    eval_model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", random_state=seed,
    )

    options = {"c1": 1.5, "c2": 1.5, "w": 0.7}
    bounds = (np.zeros(n_features), np.ones(n_features))

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=n_features,
        options=options, bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(
        pso_objective, iters=iterations, verbose=False,
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val,
        feature_names=feature_names, model=eval_model, target_k=k
    )

    top_k_idx = np.argsort(best_pos)[-k:]
    selected_cols = [feature_names[j] for j in top_k_idx]
    return selected_cols


def run_de_for_k(X_train, X_val, y_train, y_val, feature_names, k, seed=42,
                 population_size=30, generations=20):
    """Run DE for a specific k and return selected features."""
    n_features = len(feature_names)
    actual_popsize = max(1, population_size // n_features)

    eval_model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", random_state=seed,
    )

    best_result = {"auc": 0, "cols": []}

    def de_objective(position):
        top_k_idx = np.argsort(position)[-k:]
        selected_cols = [feature_names[j] for j in top_k_idx]
        auc = evaluate_feature_subset(X_train, X_val, y_train, y_val, selected_cols, eval_model)
        if auc > best_result["auc"]:
            best_result["auc"] = auc
            best_result["cols"] = selected_cols
        return 1 - auc

    bounds = [(0, 1)] * n_features

    differential_evolution(
        de_objective, bounds=bounds,
        maxiter=generations, popsize=actual_popsize,
        mutation=0.8, recombination=0.9,
        seed=seed, tol=1e-4, polish=False, disp=False
    )

    return best_result["cols"]


def sweep_k_optimization(dataset_name, seed=42):
    """
    Sweep k values for the Optimization FS paradigm.
    For each k: run PSO (top-k) and DE (top-k) individually, evaluate with XGBoost.
    Pick the k where both methods perform well.
    """
    print(f"\n{'='*60}")
    print(f" Validation Sweep: Choosing optimal k for Optimization FS")
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
    # Smaller range for optimization since each k is expensive
    k_values = [3, 5, 7, 10, 13, 15, 18, 20, 25, 30]
    k_values = [k for k in k_values if k < n_total]
    print(f"k values to sweep: {k_values}\n")

    # ── Sweep k values ────────────────────────────────────────────
    results = []

    for k in k_values:
        print(f"\n  k={k:>3d}")

        # PSO: select top-k
        print(f"    Running PSO (k={k})...", end="", flush=True)
        t0 = time.time()
        pso_features = run_pso_for_k(
            X_train, X_val, y_train, y_val, feature_names, k, seed=seed
        )
        pso_time = time.time() - t0
        print(f" {pso_time:.1f}s")

        # DE: select top-k
        print(f"    Running DE  (k={k})...", end="", flush=True)
        t0 = time.time()
        de_features = run_de_for_k(
            X_train, X_val, y_train, y_val, feature_names, k, seed=seed
        )
        de_time = time.time() - t0
        print(f" {de_time:.1f}s")

        # Evaluate PSO top-k
        pso_auc, pso_f1 = evaluate_topk(X_train, X_val, y_train, y_val, pso_features, seed)

        # Evaluate DE top-k
        de_auc, de_f1 = evaluate_topk(X_train, X_val, y_train, y_val, de_features, seed)

        # Average
        avg_auc = (pso_auc + de_auc) / 2
        avg_f1  = (pso_f1 + de_f1) / 2

        print(f"    PSO: AUC={pso_auc:.4f} F1={pso_f1:.4f}  |  "
              f"DE: AUC={de_auc:.4f} F1={de_f1:.4f}  |  "
              f"Avg: AUC={avg_auc:.4f} F1={avg_f1:.4f}")

        results.append({
            "k": k,
            "pso_auc": round(pso_auc, 4),
            "pso_f1": round(pso_f1, 4),
            "de_auc": round(de_auc, 4),
            "de_f1": round(de_f1, 4),
            "avg_auc": round(avg_auc, 4),
            "avg_f1": round(avg_f1, 4),
            "pso_time": round(pso_time, 2),
            "de_time": round(de_time, 2),
        })

    results_df = pd.DataFrame(results)

    # ── Save results ──────────────────────────────────────────────
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "k_sweep_optimization.csv", index=False)
    print(f"\nSaved sweep results to {tables_dir / 'k_sweep_optimization.csv'}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC plot
    ax1.plot(results_df["k"], results_df["pso_auc"],
             "o-", color="#2563eb", linewidth=2, markersize=5, label="PSO")
    ax1.plot(results_df["k"], results_df["de_auc"],
             "s-", color="#16a34a", linewidth=2, markersize=5, label="DE")
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
    ax2.plot(results_df["k"], results_df["pso_f1"],
             "o-", color="#2563eb", linewidth=2, markersize=5, label="PSO")
    ax2.plot(results_df["k"], results_df["de_f1"],
             "s-", color="#16a34a", linewidth=2, markersize=5, label="DE")
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

    fig.suptitle("Optimization FS: k Selection (PSO and DE evaluated individually with XGBoost)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "k_sweep_optimization.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plots_dir / 'k_sweep_optimization.png'}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Best k by Avg AUC: k={int(best_avg_auc_row['k'])}  "
          f"(PSO={best_avg_auc_row['pso_auc']:.4f}, DE={best_avg_auc_row['de_auc']:.4f}, "
          f"Avg={best_avg_auc_row['avg_auc']:.4f})")
    print(f"Best k by Avg F1:  k={int(best_avg_f1_row['k'])}  "
          f"(PSO={best_avg_f1_row['pso_f1']:.4f}, DE={best_avg_f1_row['de_f1']:.4f}, "
          f"Avg={best_avg_f1_row['avg_f1']:.4f})")
    print(f"{'─'*50}\n")

    return results_df