import pandas as pd
import numpy as np
import time
import yaml
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
import shap
import lime
import lime.lime_tabular

from src.utils.paths import BASE_DIR, CONFIG_DIR
from src.utils.data_io import load_splits

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def evaluate_topk(X_train, X_val, y_train, y_val, features, seed=42):
    """Train XGBoost on given features and return val AUC and F1."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X_train[features], y_train)
    y_pred  = model.predict(X_val[features])
    y_proba = model.predict_proba(X_val[features])[:, 1]
    return roc_auc_score(y_val, y_proba), f1_score(y_val, y_pred)


def compute_shap_ranking(X_train, model, seed=42, max_samples=10000):
    """Compute SHAP feature importance ranking (descending)."""
    np.random.seed(seed)

    if len(X_train) > max_samples:
        idx  = np.random.choice(len(X_train), max_samples, replace=False)
        X_bg = X_train.iloc[idx]
    else:
        X_bg = X_train

    explainer   = shap.TreeExplainer(model, X_bg)
    shap_values = explainer.shap_values(X_bg)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = np.argsort(mean_abs_shap)[::-1]
    return ranking, mean_abs_shap


def compute_lime_ranking(X_train, model, seed=42, n_samples=500, max_samples=10000):
    """Compute LIME feature importance ranking (descending)."""
    np.random.seed(seed)

    if len(X_train) > max_samples:
        idx      = np.random.choice(len(X_train), max_samples, replace=False)
        X_sample = X_train.iloc[idx]
    else:
        X_sample = X_train

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_sample.values,
        feature_names=X_train.columns.tolist(),
        class_names=["no_landfall", "landfall"],
        mode="classification",
        random_state=seed
    )

    n_explain  = min(n_samples, len(X_sample))
    sample_idx = np.random.choice(len(X_sample), n_explain, replace=False)

    feature_weights = {col: [] for col in X_train.columns}

    print(f"    Computing LIME explanations ({n_explain} samples)...")
    for i, idx in enumerate(sample_idx):
        if i % 100 == 0 and i > 0:
            print(f"      LIME progress: {i}/{n_explain}")
        exp = explainer.explain_instance(
            X_sample.values[idx],
            model.predict_proba,
            num_features=len(X_train.columns)
        )
        for feat, weight in exp.as_list():
            matched = [col for col in X_train.columns if col in feat]
            if matched:
                feature_weights[matched[0]].append(abs(weight))

    mean_weights = np.array([
        np.mean(feature_weights[col]) if feature_weights[col] else 0.0
        for col in X_train.columns
    ])

    ranking = np.argsort(mean_weights)[::-1]
    return ranking, mean_weights


def sweep_k_xai(dataset_name, seed=42):
    """
    Sweep k values for the XAI FS paradigm.
    Compute SHAP and LIME rankings once, then for each k:
    evaluate SHAP top-k and LIME top-k individually with XGBoost.
    """
    print(f"\n{'='*60}")
    print(f" Validation Sweep: Choosing optimal k for XAI FS")
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

    # ── Pre-train explainer model ─────────────────────────────────
    print("Pre-training XGBoost explainer model...")
    explainer_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=seed,
    )
    explainer_model.fit(X_train, y_train)
    print("Explainer model trained.\n")

    # ── Compute SHAP ranking once ─────────────────────────────────
    print("Computing SHAP feature rankings...")
    t0 = time.time()
    shap_ranking, shap_scores = compute_shap_ranking(X_train, explainer_model, seed=seed)
    shap_time = time.time() - t0
    print(f"SHAP ranking computed in {shap_time:.1f}s\n")

    # ── Compute LIME ranking once ─────────────────────────────────
    print("Computing LIME feature rankings...")
    t0 = time.time()
    lime_ranking, lime_scores = compute_lime_ranking(X_train, explainer_model, seed=seed)
    lime_time = time.time() - t0
    print(f"LIME ranking computed in {lime_time:.1f}s\n")

    # ── Sweep k values ────────────────────────────────────────────
    results = []

    for k in k_values:
        print(f"  k={k:>3d} ... ", end="", flush=True)

        # SHAP: top-k features
        shap_features = [feature_names[i] for i in shap_ranking[:k]]

        # LIME: top-k features
        lime_features = [feature_names[i] for i in lime_ranking[:k]]

        # Evaluate SHAP top-k
        shap_auc, shap_f1 = evaluate_topk(X_train, X_val, y_train, y_val, shap_features, seed)

        # Evaluate LIME top-k
        lime_auc, lime_f1 = evaluate_topk(X_train, X_val, y_train, y_val, lime_features, seed)

        # Average
        avg_auc = (shap_auc + lime_auc) / 2
        avg_f1  = (shap_f1 + lime_f1) / 2

        print(f"SHAP: AUC={shap_auc:.4f} F1={shap_f1:.4f}  |  "
              f"LIME: AUC={lime_auc:.4f} F1={lime_f1:.4f}  |  "
              f"Avg: AUC={avg_auc:.4f} F1={avg_f1:.4f}")

        results.append({
            "k": k,
            "shap_auc": round(shap_auc, 4),
            "shap_f1": round(shap_f1, 4),
            "lime_auc": round(lime_auc, 4),
            "lime_f1": round(lime_f1, 4),
            "avg_auc": round(avg_auc, 4),
            "avg_f1": round(avg_f1, 4),
        })

    results_df = pd.DataFrame(results)

    # ── Save results ──────────────────────────────────────────────
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "k_sweep_xai.csv", index=False)
    print(f"\nSaved sweep results to {tables_dir / 'k_sweep_xai.csv'}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC plot
    ax1.plot(results_df["k"], results_df["shap_auc"],
             "o-", color="#2563eb", linewidth=2, markersize=5, label="SHAP")
    ax1.plot(results_df["k"], results_df["lime_auc"],
             "s-", color="#16a34a", linewidth=2, markersize=5, label="LIME")
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
    ax2.plot(results_df["k"], results_df["shap_f1"],
             "o-", color="#2563eb", linewidth=2, markersize=5, label="SHAP")
    ax2.plot(results_df["k"], results_df["lime_f1"],
             "s-", color="#16a34a", linewidth=2, markersize=5, label="LIME")
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

    fig.suptitle("XAI FS: k Selection (SHAP and LIME evaluated individually with XGBoost)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "k_sweep_xai.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plots_dir / 'k_sweep_xai.png'}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Best k by Avg AUC: k={int(best_avg_auc_row['k'])}  "
          f"(SHAP={best_avg_auc_row['shap_auc']:.4f}, LIME={best_avg_auc_row['lime_auc']:.4f}, "
          f"Avg={best_avg_auc_row['avg_auc']:.4f})")
    print(f"Best k by Avg F1:  k={int(best_avg_f1_row['k'])}  "
          f"(SHAP={best_avg_f1_row['shap_f1']:.4f}, LIME={best_avg_f1_row['lime_f1']:.4f}, "
          f"Avg={best_avg_f1_row['avg_f1']:.4f})")
    print(f"{'─'*50}\n")

    return results_df