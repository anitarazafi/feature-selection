from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils.paths import BASE_DIR, CONFIG_DIR
import yaml
import matplotlib.cm as cm
import json

def plot_learning_curves(models, X_train, y_train, dataset_name, results_dir):
    fig, axes = plt.subplots(1, len(models), figsize=(18, 5))
    fig.suptitle(f"Learning Curves — {dataset_name}")

    for ax, (model_name, model) in zip(axes, models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=3,
            scoring="roc_auc",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
        ax.plot(train_sizes, val_scores.mean(axis=1),   label="Val")
        ax.fill_between(train_sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        ax.fill_between(train_sizes,
            val_scores.mean(axis=1) - val_scores.std(axis=1),
            val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
        ax.set_title(model_name)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("AUC")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {plot_dir / 'learning_curves.png'}")


def plot_mlp_loss(model, results_dir):
    if not hasattr(model, "loss_curve_"):
        print("No loss curve available")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, label="Train loss")
    if hasattr(model, "validation_scores_"):
        plt.plot(model.validation_scores_, label="Val score")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("MLP Loss Curve")
    plt.legend()
    plt.grid(True)

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "mlp_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_auc_curves(dataset_name):
    """
    Plot AUC vs n_features for each method, per model.
    """
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg         = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    tables_dir  = results_dir / "tables"
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tables_dir / "all_results.csv")

    # Exclude baseline from curves (it has no k variation)
    df_fs = df[df["fs_method"] != "baseline"].copy()

    models      = df_fs["model"].unique()
    methods     = df_fs["fs_method"].unique()
    colors      = cm.tab10(np.linspace(0, 1, len(methods)))
    method_color = dict(zip(methods, colors))

    # ── 1. AUC vs n_features — one subplot per model ──────────────
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(f"AUC vs Number of Features — {dataset_name}", fontsize=13)

    for ax, model_name in zip(axes, models):
        model_df = df_fs[df_fs["model"] == model_name]

        for method in methods:
            method_df = (
                model_df[model_df["fs_method"] == method]
                .sort_values("n_features")
            )
            if method_df.empty:
                continue
            ax.plot(
                method_df["n_features"],
                method_df["test_auc"],
                marker="o",
                label=method,
                color=method_color[method]
            )

        # Add baseline as horizontal dashed line
        baseline = df[
            (df["model"] == model_name) &
            (df["fs_method"] == "baseline")
        ]["test_auc"].values
        if len(baseline) > 0:
            ax.axhline(
                y=baseline[0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="baseline"
            )

        ax.set_title(model_name)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "auc_vs_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {plots_dir / 'auc_vs_features.png'}")

    # ── 2. AUC vs n_features — one subplot per method ─────────────
    fig, axes = plt.subplots(
        1, len(methods),
        figsize=(5 * len(methods), 5),
        sharey=True
    )
    if len(methods) == 1:
        axes = [axes]
    fig.suptitle(f"AUC per Method — {dataset_name}", fontsize=13)

    model_colors = cm.Set2(np.linspace(0, 1, len(models)))
    model_color  = dict(zip(models, model_colors))

    for ax, method in zip(axes, methods):
        method_df = df_fs[df_fs["fs_method"] == method]

        for model_name in models:
            mdf = (
                method_df[method_df["model"] == model_name]
                .sort_values("n_features")
            )
            if mdf.empty:
                continue
            ax.plot(
                mdf["n_features"],
                mdf["test_auc"],
                marker="o",
                label=model_name,
                color=model_color[model_name]
            )

        # Baseline per model
        for model_name in models:
            baseline = df[
                (df["model"] == model_name) &
                (df["fs_method"] == "baseline")
            ]["test_auc"].values
            if len(baseline) > 0:
                ax.axhline(
                    y=baseline[0],
                    color=model_color[model_name],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.5
                )

        ax.set_title(method)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "auc_per_method.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {plots_dir / 'auc_per_method.png'}")

    # ── 3. F1 vs n_features — one subplot per model ───────────────
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(f"F1 vs Number of Features — {dataset_name}", fontsize=13)

    for ax, model_name in zip(axes, models):
        model_df = df_fs[df_fs["model"] == model_name]

        for method in methods:
            method_df = (
                model_df[model_df["fs_method"] == method]
                .sort_values("n_features")
            )
            if method_df.empty:
                continue
            ax.plot(
                method_df["n_features"],
                method_df["test_f1_score"],
                marker="o",
                label=method,
                color=method_color[method]
            )

        baseline = df[
            (df["model"] == model_name) &
            (df["fs_method"] == "baseline")
        ]["test_f1_score"].values
        if len(baseline) > 0:
            ax.axhline(
                y=baseline[0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="baseline"
            )

        ax.set_title(model_name)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("F1 Score")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "f1_vs_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {plots_dir / 'f1_vs_features.png'}")


def plot_heatmap(dataset_name):
    """
    Heatmap of AUC and F1 — method × model.
    Best per method across all k values.
    """
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg         = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    tables_dir  = results_dir / "tables"
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(tables_dir / "best_per_method.csv")

    pivot_auc = best.pivot(
        index="fs_method", columns="model", values="test_auc"
    ).round(4)

    pivot_f1 = best.pivot(
        index="fs_method", columns="model", values="test_f1_score"
    ).round(4)

    # Sort rows — baseline first, then alphabetical
    method_order = ["baseline"] + sorted(
        [m for m in pivot_auc.index if m != "baseline"]
    )
    pivot_auc = pivot_auc.reindex(method_order)
    pivot_f1  = pivot_f1.reindex(method_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Feature Selection Comparison — {dataset_name}", fontsize=13)

    for ax, pivot, title in zip(
        axes,
        [pivot_auc, pivot_f1],
        ["AUC (best k per method)", "F1 Score (best k per method)"]
    ):
        im = ax.imshow(
            pivot.values.astype(float),
            cmap="RdYlGn",
            aspect="auto",
            vmin=float(np.nanmin(pivot.values)) - 0.01,
            vmax=float(np.nanmax(pivot.values)) + 0.01
        )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_title(title, fontsize=11)

        # Annotate each cell
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(float(val)):
                    # Bold the best value per column
                    col_max = np.nanmax(pivot.values[:, j])
                    weight  = "bold" if float(val) == float(col_max) else "normal"
                    ax.text(
                        j, i, f"{float(val):.4f}",
                        ha="center", va="center",
                        fontsize=9, fontweight=weight,
                        color="black"
                    )

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(plots_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {plots_dir / 'heatmap.png'}")

def load_all_features(results_dir, k):
    """Load selected feature lists for a given k across all methods."""
    method_map = {
        "mutual_information": "traditional",
        "chi2":               "traditional",
        "rfe":                "traditional",
        "pso":                "optimization",
        "differential_evolution": "optimization",
        "shap":               "xai",
        "lime":               "xai",
    }
    features = {}
    for method, subdir in method_map.items():
        path = results_dir / "selectors" / subdir / f"{method}_k{k}_features.json"
        if path.exists():
            with open(path) as f:
                features[method] = set(json.load(f))
        else:
            print(f"  Missing: {path.name}")
    return features


def plot_feature_overlap(dataset_name):
    """
    Feature overlap analysis across all methods for each k.
    Produces:
      1. Heatmap of Jaccard similarity between methods
      2. Bar chart of feature frequency (how many methods selected each feature)
    """
    import json
    from itertools import combinations

    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg         = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_DIR / "fs.yaml", "r") as f:
        fs_cfg = yaml.safe_load(f)
    ks = fs_cfg["common"]["n_features_to_select"]

    for k in ks:
        print(f"\n── k={k} ──────────────────────────────────")
        features = load_all_features(results_dir, k)
        if not features:
            print(f"  No features found for k={k}, skipping")
            continue

        methods = list(features.keys())

        # ── 1. Jaccard similarity heatmap ─────────────────────────
        n = len(methods)
        jaccard_matrix = np.zeros((n, n))
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                s1, s2 = features[m1], features[m2]
                intersection = len(s1 & s2)
                union        = len(s1 | s2)
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"Feature Overlap Analysis — {dataset_name} (k={k})",
            fontsize=13
        )

        ax = axes[0]
        im = ax.imshow(jaccard_matrix, cmap="YlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_title("Jaccard Similarity Between Methods")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{jaccard_matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if jaccard_matrix[i, j] < 0.7 else "white")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # ── 2. Feature frequency bar chart ────────────────────────
        from collections import Counter
        all_selected = [f for fset in features.values() for f in fset]
        freq         = Counter(all_selected)
        # Sort by frequency descending, take top 20
        top_features = sorted(freq.items(), key=lambda x: -x[1])[:20]
        feat_names   = [f[0] for f in top_features]
        feat_counts  = [f[1] for f in top_features]

        ax2 = axes[1]
        bars = ax2.barh(
            feat_names[::-1], feat_counts[::-1],
            color=cm.YlGn(np.array(feat_counts[::-1]) / len(methods))
        )
        ax2.set_xlabel("Number of methods that selected this feature")
        ax2.set_title(f"Feature Selection Frequency (top 20, k={k})")
        ax2.axvline(x=len(methods), color="red", linestyle="--",
                    linewidth=1, label="all methods")
        ax2.axvline(x=len(methods) * 0.5, color="orange", linestyle="--",
                    linewidth=1, label="50% of methods")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="x")

        # Annotate bars with count
        for bar, count in zip(bars, feat_counts[::-1]):
            ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     str(count), va="center", fontsize=8)

        plt.tight_layout()
        fname = plots_dir / f"feature_overlap_k{k}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved to {fname}")

        # ── 3. Print consensus features ───────────────────────────
        consensus = [f for f, c in freq.items() if c == len(methods)]
        majority  = [f for f, c in freq.items() if c >= len(methods) * 0.5 and c < len(methods)]
        print(f"  Consensus features (all {len(methods)} methods): {consensus}")
        print(f"  Majority features  (≥50% methods):              {majority}")


def plot_roc_curves(dataset_name):
    """
    ROC curves for best k per method, one subplot per model.
    Loads predicted probabilities from predictions JSON files.
    """
    import json
    from sklearn.metrics import roc_curve, auc

    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg         = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    tables_dir  = results_dir / "tables"
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load best k per method from summary table
    best = pd.read_csv(tables_dir / "best_per_method.csv")

    # Method → subfolder mapping
    subdir_map = {
        "mutual_information":    "traditional",
        "chi2":                  "traditional",
        "rfe":                   "traditional",
        "pso":                   "optimization",
        "differential_evolution":"optimization",
        "shap":                  "xai",
        "lime":                  "xai",
    }

    models  = best["model"].unique()
    methods = [m for m in best["fs_method"].unique() if m != "baseline"]

    # Colors per method — baseline always black
    colors = cm.tab10(np.linspace(0, 1, len(methods)))
    method_color = dict(zip(methods, colors))

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(f"ROC Curves (best k per method) — {dataset_name}", fontsize=13)

    for ax, model_name in zip(axes, models):

        # ── Baseline first ────────────────────────────────────────
        baseline_path = results_dir / "predictions" / "baseline" / f"{model_name}_n63_predictions.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                preds = json.load(f)
            y_true  = preds["test"]["y_true"]
            y_proba = preds["test"]["y_pred_proba"]
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="black", linestyle="--",
                    linewidth=2, label=f"baseline (AUC={roc_auc:.4f})")

        # ── Each method at its best k ─────────────────────────────
        for method in methods:
            row = best[
                (best["model"] == model_name) &
                (best["fs_method"] == method)
            ]
            if row.empty:
                continue

            k      = int(row["n_features"].values[0])
            subdir = subdir_map.get(method, "traditional")
            path   = (results_dir / "predictions" / subdir /
                      method / f"{model_name}_n{k}_predictions.json")

            if not path.exists():
                print(f"  Missing: {path}")
                continue

            with open(path) as f:
                preds = json.load(f)

            y_true  = preds["test"]["y_true"]
            y_proba = preds["test"]["y_pred_proba"]
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc     = auc(fpr, tpr)

            ax.plot(fpr, tpr,
                    color=method_color[method],
                    linewidth=1.8,
                    label=f"{method} k={k} (AUC={roc_auc:.4f})")

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {plots_dir / 'roc_curves.png'}")