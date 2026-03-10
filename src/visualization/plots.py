from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils.paths import BASE_DIR, CONFIG_DIR
import yaml
import matplotlib.cm as cm

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