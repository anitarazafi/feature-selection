import pandas as pd
import numpy as np
import yaml
from scipy.stats import friedmanchisquare, rankdata
from pathlib import Path

from src.utils.paths import BASE_DIR, CONFIG_DIR


def generate_compact_significance_table(dataset_name):
    """
    Generate a compact average ranks table for Friedman + Nemenyi results.
    """
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    tables_dir = results_dir / "tables"

    df = pd.read_csv(tables_dir / "all_seeds_combined.csv")

    paradigms = ["baseline", "fs_set", "op_set", "x_set"]
    paradigm_display = {
        "baseline": "Baseline",
        "fs_set": "FS-Set",
        "op_set": "Op-Set",
        "x_set": "X-Set",
    }
    classifiers = ["random_forest", "xgboost", "mlp"]
    clf_display = {
        "random_forest": "RF",
        "xgboost": "XGBoost",
        "mlp": "MLP",
    }
    metrics = [
        ("test_f1_score", "F1-score"),
        ("test_auc", "AUC-ROC"),
    ]

    k = len(paradigms)
    rows = []

    for metric_key, metric_name in metrics:
        for clf in classifiers:
            scores = {}
            for p in paradigms:
                subset = df[(df["fs_method"] == p) & (df["model"] == clf)]
                scores[p] = subset.sort_values("seed")[metric_key].values

            n_seeds = len(scores[paradigms[0]])

            # Compute average ranks (rank 1 = best)
            rank_matrix = np.zeros((n_seeds, k))
            for i in range(n_seeds):
                seed_scores = [scores[p][i] for p in paradigms]
                # Higher score = better = rank 1
                rank_matrix[i] = rankdata([-s for s in seed_scores])

            avg_ranks = rank_matrix.mean(axis=0)

            # Friedman test
            groups = [scores[p] for p in paradigms]
            stat, p_value = friedmanchisquare(*groups)

            rows.append({
                "metric": metric_name,
                "classifier": clf_display[clf],
                "ranks": {paradigm_display[p]: avg_ranks[j] for j, p in enumerate(paradigms)},
                "friedman_stat": stat,
                "friedman_p": p_value,
            })

    # Nemenyi CD
    n_seeds = len(df[df["fs_method"] == "baseline"]["seed"].unique())
    q_alpha = 2.569  # q for k=4, alpha=0.05
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_seeds))

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Average Ranks from Friedman Test (Nemenyi CD = " + f"{cd:.2f}" + r", $\alpha$ = 0.05)}")
    lines.append(r"\label{tab:friedman_ranks}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Classifier & Baseline & FS-Set & Op-Set & X-Set & Friedman $p$ \\")
    lines.append(r"\midrule")

    current_metric = None
    for row in rows:
        if current_metric is not None and row["metric"] != current_metric:
            lines.append(r"\midrule")
        current_metric = row["metric"]

        # Find best rank (lowest)
        best_rank = min(row["ranks"].values())

        cells = [row["metric"], row["classifier"]]
        for p_name in ["Baseline", "FS-Set", "Op-Set", "X-Set"]:
            rank_val = row["ranks"][p_name]
            rank_str = f"{rank_val:.2f}"
            if rank_val == best_rank:
                rank_str = r"\textbf{" + rank_str + "}"
            cells.append(rank_str)

        p_str = f"{row['friedman_p']:.4f}"
        cells.append(p_str)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    output_path = tables_dir / "friedman_ranks.tex"
    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"Saved compact significance table to {output_path}")
    print(f"\nNemenyi CD = {cd:.2f} (pairs with rank difference > CD are significantly different)")
    print(f"\nAverage Ranks:")
    for row in rows:
        rank_strs = ", ".join(f"{k}={v:.2f}" for k, v in row["ranks"].items())
        print(f"  {row['metric']:>8s} | {row['classifier']:>7s} | {rank_strs} | p={row['friedman_p']:.4f}")

    return latex_str