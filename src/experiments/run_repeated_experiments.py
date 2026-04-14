import pandas as pd
import numpy as np
import json
import yaml
import time
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

from src.utils.paths import BASE_DIR, CONFIG_DIR
from src.experiments.run_baseline import train_baseline
from src.experiments.run_traditional_fs import run_traditional_fs
from src.experiments.run_optimization_fs import run_optimization_fs
from src.experiments.run_xai_fs import run_xai_fs


#SEEDS = [42, 123, 456, 789, 1024]

SEEDS = [42, 123, 456, 789, 1024, 2023, 2024, 3333, 5555, 9999]

def run_all_experiments(dataset_name):
    """
    Run baseline + 3 FS paradigms across multiple seeds.
    Collect all results, compute mean ± std, and run significance tests.
    """
    print(f"\n{'#'*60}")
    print(f"# REPEATED EXPERIMENTS: {len(SEEDS)} seeds")
    print(f"# Seeds: {SEEDS}")
    print(f"# Dataset: {dataset_name}")
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
        df = run_traditional_fs(dataset_name, seed=seed)
        all_results.append(df)

        # Optimization FS
        df = run_optimization_fs(dataset_name, seed=seed)
        all_results.append(df)

        # XAI FS
        df = run_xai_fs(dataset_name, seed=seed)
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

    # Generate LaTeX table with mean ± std
    generate_mean_std_latex(summary_df, tables_dir)

    # Run statistical tests
    #run_significance_tests(combined_df, tables_dir)

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


def generate_mean_std_latex(summary_df, tables_dir):
    """Generate LaTeX tables with mean ± std for each paradigm."""

    paradigms = {
        "baseline": ("Baseline Classification Performance (mean $\\pm$ std)", "tab:baseline_mean_std"),
        "fs_set":   ("Traditional FS Performance (mean $\\pm$ std)", "tab:traditional_mean_std"),
        "op_set":   ("Optimization-Based FS Performance (mean $\\pm$ std)", "tab:optimization_mean_std"),
        "x_set":    ("XAI-Based FS Performance (mean $\\pm$ std)", "tab:xai_mean_std"),
    }

    metrics = [
        ("test_accuracy",  "Acc"),
        ("test_precision", "Prec"),
        ("test_recall",    "Rec"),
        ("test_f1_score",  "F1"),
        ("test_auc",       "AUC"),
    ]

    for fs_method, (caption, label) in paradigms.items():
        subset = summary_df[summary_df["fs_method"] == fs_method]
        if subset.empty:
            continue

        col_fmt = "l" + "c" * (len(metrics) + 1)

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + caption + "}")
        lines.append(r"\label{" + label + "}")
        lines.append(r"\resizebox{\columnwidth}{!}{")
        lines.append(r"\begin{tabular}{" + col_fmt + "}")
        lines.append(r"\toprule")

        header = "Model & \\# Feat. & " + " & ".join(h for _, h in metrics) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        for _, row in subset.iterrows():
            cells = [row["model"].replace("_", r"\_")]
            cells.append(str(int(row["n_features"])))
            for metric, _ in metrics:
                mean = row[f"{metric}_mean"]
                std  = row[f"{metric}_std"]
                cells.append(f"{mean:.4f}$\\pm${std:.4f}")
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\end{table}")

        latex_str = "\n".join(lines)

        output_path = tables_dir / f"{fs_method}_mean_std.tex"
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"Saved mean±std LaTeX table to {output_path}")


def run_significance_tests(df, tables_dir):
    """
    Run Friedman test + pairwise Wilcoxon signed-rank tests
    comparing paradigms across seeds for each classifier.
    """
    paradigms = ["baseline", "fs_set", "op_set", "x_set"]
    classifiers = df["model"].unique()
    metrics = ["test_f1_score", "test_auc"]

    all_test_results = []

    for metric in metrics:
        print(f"\n{'─'*60}")
        print(f"Statistical Tests for: {metric}")
        print(f"{'─'*60}")

        for clf in classifiers:
            print(f"\n  Classifier: {clf}")

            # Collect scores per paradigm across seeds
            paradigm_scores = {}
            for p in paradigms:
                subset = df[(df["fs_method"] == p) & (df["model"] == clf)]
                scores = subset.sort_values("seed")[metric].values
                if len(scores) == len(SEEDS):
                    paradigm_scores[p] = scores

            if len(paradigm_scores) < 2:
                print(f"    Skipping: not enough paradigms with {len(SEEDS)} seeds")
                continue

            # Friedman test (requires 3+ groups)
            if len(paradigm_scores) >= 3:
                groups = list(paradigm_scores.values())
                stat, p_value = friedmanchisquare(*groups)
                print(f"    Friedman test: stat={stat:.4f}, p={p_value:.4f} "
                      f"{'*' if p_value < 0.05 else '(n.s.)'}")

                all_test_results.append({
                    "metric": metric, "classifier": clf,
                    "test": "friedman",
                    "statistic": round(stat, 4),
                    "p_value": round(p_value, 4),
                    "significant": p_value < 0.05,
                })

            # Pairwise Wilcoxon signed-rank tests
            for (p1, p2) in combinations(paradigm_scores.keys(), 2):
                scores1 = paradigm_scores[p1]
                scores2 = paradigm_scores[p2]

                # Wilcoxon requires non-zero differences
                diffs = scores1 - scores2
                if np.all(diffs == 0):
                    print(f"    Wilcoxon {p1} vs {p2}: identical results, skipping")
                    all_test_results.append({
                        "metric": metric, "classifier": clf,
                        "test": f"wilcoxon_{p1}_vs_{p2}",
                        "statistic": 0, "p_value": 1.0,
                        "significant": False,
                    })
                    continue

                try:
                    stat, p_value = wilcoxon(scores1, scores2)
                    sig = p_value < 0.05
                    print(f"    Wilcoxon {p1} vs {p2}: stat={stat:.4f}, p={p_value:.4f} "
                          f"{'*' if sig else '(n.s.)'}")
                    all_test_results.append({
                        "metric": metric, "classifier": clf,
                        "test": f"wilcoxon_{p1}_vs_{p2}",
                        "statistic": round(stat, 4),
                        "p_value": round(p_value, 4),
                        "significant": sig,
                    })
                except ValueError as e:
                    print(f"    Wilcoxon {p1} vs {p2}: error — {e}")

    # Save test results
    test_df = pd.DataFrame(all_test_results)
    test_df.to_csv(tables_dir / "significance_tests.csv", index=False)
    print(f"\nSaved significance tests to {tables_dir / 'significance_tests.csv'}")

    # Generate LaTeX table for significance results
    generate_significance_latex(test_df, tables_dir)

    return test_df


def generate_significance_latex(test_df, tables_dir):
    """Generate a LaTeX table summarizing significance test results."""

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical Significance Tests (Friedman + Wilcoxon)}")
    lines.append(r"\label{tab:significance_tests}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{llllcc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Classifier & Test & Comparison & Statistic & p-value \\")
    lines.append(r"\midrule")

    for _, row in test_df.iterrows():
        metric_display = row["metric"].replace("test_", "").replace("_", " ").upper()
        clf_display = row["classifier"].replace("_", r"\_")
        test_name = row["test"]

        if test_name == "friedman":
            comparison = "All paradigms"
        else:
            parts = test_name.replace("wilcoxon_", "").split("_vs_")
            comparison = f"{parts[0]} vs {parts[1]}" if len(parts) == 2 else test_name

        comparison = comparison.replace("_", r"\_")
        p_str = f"{row['p_value']:.4f}"
        if row["significant"]:
            p_str = r"\textbf{" + p_str + "}"

        lines.append(
            f"{metric_display} & {clf_display} & "
            f"{test_name.split('_')[0].title()} & {comparison} & "
            f"{row['statistic']:.4f} & {p_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    output_path = tables_dir / "significance_tests.tex"
    with open(output_path, "w") as f:
        f.write(latex_str)
    print(f"Saved significance LaTeX table to {output_path}")


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "ibtracs.ALL.list.v04r01"
    combined_df, summary_df = run_all_experiments(dataset_name)
    print("\nDone! All repeated experiments complete.")