import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from itertools import combinations
import yaml
from pathlib import Path

from src.utils.paths import BASE_DIR, CONFIG_DIR


def critical_difference_nemenyi(k, n, alpha=0.05):
    """
    Compute the Nemenyi critical difference.
    CD = q_alpha * sqrt(k*(k+1) / (6*n))
    where k = number of groups, n = number of observations.
    q_alpha values from Demšar (2006) Table of critical values
    for the two-tailed Nemenyi test.
    """
    # q_alpha values for alpha=0.05 (from studentized range / sqrt(2))
    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q = q_table.get(k)
    if q is None:
        raise ValueError(f"q_alpha not available for k={k}")
    return q * np.sqrt(k * (k + 1) / (6 * n))


def run_friedman_nemenyi(dataset_name):
    """
    Run Friedman test + Nemenyi post-hoc on repeated experiment results.
    Compares Baseline, FS-Set, Op-Set, X-Set across 10 seeds.
    """
    print(f"\n{'='*60}")
    print(f" Friedman Test + Nemenyi Post-Hoc Analysis")
    print(f"{'='*60}\n")

    # Load data
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    tables_dir = results_dir / "tables"

    df = pd.read_csv(tables_dir / "all_seeds_combined.csv")

    paradigms = ["baseline", "fs_set", "op_set", "x_set"]
    classifiers = df["model"].unique()
    metrics = ["test_f1_score", "test_auc"]

    all_results = []

    for metric in metrics:
        metric_display = metric.replace("test_", "").replace("_", " ").upper()
        print(f"\n{'─'*60}")
        print(f" Metric: {metric_display}")
        print(f"{'─'*60}")

        for clf in classifiers:
            print(f"\n  Classifier: {clf}")

            # Collect scores per paradigm across seeds
            scores = {}
            for p in paradigms:
                subset = df[(df["fs_method"] == p) & (df["model"] == clf)]
                vals = subset.sort_values("seed")[metric].values
                if len(vals) > 0:
                    scores[p] = vals

            n_seeds = len(list(scores.values())[0])
            k = len(scores)

            print(f"  Groups: {k}, Observations per group: {n_seeds}")
            for p, vals in scores.items():
                print(f"    {p}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

            # ── Friedman test ─────────────────────────────────
            groups = [scores[p] for p in paradigms if p in scores]
            stat, p_value = friedmanchisquare(*groups)
            sig = p_value < 0.05

            print(f"\n  Friedman: chi2={stat:.4f}, p={p_value:.4f} "
                  f"{'→ SIGNIFICANT' if sig else '→ not significant'}")

            all_results.append({
                "metric": metric_display,
                "classifier": clf,
                "test": "Friedman",
                "comparison": "All paradigms",
                "statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "significant": sig,
            })

            # ── Nemenyi post-hoc (only if Friedman is significant) ──
            if sig:
                # Compute average ranks
                rank_matrix = np.zeros((n_seeds, k))
                paradigm_list = [p for p in paradigms if p in scores]

                for i in range(n_seeds):
                    seed_scores = [scores[p][i] for p in paradigm_list]
                    # Rank: higher score = better = lower rank number
                    order = np.argsort(seed_scores)[::-1]
                    ranks = np.zeros(k)
                    for rank_pos, idx in enumerate(order):
                        ranks[idx] = rank_pos + 1
                    rank_matrix[i] = ranks

                avg_ranks = rank_matrix.mean(axis=0)

                print(f"\n  Average ranks (lower = better):")
                for j, p in enumerate(paradigm_list):
                    print(f"    {p}: {avg_ranks[j]:.2f}")

                # Critical difference
                cd = critical_difference_nemenyi(k, n_seeds, alpha=0.05)
                print(f"\n  Nemenyi CD (alpha=0.05): {cd:.4f}")

                # Pairwise comparisons
                print(f"\n  Pairwise rank differences:")
                for (i, p1), (j, p2) in combinations(enumerate(paradigm_list), 2):
                    diff = abs(avg_ranks[i] - avg_ranks[j])
                    is_sig = diff > cd
                    print(f"    {p1} vs {p2}: |{avg_ranks[i]:.2f} - {avg_ranks[j]:.2f}| = {diff:.2f} "
                          f"{'> ' if is_sig else '< '}{cd:.2f} "
                          f"{'→ SIGNIFICANT' if is_sig else '→ not significant'}")

                    all_results.append({
                        "metric": metric_display,
                        "classifier": clf,
                        "test": "Nemenyi",
                        "comparison": f"{p1} vs {p2}",
                        "statistic": round(diff, 4),
                        "p_value": round(cd, 4),
                        "significant": is_sig,
                    })
            else:
                print("  Nemenyi post-hoc skipped (Friedman not significant)")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(tables_dir / "significance_tests.csv", index=False)
    print(f"\nSaved results to {tables_dir / 'significance_tests.csv'}")

    # Generate LaTeX table
    generate_significance_latex(results_df, tables_dir)

    return results_df


def generate_significance_latex(results_df, tables_dir):
    """Generate LaTeX table for significance test results."""

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical Significance Tests (Friedman + Nemenyi)}")
    lines.append(r"\label{tab:significance_tests}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{lllccc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Classifier & Comparison & Statistic & CD / p-value & Sig. \\")
    lines.append(r"\midrule")

    current_metric = None
    for _, row in results_df.iterrows():
        # Add separator between metrics
        if current_metric is not None and row["metric"] != current_metric:
            lines.append(r"\midrule")
        current_metric = row["metric"]

        clf_display = row["classifier"].replace("_", r"\_")
        comp_display = row["comparison"].replace("_", r"\_")

        if row["test"] == "Friedman":
            stat_str = f"$\\chi^2$={row['statistic']:.2f}"
            pval_str = f"p={row['p_value']:.4f}"
        else:
            stat_str = f"$\\Delta R$={row['statistic']:.2f}"
            pval_str = f"CD={row['p_value']:.2f}"

        sig_str = r"\checkmark" if row["significant"] else ""

        lines.append(
            f"{row['metric']} & {clf_display} & {comp_display} & "
            f"{stat_str} & {pval_str} & {sig_str} \\\\"
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
    return latex_str


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "ibtracs.ALL.list.v04r01"
    results_df = run_friedman_nemenyi(dataset_name)
    print("\nDone!")