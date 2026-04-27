import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, rankdata
from itertools import combinations
import yaml
from pathlib import Path

from src.utils.paths import BASE_DIR, CONFIG_DIR


def critical_difference_nemenyi(k, n, alpha=0.05):
    """
    Compute the Nemenyi critical difference.
    CD = q_alpha * sqrt(k*(k+1) / (6*n))
    """
    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q = q_table.get(k)
    if q is None:
        raise ValueError(f"q_alpha not available for k={k}. Max supported: {max(q_table.keys())}")
    return q * np.sqrt(k * (k + 1) / (6 * n))


def run_friedman_nemenyi(dataset_name):
    """
    Run Friedman test + Nemenyi post-hoc on repeated experiment results.
    Automatically detects all fs_method groups from the CSV.
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

    # Auto-detect all paradigms/aggregation strategies
    all_methods = sorted(df["fs_method"].unique())
    classifiers = sorted(df["model"].unique())
    metrics = ["test_f1_score", "test_auc"]

    print(f"Detected methods: {all_methods}")
    print(f"Detected classifiers: {classifiers}")
    print(f"Number of seeds: {df['seed'].nunique()}")

    all_results = []

    for metric in metrics:
        metric_display = metric.replace("test_", "").replace("_", " ").upper()
        print(f"\n{'='*60}")
        print(f" Metric: {metric_display}")
        print(f"{'='*60}")

        for clf in classifiers:
            print(f"\n  Classifier: {clf}")

            # Collect scores per method across seeds
            scores = {}
            for m in all_methods:
                subset = df[(df["fs_method"] == m) & (df["model"] == clf)]
                vals = subset.sort_values("seed")[metric].values
                if len(vals) > 0:
                    scores[m] = vals

            method_list = list(scores.keys())
            k = len(method_list)
            n_seeds = len(list(scores.values())[0])

            print(f"  Groups: {k}, Observations per group: {n_seeds}")
            for m, vals in scores.items():
                print(f"    {m}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

            # ── Friedman test ─────────────────────────────────
            if k < 3:
                print("  Skipping Friedman: need at least 3 groups")
                continue

            groups = [scores[m] for m in method_list]
            stat, p_value = friedmanchisquare(*groups)
            sig = p_value < 0.05

            print(f"\n  Friedman: chi2={stat:.4f}, p={p_value:.4f} "
                  f"{'→ SIGNIFICANT' if sig else '→ not significant'}")

            all_results.append({
                "metric": metric_display,
                "classifier": clf,
                "test": "Friedman",
                "comparison": "All methods",
                "statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "significant": sig,
            })

            # ── Nemenyi post-hoc ──────────────────────────────
            if sig:
                # Compute average ranks
                rank_matrix = np.zeros((n_seeds, k))
                for i in range(n_seeds):
                    seed_scores = [scores[m][i] for m in method_list]
                    rank_matrix[i] = rankdata([-s for s in seed_scores])

                avg_ranks = rank_matrix.mean(axis=0)

                print(f"\n  Average ranks (lower = better):")
                for j, m in enumerate(method_list):
                    print(f"    {m}: {avg_ranks[j]:.2f}")

                # Critical difference
                try:
                    cd = critical_difference_nemenyi(k, n_seeds, alpha=0.05)
                    print(f"\n  Nemenyi CD (alpha=0.05): {cd:.4f}")

                    # Pairwise comparisons
                    print(f"\n  Pairwise rank differences (showing only significant):")
                    sig_count = 0
                    for (i, m1), (j, m2) in combinations(enumerate(method_list), 2):
                        diff = abs(avg_ranks[i] - avg_ranks[j])
                        is_sig = diff > cd

                        all_results.append({
                            "metric": metric_display,
                            "classifier": clf,
                            "test": "Nemenyi",
                            "comparison": f"{m1} vs {m2}",
                            "statistic": round(diff, 4),
                            "p_value": round(cd, 4),
                            "significant": is_sig,
                        })

                        if is_sig:
                            better = m1 if avg_ranks[i] < avg_ranks[j] else m2
                            print(f"    {m1} vs {m2}: |{avg_ranks[i]:.2f} - {avg_ranks[j]:.2f}| = "
                                  f"{diff:.2f} > {cd:.2f} → {better} is better")
                            sig_count += 1

                    if sig_count == 0:
                        print("    No significant pairwise differences found.")
                    else:
                        print(f"\n    Total significant pairs: {sig_count} / {k*(k-1)//2}")

                except ValueError as e:
                    print(f"\n  Cannot compute Nemenyi CD: {e}")
                    print(f"  With k={k} groups, Nemenyi q_alpha table only goes up to k=10.")
                    print(f"  Consider comparing subsets of methods instead.")
            else:
                print("  Nemenyi post-hoc skipped (Friedman not significant)")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(tables_dir / "significance_tests.csv", index=False)
    print(f"\nSaved results to {tables_dir / 'significance_tests.csv'}")

    return results_df


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "ibtracs.ALL.list.v04r01"
    results_df = run_friedman_nemenyi(dataset_name)
    print("\nDone!")