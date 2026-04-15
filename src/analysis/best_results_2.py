import pandas as pd
import yaml
from pathlib import Path

from src.utils.paths import BASE_DIR, CONFIG_DIR


def generate_comparison_table_2(dataset_name):
    """
    Read summary_mean_std.csv and generate a comparison table
    showing best F1 and AUC per paradigm with mean ± std.
    """
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    tables_dir = results_dir / "tables"

    df = pd.read_csv(tables_dir / "summary_mean_std.csv")

    paradigms = {
        "baseline": "Baseline",
        "fs_set":   "FS-Set",
        "op_set":   "Op-Set",
        "x_set":    "X-Set",
    }

    # Feature count ranges from all_seeds_combined
    combined_path = tables_dir / "all_seeds_combined.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        feat_labels = {}
        for key, label in paradigms.items():
            subset = combined[combined["fs_method"] == key]
            if not subset.empty:
                min_f = int(subset["n_features"].min())
                max_f = int(subset["n_features"].max())
                if min_f == max_f:
                    feat_labels[key] = str(min_f)
                else:
                    feat_labels[key] = f"{min_f}--{max_f}"
    else:
        feat_labels = {key: str(int(df[df["fs_method"] == key]["n_features"].iloc[0]))
                       for key in paradigms}

    # Extract best F1 and AUC per paradigm
    summary = {}
    for key in paradigms:
        subset = df[df["fs_method"] == key]
        summary[key] = {
            "best_f1_mean": subset["test_f1_score_mean"].max(),
            "best_f1_std":  subset.loc[subset["test_f1_score_mean"].idxmax(), "test_f1_score_std"],
            "best_auc_mean": subset["test_auc_mean"].max(),
            "best_auc_std":  subset.loc[subset["test_auc_mean"].idxmax(), "test_auc_std"],
        }

    best_f1_key  = max(summary, key=lambda k: summary[k]["best_f1_mean"])
    best_auc_key = max(summary, key=lambda k: summary[k]["best_auc_mean"])

    # Build LaTeX
    keys = list(paradigms.keys())

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best Test-Set Results per Paradigm (mean $\pm$ std, 10 runs)}")
    lines.append(r"\label{tab:comparison_summary}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{l" + "c" * len(keys) + "}")
    lines.append(r"\toprule")

    # Header row 1: paradigm names
    header1 = " & " + " & ".join(paradigms[k] for k in keys) + r" \\"
    lines.append(header1)

    # Header row 2: feature counts
    header2 = " & " + " & ".join(f"({feat_labels[k]})" for k in keys) + r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Best F1 row
    f1_cells = []
    for k in keys:
        mean = summary[k]["best_f1_mean"]
        std = summary[k]["best_f1_std"]
        val = f"{mean:.4f}$\\pm${std:.4f}"
        if k == best_f1_key:
            val = r"\textbf{" + val + "}"
        f1_cells.append(val)
    lines.append("Best F1-score & " + " & ".join(f1_cells) + r" \\")

    # Best AUC row
    auc_cells = []
    for k in keys:
        mean = summary[k]["best_auc_mean"]
        std = summary[k]["best_auc_std"]
        val = f"{mean:.4f}$\\pm${std:.4f}"
        if k == best_auc_key:
            val = r"\textbf{" + val + "}"
        auc_cells.append(val)
    lines.append("Best AUC-ROC & " + " & ".join(auc_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    output_path = tables_dir / "comparison_summary.tex"
    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"Saved comparison table to {output_path}")
    print(f"\nSummary:")
    for k in keys:
        s = summary[k]
        print(f"  {paradigms[k]} ({feat_labels[k]} feat.): "
              f"F1={s['best_f1_mean']:.4f}±{s['best_f1_std']:.4f}, "
              f"AUC={s['best_auc_mean']:.4f}±{s['best_auc_std']:.4f}")
    print(f"\n  Best F1:  {paradigms[best_f1_key]}")
    print(f"  Best AUC: {paradigms[best_auc_key]}")

    return latex_str