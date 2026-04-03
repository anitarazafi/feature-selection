import pandas as pd
import yaml
from pathlib import Path

from src.utils.paths import BASE_DIR, CONFIG_DIR


def generate_comparison_table(dataset_name):
    """
    Read baseline, traditional, optimization, and xai CSV results,
    extract best F1 and AUC per paradigm, and generate a LaTeX comparison table.
    """
    # Load config
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]
    tables_dir = results_dir / "tables"

    # Define paradigms and their CSV files
    paradigms = {
        "Baseline": "baseline.csv",
        "FS-Set":   "traditional.csv",
        "Op-Set":   "optimization.csv",
        "X-Set":    "xai.csv",
    }

    # Read each CSV and extract best F1, best AUC, and n_features
    summary = {}
    for name, csv_file in paradigms.items():
        csv_path = tables_dir / csv_file
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping {name}")
            continue

        df = pd.read_csv(csv_path)
        summary[name] = {
            "best_f1":     df["test_f1_score"].max(),
            "best_auc":    df["test_auc"].max(),
            "n_features":  int(df["n_features"].iloc[0]),
        }

    # Find which paradigm has the best F1 and best AUC
    best_f1_paradigm  = max(summary, key=lambda k: summary[k]["best_f1"])
    best_auc_paradigm = max(summary, key=lambda k: summary[k]["best_auc"])

    # Build LaTeX table
    paradigm_names = list(summary.keys())

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best Test-Set Results per Paradigm (across all classifiers)}")
    lines.append(r"\label{tab:comparison_summary}")
    lines.append(r"\begin{tabular}{l" + "c" * len(paradigm_names) + "}")
    lines.append(r"\toprule")

    # Header row 1: paradigm names
    header1 = " & " + " & ".join(paradigm_names) + r" \\"
    lines.append(header1)

    # Header row 2: number of features
    feat_cells = []
    for name in paradigm_names:
        n = summary[name]["n_features"]
        feat_cells.append(f"({n})")
    header2 = " & " + " & ".join(feat_cells) + r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Best F1 row
    f1_cells = []
    for name in paradigm_names:
        val = summary[name]["best_f1"]
        if name == best_f1_paradigm:
            f1_cells.append(f"\\textbf{{{val:.4f}}}")
        else:
            f1_cells.append(f"{val:.4f}")
    lines.append("Best F1 & " + " & ".join(f1_cells) + r" \\")

    # Best AUC row
    auc_cells = []
    for name in paradigm_names:
        val = summary[name]["best_auc"]
        if name == best_auc_paradigm:
            auc_cells.append(f"\\textbf{{{val:.4f}}}")
        else:
            auc_cells.append(f"{val:.4f}")
    lines.append("Best AUC & " + " & ".join(auc_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    # Save
    output_path = tables_dir / "comparison_summary.tex"
    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"Saved comparison table to {output_path}")
    print(f"\nSummary:")
    for name in paradigm_names:
        s = summary[name]
        print(f"  {name} ({s['n_features']} feat.): best F1={s['best_f1']:.4f}, best AUC={s['best_auc']:.4f}")
    print(f"\n  Best F1:  {best_f1_paradigm} ({summary[best_f1_paradigm]['best_f1']:.4f})")
    print(f"  Best AUC: {best_auc_paradigm} ({summary[best_auc_paradigm]['best_auc']:.4f})")

    return latex_str