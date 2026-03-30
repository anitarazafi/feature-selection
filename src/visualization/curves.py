import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

from src.utils.paths import BASE_DIR, CONFIG_DIR


def plot_roc_curves(dataset_name):
    """
    Generate one ROC curve plot per classifier, each showing
    Baseline, FS-Set, Op-Set, and X-Set curves.
    """
    print(f"\n{'='*60}")
    print(f" Generating ROC Curves for: {dataset_name}")
    print(f"{'='*60}")

    # Load config
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    cfg = all_configs[dataset_name]
    results_dir = BASE_DIR / cfg["paths"]["results"]

    # Define the paradigms and their prediction paths
    classifiers = ["random_forest", "xgboost", "mlp"]
    classifier_display = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "mlp": "MLP",
    }

    # Map paradigm name -> (prediction subdirectory, fs_name, label, color, linestyle)
    paradigms = {
        "Baseline (63 feat.)": {
            "path": results_dir / "predictions" / "baseline",
            "pattern": "{model}_n63_predictions.json",
            "color": "#888888",
            "linestyle": "--",
        },
        "FS-Set (16 feat.)": {
            "path": results_dir / "predictions" / "traditional" / "fs_set",
            "pattern": "{model}_n16_predictions.json",
            "color": "#4A9E4A",
            "linestyle": "-",
        },
        "Op-Set (11 feat.)": {
            "path": results_dir / "predictions" / "optimization" / "op_set",
            "pattern": "{model}_n11_predictions.json",
            "color": "#C47A2A",
            "linestyle": "-",
        },
        "X-Set (12 feat.)": {
            "path": results_dir / "predictions" / "xai" / "x_set",
            "pattern": "{model}_n12_predictions.json",
            "color": "#7B5EA7",
            "linestyle": "-",
        },
    }

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_name in classifiers:
        fig, ax = plt.subplots(figsize=(5, 4.5))

        for paradigm_label, info in paradigms.items():
            pred_file = info["path"] / info["pattern"].format(model=model_name)

            if not pred_file.exists():
                # Try to find the file by scanning the directory
                pred_dir = info["path"]
                if pred_dir.exists():
                    matches = list(pred_dir.glob(f"{model_name}_n*_predictions.json"))
                    if matches:
                        pred_file = matches[0]
                    else:
                        print(f"  Warning: No prediction file found for {model_name} in {pred_dir}")
                        continue
                else:
                    print(f"  Warning: Directory not found: {pred_dir}")
                    continue

            with open(pred_file, "r") as f:
                preds = json.load(f)

            y_true = np.array(preds["test"]["y_true"])
            y_proba = np.array(preds["test"]["y_pred_proba"])

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr,
                    color=info["color"],
                    linestyle=info["linestyle"],
                    linewidth=1.8,
                    label=f'{paradigm_label} (AUC = {roc_auc:.4f})')

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve — {classifier_display[model_name]}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(plots_dir / f"roc_{model_name}.png", dpi=200, bbox_inches='tight')
        fig.savefig(plots_dir / f"roc_{model_name}.pdf", dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved ROC curve for {classifier_display[model_name]}")

    print(f"\nAll ROC curves saved to {plots_dir}")