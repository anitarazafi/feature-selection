import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from src.visualization.plots import plot_learning_curves, plot_mlp_loss
import time
import json
from pathlib import Path

import yaml
from src.utils.paths import BASE_DIR, CONFIG_DIR
from src.utils.data_io import load_splits
from src.utils.load_models import load_models

def generate_latex_table(results_df, output_path, caption="Baseline Performance Metrics", label="tab:baseline_results"):
    """Generate a LaTeX table from the results DataFrame and save it to a .tex file."""

    display_cols = {
        "model":          "Model",
        "n_features":     "\\# Features",
        "test_accuracy":  "Acc",
        "test_precision": "Prec",
        "test_recall":    "Rec",
        "test_f1_score":  "F1",
        "test_auc":       "AUC",
        "train_time":     "Time (s)",
    }

    df = results_df[list(display_cols.keys())].copy()

    n_cols = len(display_cols)
    col_fmt = "l" + "c" * (n_cols - 1)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    header = " & ".join(display_cols.values()) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells = []
        for col in display_cols.keys():
            val = row[col]
            if col == "model":
                cells.append(str(val).replace("_", r"\_"))
            elif col == "n_features":
                cells.append(str(int(val)))
            elif col == "train_time":
                cells.append(f"{val:.2f}")
            else:
                cells.append(f"{val:.4f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"Saved LaTeX table to {output_path}")
    return latex_str

def train_baseline(dataset_name, seed=42):
    print(f"\n{'='*60}")
    print(f"= Training baseline models for: {dataset_name} (seed={seed})")

    configs = CONFIG_DIR / "datasets.yaml"
    with open(configs, "r") as f:
        all_configs = yaml.safe_load(f)
    
    if dataset_name not in all_configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in config. Available: {list(all_configs.keys())}")
    
    cfg = all_configs[dataset_name]

    splits_rel = cfg["paths"]["splits"]
    splits_dir = BASE_DIR / splits_rel
    results_rel = cfg["paths"]["results"]
    results_dir = BASE_DIR / results_rel

    splits = load_splits(splits_dir)
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]
    y_test  = splits["y_test"]

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Features: {X_train.shape[1]}\n")

    results = []
    n_features = X_train.shape[1]

    MODELS = load_models(seed=seed)

    for model_name, model in MODELS.items():
        print(f"Training {model_name}...")

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        def get_metrics(y_true, y_pred, y_proba):
            return {
                "accuracy":  accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall":    recall_score(y_true, y_pred),
                "f1_score":  f1_score(y_true, y_pred),
                "auc":       roc_auc_score(y_true, y_proba),
            }

        # Val metrics
        y_val_pred   = model.predict(X_val)
        y_val_proba  = model.predict_proba(X_val)[:, 1]
        val_metrics  = get_metrics(y_val, y_val_pred, y_val_proba)

        # Test metrics
        y_test_pred  = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = get_metrics(y_test, y_test_pred, y_test_proba)

        print(f"  Training time: {train_time:.2f}s")
        print(f"  {'Metric':<12} {'Val':>8} {'Test':>8}")
        print(f"  {'-'*30}")
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            print(f"  {metric:<12} {val_metrics[metric]:>8.4f} {test_metrics[metric]:>8.4f}")
        print()

        # Save predictions
        pred_dir = results_dir / "predictions" / "baseline"
        pred_dir.mkdir(parents=True, exist_ok=True)
        predictions = {
            "val": {
                "y_true": y_val.tolist(),
                "y_pred": y_val_pred.tolist(),
                "y_pred_proba": y_val_proba.tolist(),
            },
            "test": {
                "y_true": y_test.tolist(),
                "y_pred": y_test_pred.tolist(),
                "y_pred_proba": y_test_proba.tolist(),
            },
            "model": model_name,
            "n_features": n_features,
            "seed": seed
        }
        with open(pred_dir / f"{model_name}_n{n_features}_seed{seed}_predictions.json", "w") as f:
            json.dump(predictions, f)

        # Save model
        model_dir = results_dir / "models" / "baseline"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / f"{model_name}_seed{seed}.pkl", "wb") as f:
            pickle.dump(model, f)

        results.append({
            "model":      model_name,
            "fs_method":     "baseline",
            "n_features": n_features,
            "seed":       seed,
            # val metrics
            "val_accuracy":  round(val_metrics["accuracy"],  4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall":    round(val_metrics["recall"],    4),
            "val_f1_score":  round(val_metrics["f1_score"],  4),
            "val_auc":       round(val_metrics["auc"],       4),            
            # test metrics
            "test_accuracy":  round(test_metrics["accuracy"],  4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall":    round(test_metrics["recall"],    4),
            "test_f1_score":  round(test_metrics["f1_score"],  4),
            "test_auc":       round(test_metrics["auc"],       4),
            "train_time": round(train_time, 2),
        })

    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / f"baseline_seed{seed}.csv", index=False)
    print(f"\nSaved results to {tables_dir / f'baseline_seed{seed}.csv'}")
    print(f"Saved models to {results_dir / 'models' / 'baseline'}\n")

    # Generate LaTeX table
    generate_latex_table(
        results_df,
        output_path=tables_dir / f"baseline_seed{seed}.tex",
        caption="Baseline Classification Performance (All Features)",
        label="tab:baseline_results",
    )

    # After training all models
    #plot_learning_curves(MODELS, X_train, y_train, dataset_name, results_dir)

    # MLP specific
    #if "mlp" in MODELS:
    #    plot_mlp_loss(MODELS["mlp"], results_dir)

    return results_df