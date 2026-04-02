import pandas as pd
import pickle
import time
import json
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, RFE

from src.utils.paths import BASE_DIR, CONFIG_DIR
from src.utils.data_io import load_splits
from src.utils.load_models import load_models


def get_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_proba),
    }


def apply_mutual_information(X_train, X_val, X_test, y_train, k):
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    selected_cols = X_train.columns[selector.get_support()]
    return X_train[selected_cols], X_val[selected_cols], X_test[selected_cols], selector


def apply_chi2(X_train, X_val, X_test, y_train, k):
    X_train_abs = X_train.abs()
    X_val_abs   = X_val.abs()
    X_test_abs  = X_test.abs()
    selector = SelectKBest(chi2, k=k)
    selector.fit(X_train_abs, y_train)
    selected_cols = X_train.columns[selector.get_support()]
    return X_train[selected_cols], X_val[selected_cols], X_test[selected_cols], selector


def apply_rfe(X_train, X_val, X_test, y_train, estimator, k, cfg):
    step = cfg.get("step", 5)
    selector = RFE(estimator=estimator, n_features_to_select=k, step=step)
    selector.fit(X_train, y_train)
    selected_cols = X_train.columns[selector.get_support()]
    return X_train[selected_cols], X_val[selected_cols], X_test[selected_cols], selector


def train_and_evaluate(MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                       fs_name, n_features, selected_features,
                       results_dir, fs_time, results):
    for model_name, model in MODELS.items():
        print(f"  Training {model_name}...")

        start_time = time.time()
        model.fit(X_tr, y_train)
        train_time = time.time() - start_time

        y_val_pred   = model.predict(X_v)
        y_val_proba  = model.predict_proba(X_v)[:, 1]
        val_metrics  = get_metrics(y_val, y_val_pred, y_val_proba)

        y_test_pred  = model.predict(X_te)
        y_test_proba = model.predict_proba(X_te)[:, 1]
        test_metrics = get_metrics(y_test, y_test_pred, y_test_proba)

        print(f"    Training time: {train_time:.2f}s")
        print(f"    {'Metric':<12} {'Val':>8} {'Test':>8}")
        print(f"    {'-'*30}")
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            print(f"    {metric:<12} {val_metrics[metric]:>8.4f} {test_metrics[metric]:>8.4f}")
        print()

        # Save predictions
        pred_dir = results_dir / "predictions" / "traditional" / fs_name
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
            "selected_features": selected_features,
        }
        with open(pred_dir / f"{model_name}_n{n_features}_predictions.json", "w") as f:
            json.dump(predictions, f)

        # Save model
        model_dir = results_dir / "models" / "traditional" / fs_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / f"{model_name}_n{n_features}.pkl", "wb") as f:
            pickle.dump(model, f)

        results.append({
            "model":         model_name,
            "fs_method":     fs_name,
            "n_features":    n_features,
            "fs_time":       round(fs_time, 2),
            "val_accuracy":  round(val_metrics["accuracy"],  4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall":    round(val_metrics["recall"],    4),
            "val_f1_score":  round(val_metrics["f1_score"],  4),
            "val_auc":       round(val_metrics["auc"],       4),
            "test_accuracy":  round(test_metrics["accuracy"],  4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall":    round(test_metrics["recall"],    4),
            "test_f1_score":  round(test_metrics["f1_score"],  4),
            "test_auc":       round(test_metrics["auc"],       4),
            "train_time":    round(train_time, 2),
        })


def generate_latex_table(results_df, output_path, caption, label):
    """Generate a LaTeX table for performance metrics."""

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


def generate_feature_selection_latex(selected_sets, fs_set, output_path,
                                     caption, label):
    """Generate a LaTeX table showing selected features per method and the union."""

    all_features = sorted(fs_set)
    methods = list(selected_sets.keys())

    # Column format: l for feature name, c for each method, c for union
    col_fmt = "l" + "c" * (len(methods) + 1)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # Header row
    method_headers = [m.replace("_", r"\_") for m in methods]
    header = "Feature & " + " & ".join(method_headers) + r" & FS-Set \\"
    lines.append(header)
    lines.append(r"\midrule")

    # One row per feature
    for feat in all_features:
        feat_display = feat.replace("_", r"\_")
        cells = [feat_display]
        for method in methods:
            if feat in selected_sets[method]:
                cells.append(r"\checkmark")
            else:
                cells.append("")
        # Union column — always checkmark since all_features comes from fs_set
        cells.append(r"\checkmark")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # Summary row with totals
    count_cells = [r"\textbf{Total}"]
    for method in methods:
        count_cells.append(str(len(selected_sets[method])))
    count_cells.append(str(len(fs_set)))
    lines.append(" & ".join(count_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"Saved feature selection LaTeX table to {output_path}")
    return latex_str


def run_traditional_fs(dataset_name):
    print(f"\n{'='*60}")
    print(f"= Running Traditional Feature Selection for: {dataset_name}")
    print(f"{'='*60}\n")

    # Load dataset config
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    if dataset_name not in all_configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets.yaml")
    cfg = all_configs[dataset_name]

    # Load fs config
    from src.utils.load_fs_config import load_fs_config
    fs_cfg = load_fs_config()

    traditional_cfg      = fs_cfg.get("traditional", {})
    k                    = fs_cfg["common"]["n_features_to_select"]

    splits_dir  = BASE_DIR / cfg["paths"]["splits"]
    results_dir = BASE_DIR / cfg["paths"]["results"]

    splits  = load_splits(splits_dir)
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]
    y_test  = splits["y_test"]

    print(f"Training samples:    {len(X_train)}")
    print(f"Validation samples:  {len(X_val)}")
    print(f"Test samples:        {len(X_test)}")
    print(f"Features (original): {X_train.shape[1]}")
    print(f"k (features to select): {k}\n")

    MODELS  = load_models()
    results = []

    # Store selected feature sets for union
    selected_sets = {}
    fs_times = {}

    # ── 1. Mutual Information (selection only) ────────────────────
    mi_cfg = traditional_cfg.get("mutual_information", {})
    if mi_cfg.get("enabled", False):
        print(f"\n{'─'*60}")
        print(f"Feature Selection: MUTUAL INFORMATION (k={k})")
        print(f"{'─'*60}")

        fs_start = time.time()
        _, _, _, selector = apply_mutual_information(
            X_train, X_val, X_test, y_train, k=k
        )
        fs_time = time.time() - fs_start
        selected_features = X_train.columns[selector.get_support()].tolist()
        selected_sets["mutual_information"] = set(selected_features)
        fs_times["mutual_information"] = fs_time

        print(f"Selected features ({k}): {selected_features}")
        print(f"Feature selection time: {fs_time:.2f}s")

        selector_dir = results_dir / "selectors" / "traditional"
        selector_dir.mkdir(parents=True, exist_ok=True)
        with open(selector_dir / f"mutual_information_k{k}_selector.pkl", "wb") as f:
            pickle.dump(selector, f)
        with open(selector_dir / f"mutual_information_k{k}_features.json", "w") as f:
            json.dump(selected_features, f)

    # ── 2. Chi-Square (selection only) ────────────────────────────
    chi2_cfg = traditional_cfg.get("chi2", {})
    if chi2_cfg.get("enabled", False):
        print(f"\n{'─'*60}")
        print(f"Feature Selection: CHI-SQUARE (k={k})")
        print(f"{'─'*60}")

        fs_start = time.time()
        _, _, _, selector = apply_chi2(
            X_train, X_val, X_test, y_train, k=k
        )
        fs_time = time.time() - fs_start
        selected_features = X_train.columns[selector.get_support()].tolist()
        selected_sets["chi2"] = set(selected_features)
        fs_times["chi2"] = fs_time

        print(f"Selected features ({k}): {selected_features}")
        print(f"Feature selection time: {fs_time:.2f}s")

        selector_dir = results_dir / "selectors" / "traditional"
        selector_dir.mkdir(parents=True, exist_ok=True)
        with open(selector_dir / f"chi2_k{k}_selector.pkl", "wb") as f:
            pickle.dump(selector, f)
        with open(selector_dir / f"chi2_k{k}_features.json", "w") as f:
            json.dump(selected_features, f)

    # ── 3. RFE (selection only) ───────────────────────────────────
    rfe_cfg = traditional_cfg.get("rfe", {})
    if rfe_cfg.get("enabled", False):
        estimator_name = rfe_cfg.get("estimator", "random_forest")
        estimator      = MODELS[estimator_name]

        print(f"\n{'─'*60}")
        print(f"Feature Selection: RFE (k={k}, estimator={estimator_name})")
        print(f"{'─'*60}")

        fs_start = time.time()
        _, _, _, selector = apply_rfe(
            X_train, X_val, X_test, y_train,
            estimator=estimator, k=k, cfg=rfe_cfg
        )
        fs_time = time.time() - fs_start
        selected_features = X_train.columns[selector.get_support()].tolist()
        selected_sets["rfe"] = set(selected_features)
        fs_times["rfe"] = fs_time

        print(f"Selected features ({k}): {selected_features}")
        print(f"Feature selection time: {fs_time:.2f}s")

        selector_dir = results_dir / "selectors" / "traditional"
        selector_dir.mkdir(parents=True, exist_ok=True)
        with open(selector_dir / f"rfe_k{k}_selector.pkl", "wb") as f:
            pickle.dump(selector, f)
        with open(selector_dir / f"rfe_k{k}_features.json", "w") as f:
            json.dump(selected_features, f)

    # ── 4. FS-Set (Union) — Train only on this ───────────────────
    if selected_sets:
        print(f"\n{'─'*60}")
        print(f"Constructing FS-Set (Union of Traditional Methods)")
        print(f"{'─'*60}")

        fs_set = sorted(set.union(*selected_sets.values()))
        n_fs_set = len(fs_set)
        total_fs_time = sum(fs_times.values())

        print(f"Individual method contributions:")
        for method, feats in selected_sets.items():
            print(f"  {method}: {sorted(feats)}")
        print(f"\nFS-Set (union): {fs_set}")
        print(f"FS-Set size: {n_fs_set}")
        print(f"Total FS time: {total_fs_time:.2f}s\n")

        # Save FS-Set info
        selector_dir = results_dir / "selectors" / "traditional"
        selector_dir.mkdir(parents=True, exist_ok=True)
        fs_set_info = {
            "fs_set": fs_set,
            "n_features": n_fs_set,
            "contributing_methods": {m: sorted(f) for m, f in selected_sets.items()},
            "fs_times": {m: round(t, 2) for m, t in fs_times.items()},
        }
        with open(selector_dir / "fs_set.json", "w") as f:
            json.dump(fs_set_info, f, indent=2)

        # Train and evaluate on FS-Set only
        X_tr_fs  = X_train[fs_set]
        X_v_fs   = X_val[fs_set]
        X_te_fs  = X_test[fs_set]

        train_and_evaluate(
            MODELS, X_tr_fs, X_v_fs, X_te_fs, y_train, y_val, y_test,
            "fs_set", n_fs_set, fs_set,
            results_dir, total_fs_time, results
        )

    # ── Save results ──────────────────────────────────────────────
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / "traditional.csv", index=False)
    print(f"\nSaved results to {tables_dir / 'traditional.csv'}")
    print(f"Saved models to {results_dir / 'models' / 'traditional'}")

    method_names = ", ".join(m.replace("_", " ").title() for m in selected_sets.keys())
    caption = f"Classification Performance with Traditional FS ({method_names} $\\rightarrow$ FS-Set)"

    # Generate performance LaTeX table
    generate_latex_table(
        results_df,
        output_path=tables_dir / "traditional.tex",
        caption=caption,
        label="tab:traditional_fs_results",
    )

    # Generate feature selection LaTeX table
    generate_feature_selection_latex(
        selected_sets, fs_set,
        output_path=tables_dir / "traditional_features.tex",
        caption="Features Selected by Traditional Methods",
        label="tab:traditional_fs_features",
    )

    return results_df