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

        print(f"  Training time: {train_time:.2f}s")
        print(f"  {'Metric':<12} {'Val':>8} {'Test':>8}")
        print(f"  {'-'*30}")
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            print(f"  {metric:<12} {val_metrics[metric]:>8.4f} {test_metrics[metric]:>8.4f}")
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

    traditional_cfg         = fs_cfg.get("traditional", {})
    n_features_to_select    = fs_cfg["common"]["n_features_to_select"]

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
    print(f"Features (original): {X_train.shape[1]}\n")

    MODELS  = load_models()
    results = []

    # ── 1. Mutual Information (run for each k) ────────────────────
    mi_cfg = traditional_cfg.get("mutual_information", {})
    if mi_cfg.get("enabled", False):
        for k in n_features_to_select:
            if k >= X_train.shape[1]:
                print(f"Skipping mutual_information k={k}: exceeds available features")
                continue

            print(f"\n{'─'*60}")
            print(f"Feature Selection: MUTUAL INFORMATION (k={k})")
            print(f"{'─'*60}")

            fs_start = time.time()
            X_tr, X_v, X_te, selector = apply_mutual_information(
                X_train, X_val, X_test, y_train, k=k
            )
            fs_time = time.time() - fs_start
            selected_features = X_tr.columns.tolist()

            print(f"Selected features: {k} / {X_train.shape[1]}")
            print(f"Feature selection time: {fs_time:.2f}s\n")

            selector_dir = results_dir / "selectors" / "traditional"
            selector_dir.mkdir(parents=True, exist_ok=True)
            with open(selector_dir / f"mutual_information_k{k}_selector.pkl", "wb") as f:
                pickle.dump(selector, f)
            with open(selector_dir / f"mutual_information_k{k}_features.json", "w") as f:
                json.dump(selected_features, f)

            train_and_evaluate(
                MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                "mutual_information", k, selected_features,
                results_dir, fs_time, results
            )

    # ── 2. Chi-Square (run for each k) ────────────────────────────
    chi2_cfg = traditional_cfg.get("chi2", {})
    if chi2_cfg.get("enabled", False):
        for k in n_features_to_select:
            if k >= X_train.shape[1]:
                print(f"Skipping chi2 k={k}: exceeds available features")
                continue

            print(f"\n{'─'*60}")
            print(f"Feature Selection: CHI2 (k={k})")
            print(f"{'─'*60}")

            fs_start = time.time()
            X_tr, X_v, X_te, selector = apply_chi2(
                X_train, X_val, X_test, y_train, k=k
            )
            fs_time = time.time() - fs_start
            selected_features = X_tr.columns.tolist()

            print(f"Selected features: {k} / {X_train.shape[1]}")
            print(f"Feature selection time: {fs_time:.2f}s\n")

            selector_dir = results_dir / "selectors" / "traditional"
            selector_dir.mkdir(parents=True, exist_ok=True)
            with open(selector_dir / f"chi2_k{k}_selector.pkl", "wb") as f:
                pickle.dump(selector, f)
            with open(selector_dir / f"chi2_k{k}_features.json", "w") as f:
                json.dump(selected_features, f)

            train_and_evaluate(
                MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                "chi2", k, selected_features,
                results_dir, fs_time, results
            )

    # ── 3. RFE (run for each k) ───────────────────────────────────
    rfe_cfg = traditional_cfg.get("rfe", {})
    if rfe_cfg.get("enabled", False):
        estimator_name = rfe_cfg.get("estimator", "random_forest")
        estimator      = MODELS[estimator_name]

        for k in n_features_to_select:
            if k >= X_train.shape[1]:
                print(f"Skipping RFE k={k}: exceeds available features")
                continue

            print(f"\n{'─'*60}")
            print(f"Feature Selection: RFE (k={k}, estimator={estimator_name})")
            print(f"{'─'*60}")

            fs_start = time.time()
            X_tr, X_v, X_te, selector = apply_rfe(
                X_train, X_val, X_test, y_train,
                estimator=estimator, k=k, cfg=rfe_cfg
            )
            fs_time = time.time() - fs_start
            selected_features = X_tr.columns.tolist()

            print(f"Selected features: {k} / {X_train.shape[1]}")
            print(f"Feature selection time: {fs_time:.2f}s\n")

            selector_dir = results_dir / "selectors" / "traditional"
            selector_dir.mkdir(parents=True, exist_ok=True)
            with open(selector_dir / f"rfe_k{k}_selector.pkl", "wb") as f:
                pickle.dump(selector, f)
            with open(selector_dir / f"rfe_k{k}_features.json", "w") as f:
                json.dump(selected_features, f)

            train_and_evaluate(
                MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                "rfe", k, selected_features,
                results_dir, fs_time, results
            )

    # Save results
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / "traditional.csv", index=False)
    print(f"\nSaved results to {tables_dir / 'traditional.csv'}")
    print(f"Saved models to {results_dir / 'models' / 'traditional'}\n")

    return results_df