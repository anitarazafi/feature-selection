import pandas as pd
import pickle
import time
import json
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import shap
import lime
import lime.lime_tabular
from copy import deepcopy

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


# ── SHAP ──────────────────────────────────────────────────────────────────────

def apply_shap(X_train, X_val, X_test, model, cfg, k):
    """
    Fit SHAP explainer on train, rank features by mean absolute SHAP value,
    select top-k.
    """
    max_samples = cfg.get("max_samples", 10000)

    # Sample for SHAP computation if dataset is large
    if len(X_train) > max_samples:
        idx     = np.random.choice(len(X_train), max_samples, replace=False)
        X_bg    = X_train.iloc[idx]
    else:
        X_bg    = X_train

    # TreeExplainer is fast for RF and XGBoost
    # For MLP we fall back to KernelExplainer with a smaller background
    if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
        explainer   = shap.TreeExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_bg)
    else:
        # MLP — use smaller background for speed
        background  = shap.sample(X_bg, 100)
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_bg[:200])

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # use class 1 (landfall)

    # Rank by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X_train.columns)
    selected_cols = feature_importance.nlargest(k).index.tolist()

    return (
        X_train[selected_cols],
        X_val[selected_cols],
        X_test[selected_cols],
        selected_cols,
        feature_importance
    )


# ── LIME ──────────────────────────────────────────────────────────────────────

def apply_lime(X_train, X_val, X_test, y_train, model, cfg, k):
    """
    Fit LIME explainer, aggregate local feature importances over a sample,
    select top-k by mean absolute weight.
    """
    n_samples   = cfg.get("n_samples", 500)
    max_samples = cfg.get("max_samples", 10000)

    # Sample instances to explain
    if len(X_train) > max_samples:
        idx      = np.random.choice(len(X_train), max_samples, replace=False)
        X_sample = X_train.iloc[idx]
    else:
        X_sample = X_train

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data  = X_sample.values,
        feature_names  = X_train.columns.tolist(),
        class_names    = ["no_landfall", "landfall"],
        mode           = "classification",
        random_state   = 42
    )

    # Explain a subset of instances
    n_explain  = min(n_samples, len(X_sample))
    sample_idx = np.random.choice(len(X_sample), n_explain, replace=False)

    feature_weights = {col: [] for col in X_train.columns}

    print(f"  Running LIME on {n_explain} samples...")
    for i, idx in enumerate(sample_idx):
        if i % 100 == 0:
            print(f"  LIME progress: {i}/{n_explain}")
        exp = explainer.explain_instance(
            X_sample.values[idx],
            model.predict_proba,
            num_features=len(X_train.columns)
        )
        for feat, weight in exp.as_list():
            # LIME feature names include conditions e.g. "WIND > 50.0"
            # match back to original column name
            matched = [col for col in X_train.columns if col in feat]
            if matched:
                feature_weights[matched[0]].append(abs(weight))

    # Aggregate mean absolute weight per feature
    mean_weights = {
        col: np.mean(weights) if weights else 0.0
        for col, weights in feature_weights.items()
    }
    feature_importance = pd.Series(mean_weights).sort_values(ascending=False)
    selected_cols      = feature_importance.nlargest(k).index.tolist()

    return (
        X_train[selected_cols],
        X_val[selected_cols],
        X_test[selected_cols],
        selected_cols,
        feature_importance
    )


# ── Train & Evaluate ──────────────────────────────────────────────────────────

def train_and_evaluate(MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                       fs_name, n_features, selected_features,
                       results_dir, fs_time, results):
    for model_name, model in MODELS.items():
        print(f"  Training {model_name}...")

        start_time   = time.time()
        model.fit(X_tr, y_train)
        train_time   = time.time() - start_time

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
        pred_dir = results_dir / "predictions" / "xai" / fs_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        with open(pred_dir / f"{model_name}_n{n_features}_predictions.json", "w") as f:
            json.dump({
                "val": {
                    "y_true":      y_val.tolist(),
                    "y_pred":      y_val_pred.tolist(),
                    "y_pred_proba": y_val_proba.tolist(),
                },
                "test": {
                    "y_true":      y_test.tolist(),
                    "y_pred":      y_test_pred.tolist(),
                    "y_pred_proba": y_test_proba.tolist(),
                },
                "model":             model_name,
                "n_features":        n_features,
                "selected_features": selected_features,
            }, f)

        # Save model
        model_dir = results_dir / "models" / "xai" / fs_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / f"{model_name}_n{n_features}.pkl", "wb") as f:
            pickle.dump(model, f)

        results.append({
            "model":          model_name,
            "fs_method":      fs_name,
            "n_features":     n_features,
            "fs_time":        round(fs_time, 2),
            "val_accuracy":   round(val_metrics["accuracy"],  4),
            "val_precision":  round(val_metrics["precision"], 4),
            "val_recall":     round(val_metrics["recall"],    4),
            "val_f1_score":   round(val_metrics["f1_score"],  4),
            "val_auc":        round(val_metrics["auc"],       4),
            "test_accuracy":  round(test_metrics["accuracy"],  4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall":    round(test_metrics["recall"],    4),
            "test_f1_score":  round(test_metrics["f1_score"],  4),
            "test_auc":       round(test_metrics["auc"],       4),
            "train_time":     round(train_time, 2),
        })


# ── Main ──────────────────────────────────────────────────────────────────────

def run_xai_fs(dataset_name):
    print(f"\n{'='*60}")
    print(f"= Running XAI-Based Feature Selection for: {dataset_name}")
    print(f"{'='*60}\n")

    # Load configs
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    if dataset_name not in all_configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets.yaml")
    cfg = all_configs[dataset_name]

    from src.utils.load_fs_config import load_fs_config
    fs_cfg = load_fs_config()

    xai_cfg              = fs_cfg.get("xai", {})
    n_features_to_select = fs_cfg["common"]["n_features_to_select"]

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

    # Pre-train a single XGBoost for SHAP and LIME explanation
    # This is the "explainer model" — separate from the final evaluation models
    print("Pre-training explainer model (XGBoost) for SHAP and LIME...\n")
    explainer_model = deepcopy(list(MODELS.values())[1])  # deep copy — isolated from MODELS
    explainer_model.fit(X_train, y_train)

    # ── SHAP ──────────────────────────────────────────────────────
    shap_cfg = xai_cfg.get("shap", {})
    if shap_cfg.get("enabled", False):
        for k in n_features_to_select:
            print(f"\n{'─'*60}")
            print(f"Feature Selection: SHAP (k={k})")
            print(f"{'─'*60}")

            fs_start = time.time()
            X_tr, X_v, X_te, selected_features, feature_importance = apply_shap(
                X_train, X_val, X_test,
                model=explainer_model,
                cfg=shap_cfg,
                k=k
            )
            fs_time = time.time() - fs_start

            print(f"Selected features: {k} / {X_train.shape[1]}")
            print(f"Feature selection time: {fs_time:.2f}s")
            print(f"Top features: {selected_features}\n")

            # Save importance scores and selected features
            selector_dir = results_dir / "selectors" / "xai"
            selector_dir.mkdir(parents=True, exist_ok=True)
            with open(selector_dir / f"shap_k{k}_features.json", "w") as f:
                json.dump(selected_features, f)
            feature_importance.to_csv(
                selector_dir / f"shap_k{k}_importance.csv", header=True
            )

            train_and_evaluate(
                MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                "shap", k, selected_features,
                results_dir, fs_time, results
            )

    # ── LIME ──────────────────────────────────────────────────────
    lime_cfg = xai_cfg.get("lime", {})
    if lime_cfg.get("enabled", False):
        for k in n_features_to_select:
            print(f"\n{'─'*60}")
            print(f"Feature Selection: LIME (k={k})")
            print(f"{'─'*60}")

            fs_start = time.time()
            X_tr, X_v, X_te, selected_features, feature_importance = apply_lime(
                X_train, X_val, X_test, y_train,
                model=explainer_model,
                cfg=lime_cfg,
                k=k
            )
            fs_time = time.time() - fs_start

            print(f"Selected features: {k} / {X_train.shape[1]}")
            print(f"Feature selection time: {fs_time:.2f}s")
            print(f"Top features: {selected_features}\n")

            # Save importance scores and selected features
            selector_dir = results_dir / "selectors" / "xai"
            selector_dir.mkdir(parents=True, exist_ok=True)
            with open(selector_dir / f"lime_k{k}_features.json", "w") as f:
                json.dump(selected_features, f)
            feature_importance.to_csv(
                selector_dir / f"lime_k{k}_importance.csv", header=True
            )

            train_and_evaluate(
                MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                "lime", k, selected_features,
                results_dir, fs_time, results
            )

    # Save results
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / "xai.csv", index=False)
    print(f"\nSaved results to {tables_dir / 'xai.csv'}")
    print(f"Saved models to {results_dir / 'models' / 'xai'}\n")

    return results_df