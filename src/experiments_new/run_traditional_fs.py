import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import yaml
import warnings
from src.utils.data_io import load_splits
from src.utils.load_models import load_models
from src.utils.paths import BASE_DIR, CONFIG_DIR

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def apply_mutual_information(X_train, X_val, X_test, y_train, k, seed=42):
    def mi_scorer(X, y):
        return mutual_info_classif(X, y, random_state=seed)
    selector = SelectKBest(mi_scorer, k=k)
    selector.fit(X_train, y_train)
    selected_cols = X_train.columns[selector.get_support()]
    scores = selector.scores_
    return selected_cols.tolist(), scores


def apply_rfe(X_train, X_val, X_test, y_train, estimator, k, cfg):
    step = cfg.get("step", 5)
    selector = RFE(estimator=estimator, n_features_to_select=k, step=step)
    selector.fit(X_train, y_train)
    selected_cols = X_train.columns[selector.get_support()]
    scores = 1.0 / selector.ranking_
    return selected_cols.tolist(), scores, selector


def get_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_proba),
    }


def normalize_scores(scores):
    min_s = scores.min()
    max_s = scores.max()
    if max_s == min_s:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)


def aggregate_features(selected_sets, importance_scores, feature_names, k):
    aggregated = {}

    # Union
    union_set = sorted(set.union(*[set(s) for s in selected_sets.values()]))
    aggregated["union"] = union_set

    # Intersection
    intersection_set = sorted(set.intersection(*[set(s) for s in selected_sets.values()]))
    aggregated["intersection"] = intersection_set

    # Weighted
    normalized = {}
    for method, scores in importance_scores.items():
        normalized[method] = normalize_scores(scores)

    combined_scores = np.mean(
        np.array([normalized[m] for m in importance_scores.keys()]),
        axis=0
    )
    combined_importance = pd.Series(combined_scores, index=feature_names)
    weighted_set = combined_importance.nlargest(k).index.tolist()
    aggregated["weighted"] = sorted(weighted_set)

    return aggregated, combined_importance


def train_and_evaluate(MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                       fs_name, n_features, selected_features,
                       results_dir, fs_time, results, seed=42):
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
            "seed": seed,
        }
        with open(pred_dir / f"{model_name}_n{n_features}_seed{seed}_predictions.json", "w") as f:
            json.dump(predictions, f)

        model_dir = results_dir / "models" / "traditional" / fs_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / f"{model_name}_n{n_features}_seed{seed}.pkl", "wb") as f:
            pickle.dump(model, f)

        results.append({
            "model":         model_name,
            "fs_method":     fs_name,
            "n_features":    n_features,
            "seed":          seed,
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


def run_traditional_fs(dataset_name, seed=42, k=None):
    print(f"\n{'='*60}")
    print(f"= Running Traditional Feature Selection for: {dataset_name} (seed={seed})")
    print(f"{'='*60}\n")

    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    if dataset_name not in all_configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets.yaml")
    cfg = all_configs[dataset_name]

    from src.utils.load_fs_config import load_fs_config
    fs_cfg = load_fs_config()

    traditional_cfg = fs_cfg.get("traditional", {})

    if k is None:
        k = fs_cfg["common"]["n_features_to_select"]

    splits_dir  = BASE_DIR / cfg["paths"]["splits"]
    results_dir = BASE_DIR / cfg["paths"]["results"]

    splits  = load_splits(splits_dir)
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]
    y_test  = splits["y_test"]

    feature_names = list(X_train.columns)

    print(f"Training samples:    {len(X_train)}")
    print(f"Validation samples:  {len(X_val)}")
    print(f"Test samples:        {len(X_test)}")
    print(f"Features (original): {X_train.shape[1]}")
    print(f"k (features to select): {k}\n")

    MODELS  = load_models(seed=seed)
    results = []

    selected_sets = {}
    importance_scores = {}
    fs_times = {}

    # ── 1. Mutual Information ─────────────────────────────────────
    mi_cfg = traditional_cfg.get("mutual_information", {})
    if mi_cfg.get("enabled", False):
        print(f"\n{'─'*60}")
        print(f"Feature Selection: MUTUAL INFORMATION (k={k})")
        print(f"{'─'*60}")

        fs_start = time.time()
        selected_features, scores = apply_mutual_information(
            X_train, X_val, X_test, y_train, k=k, seed=seed
        )
        fs_time = time.time() - fs_start
        selected_sets["mutual_information"] = selected_features
        importance_scores["mutual_information"] = scores
        fs_times["mutual_information"] = fs_time

        print(f"Selected features ({k}): {selected_features}")
        print(f"Feature selection time: {fs_time:.2f}s")

        selector_dir = results_dir / "selectors" / "traditional"
        selector_dir.mkdir(parents=True, exist_ok=True)
        with open(selector_dir / f"mutual_information_k{k}_seed{seed}_features.json", "w") as f:
            json.dump(selected_features, f)

    # ── 2. RFE ────────────────────────────────────────────────────
    rfe_cfg = traditional_cfg.get("rfe", {})
    if rfe_cfg.get("enabled", False):
        estimator_name = rfe_cfg.get("estimator", "random_forest")
        estimator      = MODELS[estimator_name]

        print(f"\n{'─'*60}")
        print(f"Feature Selection: RFE (k={k}, estimator={estimator_name})")
        print(f"{'─'*60}")

        fs_start = time.time()
        selected_features, scores, selector = apply_rfe(
            X_train, X_val, X_test, y_train,
            estimator=estimator, k=k, cfg=rfe_cfg
        )
        fs_time = time.time() - fs_start
        selected_sets["rfe"] = selected_features
        importance_scores["rfe"] = scores
        fs_times["rfe"] = fs_time

        print(f"Selected features ({k}): {selected_features}")
        print(f"Feature selection time: {fs_time:.2f}s")

        selector_dir = results_dir / "selectors" / "traditional"
        selector_dir.mkdir(parents=True, exist_ok=True)
        with open(selector_dir / f"rfe_k{k}_seed{seed}_selector.pkl", "wb") as f:
            pickle.dump(selector, f)
        with open(selector_dir / f"rfe_k{k}_seed{seed}_features.json", "w") as f:
            json.dump(selected_features, f)

    # ── 3. Aggregate and train ────────────────────────────────────
    if selected_sets:
        total_fs_time = sum(fs_times.values())

        aggregated, combined_importance = aggregate_features(
            selected_sets, importance_scores, feature_names, k
        )

        print(f"\n{'─'*60}")
        print(f"Aggregation Results")
        print(f"{'─'*60}")
        print(f"Individual method contributions:")
        for method, feats in selected_sets.items():
            print(f"  {method}: {sorted(feats)}")

        selector_dir = results_dir / "selectors" / "traditional"
        selector_dir.mkdir(parents=True, exist_ok=True)
        combined_importance.to_csv(
            selector_dir / f"combined_importance_k{k}_seed{seed}.csv", header=True
        )

        for agg_name, feature_set in aggregated.items():
            n_features = len(feature_set)
            fs_name = f"fs_set_{agg_name}"

            print(f"\n{'─'*60}")
            print(f"FS-Set ({agg_name}): {feature_set}")
            print(f"FS-Set ({agg_name}) size: {n_features}")
            print(f"{'─'*60}\n")

            if n_features == 0:
                print(f"  WARNING: {agg_name} produced empty set, skipping.")
                continue

            set_info = {
                "fs_set": feature_set,
                "n_features": n_features,
                "aggregation": agg_name,
                "contributing_methods": {m: sorted(f) for m, f in selected_sets.items()},
                "fs_times": {m: round(t, 2) for m, t in fs_times.items()},
                "seed": seed,
            }
            with open(selector_dir / f"fs_set_{agg_name}_seed{seed}.json", "w") as f:
                json.dump(set_info, f, indent=2)

            X_tr_fs = X_train[feature_set]
            X_v_fs  = X_val[feature_set]
            X_te_fs = X_test[feature_set]

            train_and_evaluate(
                MODELS, X_tr_fs, X_v_fs, X_te_fs, y_train, y_val, y_test,
                fs_name, n_features, feature_set,
                results_dir, total_fs_time, results, seed=seed
            )

    # ── Save results ──────────────────────────────────────────────
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / f"traditional_seed{seed}.csv", index=False)
    print(f"\nSaved results to {tables_dir / f'traditional_seed{seed}.csv'}")
    print(f"Saved models to {results_dir / 'models' / 'traditional'}")

    return results_df