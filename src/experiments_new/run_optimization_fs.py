import pandas as pd
import pickle
import time
import json
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from scipy.optimize import differential_evolution
import pyswarms as ps

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


def evaluate_feature_subset(X_train, X_val, y_train, y_val, selected_cols, model):
    if len(selected_cols) == 0:
        return 0.0
    model.fit(X_train[selected_cols], y_train)
    y_proba = model.predict_proba(X_val[selected_cols])[:, 1]
    return roc_auc_score(y_val, y_proba)


def normalize_scores(scores):
    min_s = scores.min()
    max_s = scores.max()
    if max_s == min_s:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)


def aggregate_features(selected_sets, importance_scores, feature_names, k):
    aggregated = {}

    union_set = sorted(set.union(*[set(s) for s in selected_sets.values()]))
    aggregated["union"] = union_set

    intersection_set = sorted(set.intersection(*[set(s) for s in selected_sets.values()]))
    aggregated["intersection"] = intersection_set

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


# ── PSO ───────────────────────────────────────────────────────────────────────

def pso_objective(position, X_train, X_val, y_train, y_val, feature_names, model, target_k):
    n_particles = position.shape[0]
    costs = np.zeros(n_particles)
    for i in range(n_particles):
        top_k_idx    = np.argsort(position[i])[-target_k:]
        selected_cols = [feature_names[j] for j in top_k_idx]
        auc = evaluate_feature_subset(
            X_train, X_val, y_train, y_val, selected_cols, model
        )
        costs[i] = 1 - auc
    return costs


def apply_pso(X_train, X_val, X_test, y_train, y_val, cfg, target_k, model, seed=42):
    n_particles  = cfg.get("n_particles", 30)
    iterations   = cfg.get("iterations", 20)
    w            = cfg.get("w", 0.7)
    c1           = cfg.get("c1", 1.5)
    c2           = cfg.get("c2", 1.5)
    n_features   = X_train.shape[1]
    feature_names = list(X_train.columns)

    np.random.seed(seed)

    options = {"c1": c1, "c2": c2, "w": w}
    bounds  = (np.zeros(n_features), np.ones(n_features))

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_features,
        options=options,
        bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(
        pso_objective,
        iters=iterations,
        verbose=False,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_names=feature_names,
        model=model,
        target_k=target_k
    )

    top_k_idx     = np.argsort(best_pos)[-target_k:]
    selected_cols = [feature_names[j] for j in top_k_idx]
    best_auc      = 1 - best_cost

    scores = best_pos

    print(f"  PSO best val AUC: {best_auc:.4f}")
    return selected_cols, scores


# ── DE ────────────────────────────────────────────────────────────────────────

def apply_de(X_train, X_val, X_test, y_train, y_val, cfg, target_k, model, seed=42):
    population_size = cfg.get("population_size", 30)
    generations     = cfg.get("generations", 20)
    F               = cfg.get("F", 0.8)
    CR              = cfg.get("CR", 0.9)
    n_features      = X_train.shape[1]
    feature_names   = list(X_train.columns)

    actual_popsize = max(1, population_size // n_features)

    best_result = {"auc": 0, "cols": []}

    def de_objective(position):
        top_k_idx     = np.argsort(position)[-target_k:]
        selected_cols = [feature_names[j] for j in top_k_idx]
        auc = evaluate_feature_subset(
            X_train, X_val, y_train, y_val, selected_cols, model
        )
        if auc > best_result["auc"]:
            best_result["auc"]  = auc
            best_result["cols"] = selected_cols
        return 1 - auc

    bounds = [(0, 1)] * n_features

    result = differential_evolution(
        de_objective,
        bounds=bounds,
        maxiter=generations,
        popsize=actual_popsize,
        mutation=F,
        recombination=CR,
        seed=seed,
        tol=1e-4,
        polish=False,
        disp=True
    )

    selected_cols = best_result["cols"]
    best_auc      = best_result["auc"]

    scores = result.x

    print(f"  DE best val AUC: {best_auc:.4f}")
    return selected_cols, scores


# ── Train & Evaluate ──────────────────────────────────────────────────────────

def train_and_evaluate(MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                       fs_name, n_features, selected_features,
                       results_dir, fs_time, results, seed=42):
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

        print(f"    Training time: {train_time:.2f}s")
        print(f"    {'Metric':<12} {'Val':>8} {'Test':>8}")
        print(f"    {'-'*30}")
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            print(f"    {metric:<12} {val_metrics[metric]:>8.4f} {test_metrics[metric]:>8.4f}")
        print()

        pred_dir = results_dir / "predictions" / "optimization" / fs_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        with open(pred_dir / f"{model_name}_n{n_features}_seed{seed}_predictions.json", "w") as f:
            json.dump({
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
            }, f)

        model_dir = results_dir / "models" / "optimization" / fs_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / f"{model_name}_n{n_features}_seed{seed}.pkl", "wb") as f:
            pickle.dump(model, f)

        results.append({
            "model":          model_name,
            "fs_method":      fs_name,
            "n_features":     n_features,
            "seed":           seed,
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

def run_optimization_fs(dataset_name, seed=42, k=None):
    print(f"\n{'='*60}")
    print(f"= Running Optimization-Based FS for: {dataset_name} (seed={seed})")
    print(f"{'='*60}\n")

    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    if dataset_name not in all_configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets.yaml")
    cfg = all_configs[dataset_name]

    from src.utils.load_fs_config import load_fs_config
    fs_cfg = load_fs_config()

    opt_cfg = fs_cfg.get("optimization", {})

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

    eval_model = list(MODELS.values())[1]

    selected_sets = {}
    importance_scores = {}
    fs_times = {}

    # ── 1. PSO ────────────────────────────────────────────────────
    pso_cfg = opt_cfg.get("pso", {})
    if pso_cfg.get("enabled", False):
        print(f"\n{'─'*60}")
        print(f"Feature Selection: PSO (k={k})")
        print(f"{'─'*60}")

        fs_start = time.time()
        selected_features, scores = apply_pso(
            X_train, X_val, X_test, y_train, y_val,
            cfg=pso_cfg, target_k=k, model=eval_model, seed=seed
        )
        fs_time = time.time() - fs_start
        selected_sets["pso"] = selected_features
        importance_scores["pso"] = scores
        fs_times["pso"] = fs_time

        print(f"Selected features ({k}): {selected_features}")
        print(f"Feature selection time: {fs_time:.2f}s")

        selector_dir = results_dir / "selectors" / "optimization"
        selector_dir.mkdir(parents=True, exist_ok=True)
        with open(selector_dir / f"pso_k{k}_seed{seed}_features.json", "w") as f:
            json.dump(selected_features, f)

    # ── 2. Differential Evolution ─────────────────────────────────
    de_cfg = opt_cfg.get("differential_evolution", {})
    if de_cfg.get("enabled", False):
        print(f"\n{'─'*60}")
        print(f"Feature Selection: DIFFERENTIAL EVOLUTION (k={k})")
        print(f"{'─'*60}")

        fs_start = time.time()
        selected_features, scores = apply_de(
            X_train, X_val, X_test, y_train, y_val,
            cfg=de_cfg, target_k=k, model=eval_model, seed=seed
        )
        fs_time = time.time() - fs_start
        selected_sets["differential_evolution"] = selected_features
        importance_scores["differential_evolution"] = scores
        fs_times["differential_evolution"] = fs_time

        print(f"Selected features ({k}): {selected_features}")
        print(f"Feature selection time: {fs_time:.2f}s")

        selector_dir = results_dir / "selectors" / "optimization"
        selector_dir.mkdir(parents=True, exist_ok=True)
        with open(selector_dir / f"de_k{k}_seed{seed}_features.json", "w") as f:
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

        selector_dir = results_dir / "selectors" / "optimization"
        selector_dir.mkdir(parents=True, exist_ok=True)
        combined_importance.to_csv(
            selector_dir / f"combined_importance_k{k}_seed{seed}.csv", header=True
        )

        for agg_name, feature_set in aggregated.items():
            n_features = len(feature_set)
            fs_name = f"op_set_{agg_name}"

            print(f"\n{'─'*60}")
            print(f"Op-Set ({agg_name}): {feature_set}")
            print(f"Op-Set ({agg_name}) size: {n_features}")
            print(f"{'─'*60}\n")

            if n_features == 0:
                print(f"  WARNING: {agg_name} produced empty set, skipping.")
                continue

            set_info = {
                "op_set": feature_set,
                "n_features": n_features,
                "aggregation": agg_name,
                "contributing_methods": {m: sorted(f) for m, f in selected_sets.items()},
                "fs_times": {m: round(t, 2) for m, t in fs_times.items()},
                "seed": seed,
            }
            with open(selector_dir / f"op_set_{agg_name}_seed{seed}.json", "w") as f:
                json.dump(set_info, f, indent=2)

            X_tr_op = X_train[feature_set]
            X_v_op  = X_val[feature_set]
            X_te_op = X_test[feature_set]

            train_and_evaluate(
                MODELS, X_tr_op, X_v_op, X_te_op, y_train, y_val, y_test,
                fs_name, n_features, feature_set,
                results_dir, total_fs_time, results, seed=seed
            )

    # ── Save results ──────────────────────────────────────────────
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / f"optimization_seed{seed}.csv", index=False)
    print(f"\nSaved results to {tables_dir / f'optimization_seed{seed}.csv'}")
    print(f"Saved models to {results_dir / 'models' / 'optimization'}")

    return results_df