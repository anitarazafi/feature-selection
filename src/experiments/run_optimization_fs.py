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
    """
    Evaluate a feature subset using AUC on validation set.
    Returns negative AUC (since optimizers minimize).
    """
    if len(selected_cols) == 0:
        return 0.0  # penalize empty subsets

    model.fit(X_train[selected_cols], y_train)
    y_proba = model.predict_proba(X_val[selected_cols])[:, 1]
    return roc_auc_score(y_val, y_proba)


# ── PSO ───────────────────────────────────────────────────────────────────────

def pso_objective(position, X_train, X_val, y_train, y_val, feature_names, model, target_k):
    """
    Objective function for PSO.
    position: (n_particles, n_features) binary-like matrix
    Returns cost (1 - AUC) for each particle.
    """
    n_particles = position.shape[0]
    costs = np.zeros(n_particles)

    for i in range(n_particles):
        # Convert continuous position to binary by selecting top-k features
        top_k_idx    = np.argsort(position[i])[-target_k:]
        selected_cols = [feature_names[j] for j in top_k_idx]
        auc = evaluate_feature_subset(
            X_train, X_val, y_train, y_val, selected_cols, model
        )
        costs[i] = 1 - auc  # minimize cost = maximize AUC

    return costs


def apply_pso(X_train, X_val, X_test, y_train, y_val, cfg, target_k, model):
    n_particles  = cfg.get("n_particles", 30)
    iterations   = cfg.get("iterations", 20)
    w            = cfg.get("w", 0.7)
    c1           = cfg.get("c1", 1.5)
    c2           = cfg.get("c2", 1.5)
    n_features   = X_train.shape[1]
    feature_names = list(X_train.columns)

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

    # Get final selected features from best position
    top_k_idx     = np.argsort(best_pos)[-target_k:]
    selected_cols = [feature_names[j] for j in top_k_idx]
    best_auc      = 1 - best_cost

    print(f"  PSO best val AUC: {best_auc:.4f}")
    return X_train[selected_cols], X_val[selected_cols], X_test[selected_cols], selected_cols, optimizer


# ── DE ────────────────────────────────────────────────────────────────────────

def apply_de(X_train, X_val, X_test, y_train, y_val, cfg, target_k, model):
    population_size = cfg.get("population_size", 30)
    generations     = cfg.get("generations", 20)
    F               = cfg.get("F", 0.8)
    CR              = cfg.get("CR", 0.9)
    n_features      = X_train.shape[1]
    feature_names   = list(X_train.columns)

    # scipy popsize is a multiplier — convert to actual population size
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
        popsize=actual_popsize,  # now correctly 30//63 = 1 → 63 individuals
        mutation=F,
        recombination=CR,
        seed=42,
        tol=1e-4,
        polish=False,
        disp=True
    )

    selected_cols = best_result["cols"]
    best_auc      = best_result["auc"]

    print(f"  DE best val AUC: {best_auc:.4f}")
    return X_train[selected_cols], X_val[selected_cols], X_test[selected_cols], selected_cols, result


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
        pred_dir = results_dir / "predictions" / "optimization" / fs_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        with open(pred_dir / f"{model_name}_n{n_features}_predictions.json", "w") as f:
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
            }, f)

        # Save model
        model_dir = results_dir / "models" / "optimization" / fs_name
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

def run_optimization_fs(dataset_name):
    print(f"\n{'='*60}")
    print(f"= Running Optimization-Based Feature Selection for: {dataset_name}")
    print(f"{'='*60}\n")

    # Load configs
    with open(CONFIG_DIR / "datasets.yaml", "r") as f:
        all_configs = yaml.safe_load(f)
    if dataset_name not in all_configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets.yaml")
    cfg = all_configs[dataset_name]

    from src.utils.load_fs_config import load_fs_config
    fs_cfg = load_fs_config()

    opt_cfg              = fs_cfg.get("optimization", {})
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

    # Use a single estimator for optimization fitness evaluation
    # (separate from MODELS to avoid polluting trained models)
    eval_model = list(MODELS.values())[1]  # xgboost — fastest and most stable

    # ── PSO ───────────────────────────────────────────────────────
    pso_cfg = opt_cfg.get("pso", {})
    if pso_cfg.get("enabled", False):
        for k in n_features_to_select:
            print(f"\n{'─'*60}")
            print(f"Feature Selection: PSO (k={k})")
            print(f"{'─'*60}")

            fs_start = time.time()
            X_tr, X_v, X_te, selected_features, optimizer = apply_pso(
                X_train, X_val, X_test, y_train, y_val,
                cfg=pso_cfg, target_k=k, model=eval_model
            )
            fs_time = time.time() - fs_start

            print(f"Selected features: {k} / {X_train.shape[1]}")
            print(f"Feature selection time: {fs_time:.2f}s")
            print(f"Features: {selected_features}\n")

            # Save selector
            selector_dir = results_dir / "selectors" / "optimization"
            selector_dir.mkdir(parents=True, exist_ok=True)
            with open(selector_dir / f"pso_k{k}_features.json", "w") as f:
                json.dump(selected_features, f)
            # with open(selector_dir / f"pso_k{k}_optimizer.pkl", "wb") as f:
            #     pickle.dump(optimizer, f)

            train_and_evaluate(
                MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                "pso", k, selected_features,
                results_dir, fs_time, results
            )

    # ── DE ────────────────────────────────────────────────────────
    de_cfg = opt_cfg.get("differential_evolution", {})
    if de_cfg.get("enabled", False):
        for k in n_features_to_select:
            print(f"\n{'─'*60}")
            print(f"Feature Selection: DIFFERENTIAL EVOLUTION (k={k})")
            print(f"{'─'*60}")

            fs_start = time.time()
            X_tr, X_v, X_te, selected_features, result = apply_de(
                X_train, X_val, X_test, y_train, y_val,
                cfg=de_cfg, target_k=k, model=eval_model
            )
            fs_time = time.time() - fs_start

            print(f"Selected features: {k} / {X_train.shape[1]}")
            print(f"Feature selection time: {fs_time:.2f}s")
            print(f"Features: {selected_features}\n")

            # Save selector
            selector_dir = results_dir / "selectors" / "optimization"
            selector_dir.mkdir(parents=True, exist_ok=True)
            with open(selector_dir / f"de_k{k}_features.json", "w") as f:
                json.dump(selected_features, f)
            with open(selector_dir / f"de_k{k}_result.pkl", "wb") as f:
                pickle.dump(result, f)

            train_and_evaluate(
                MODELS, X_tr, X_v, X_te, y_train, y_val, y_test,
                "differential_evolution", k, selected_features,
                results_dir, fs_time, results
            )

    # Save results
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / "optimization.csv", index=False)
    print(f"\nSaved results to {tables_dir / 'optimization.csv'}")
    print(f"Saved models to {results_dir / 'models' / 'optimization'}\n")

    return results_df