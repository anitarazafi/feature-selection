# experiments/run_embedded_fs.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
import json
from pathlib import Path

from src.utils.paths import SPLITS_DIR, MODELS_DIR, FEATURES_DIR, COMPARISONS_DIR
from src.utils.data_io import load_splits

# Datasets and models
DATASETS = ["ibtracs.last3years"]
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
}

# Regularization strengths to try (smaller C = stronger regularization)
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

def l1_feature_selection(X_train, X_test, y_train, C):
    """
    L1 (LASSO) regularization for feature selection.
    Features with zero coefficients are removed.
    """
    # Train L1 logistic regression (NEW syntax)
    l1_model = LogisticRegression(
        l1_ratio=1.0,  # 1.0 = pure L1 (LASSO)
        solver='saga',  # saga supports l1_ratio
        C=C,
        max_iter=5000,
        random_state=42
    )
    l1_model.fit(X_train, y_train)
    
    # Get feature coefficients
    coefficients = l1_model.coef_[0]
    
    # Select features with non-zero coefficients
    selected_indices = np.where(coefficients != 0)[0]
    selected_features = X_train.columns[selected_indices].tolist()
    
    # If no features selected (C too small), select top features by absolute coefficient
    if len(selected_features) == 0:
        top_n = min(10, len(coefficients))
        selected_indices = np.argsort(np.abs(coefficients))[-top_n:]
        selected_features = X_train.columns[selected_indices].tolist()
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_test_selected, selected_features, coefficients


def l2_feature_selection(X_train, X_test, y_train, C, top_k=30):
    """
    L2 (Ridge) regularization for feature ranking.
    Select top_k features with highest absolute coefficients.
    """
    # Train L2 logistic regression (NEW syntax)
    l2_model = LogisticRegression(
        l1_ratio=0.0,  # 0.0 = pure L2 (Ridge)
        solver='saga',  # saga supports l1_ratio
        C=C,
        max_iter=5000,
        random_state=42
    )
    l2_model.fit(X_train, y_train)
    
    # Get feature coefficients
    coefficients = l2_model.coef_[0]
    
    # Select top_k features by absolute coefficient value
    top_indices = np.argsort(np.abs(coefficients))[-top_k:]
    selected_features = X_train.columns[top_indices].tolist()
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_test_selected, selected_features, coefficients

def train_with_selected_features(X_train, X_test, y_train, y_test,
                                  method_name, C_value, dataset_name, selected_features):
    """Train all models with selected features and return results."""
    
    results = []
    n_features = len(selected_features)
    
    for model_name, model in MODELS.items():
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Save model
        model_dir = MODELS_DIR / "embedded_fs" / dataset_name / method_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        import pickle
        model_filename = f"{model_name}_C{C_value}.pkl"
        with open(model_dir / model_filename, "wb") as f:
            pickle.dump(model, f)
        
        # Store results
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "method": method_name,
            "C": C_value,
            "n_features": n_features,
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "train_time": train_time
        })
    
    return results


def run_embedded_fs(dataset_name):
    """Run embedded feature selection methods (L1 and L2)."""
    
    print(f"\n{'='*60}")
    print(f"Embedded Feature Selection: {dataset_name}")
    print(f"{'='*60}\n")
    
    # Load data
    splits = load_splits(dataset_name)
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train)}\n")
    
    all_results = []
    
    # ==================== L1 (LASSO) ====================
    print(f"\n{'='*60}")
    print("L1 REGULARIZATION (LASSO)")
    print(f"{'='*60}\n")
    
    for C in C_VALUES:
        print(f"\n--- C = {C} ---")
        
        # Select features using L1
        X_train_selected, X_test_selected, selected_features, coefficients = l1_feature_selection(
            X_train, X_test, y_train, C
        )
        
        n_features = len(selected_features)
        print(f"Selected {n_features} features with non-zero coefficients")
        
        # Save selected features and coefficients
        features_dir = FEATURES_DIR / dataset_name / "l1_lasso"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        feature_importance = {
            feat: float(coef) 
            for feat, coef in zip(X_train.columns, coefficients)
            if coef != 0
        }
        
        with open(features_dir / f"selected_C{C}.json", "w") as f:
            json.dump({
                "C": C,
                "n_features": n_features,
                "selected_features": selected_features,
                "coefficients": feature_importance
            }, f, indent=2)
        
        # Train models and collect results
        results = train_with_selected_features(
            X_train_selected, X_test_selected, y_train, y_test,
            "l1_lasso", C, dataset_name, selected_features
        )
        
        all_results.extend(results)
        
        # Print summary
        for r in results:
            print(f"  {r['model']:20s} - F1: {r['f1_score']:.4f}, AUC: {r['auc']:.4f}")
    
    # ==================== L2 (RIDGE) ====================
    print(f"\n{'='*60}")
    print("L2 REGULARIZATION (RIDGE)")
    print(f"{'='*60}\n")
    
    # For L2, we'll select different numbers of top features
    TOP_K_VALUES = [10, 20, 30, 40, 50]
    
    for top_k in TOP_K_VALUES:
        if top_k >= X_train.shape[1]:
            continue
        
        print(f"\n--- Top {top_k} features (C=1.0) ---")
        
        # Select features using L2
        X_train_selected, X_test_selected, selected_features, coefficients = l2_feature_selection(
            X_train, X_test, y_train, C=1.0, top_k=top_k
        )
        
        n_features = len(selected_features)
        print(f"Selected {n_features} features")
        
        # Save selected features and coefficients
        features_dir = FEATURES_DIR / dataset_name / "l2_ridge"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        feature_importance = {
            feat: float(coef)
            for feat, coef in zip(X_train.columns, coefficients)
        }
        # Sort by absolute value
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k])
        
        with open(features_dir / f"selected_top{top_k}.json", "w") as f:
            json.dump({
                "top_k": top_k,
                "n_features": n_features,
                "selected_features": selected_features,
                "coefficients": feature_importance
            }, f, indent=2)
        
        # Train models and collect results
        results = train_with_selected_features(
            X_train_selected, X_test_selected, y_train, y_test,
            "l2_ridge", f"top{top_k}", dataset_name, selected_features
        )
        
        all_results.extend(results)
        
        # Print summary
        for r in results:
            print(f"  {r['model']:20s} - F1: {r['f1_score']:.4f}, AUC: {r['auc']:.4f}")
    
    # Save all results
    results_dir = COMPARISONS_DIR / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / f"embedded_fs_{dataset_name}.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Embedded FS complete for {dataset_name}")
    print(f"✓ Saved results to {results_dir / f'embedded_fs_{dataset_name}.csv'}")
    print(f"{'='*60}\n")
    
    return results_df


if __name__ == "__main__":
    all_results = []
    
    for ds in DATASETS:
        results = run_embedded_fs(ds)
        all_results.append(results)
    
    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(COMPARISONS_DIR / "tables" / "embedded_fs_all.csv", index=False)
    
    print("\n" + "="*60)
    print("ALL EMBEDDED FEATURE SELECTION COMPLETE")
    print("="*60)
    print(f"\nL1 Results by C value:")
    print(combined[combined['method'] == 'l1_lasso'].groupby('C')[['n_features', 'f1_score']].mean().to_string())
    print(f"\nL2 Results by top_k:")
    print(combined[combined['method'] == 'l2_ridge'].groupby('C')[['n_features', 'f1_score']].mean().to_string())