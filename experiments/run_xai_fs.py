import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.utils.paths import SPLITS_DIR, MODELS_DIR, FEATURES_DIR, COMPARISONS_DIR
from src.utils.data_io import load_splits

# Datasets and models
DATASETS = ["ibtracs.last3years"]
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
}

# Number of top features to select
TOP_K_VALUES = [10, 20, 30, 40, 50]


def shap_feature_selection(X_train, X_test, y_train, top_k):
    """
    Use SHAP values to select top features.
    SHAP provides game-theory based feature importance.
    """
    import shap
    
    # Train a model for SHAP analysis (use tree-based for speed)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Calculate SHAP values
    # Use TreeExplainer for tree-based models (faster)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # For binary classification, shap_values might be a list or 2D array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Get values for positive class
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]  # Get positive class if 3D
    
    # Calculate mean absolute SHAP value for each feature
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Ensure it's 1D
    if len(feature_importance.shape) > 1:
        feature_importance = feature_importance.flatten()
    
    # Create feature importance dict
    feature_importance_dict = {}
    for i, col in enumerate(X_train.columns):
        feature_importance_dict[col] = float(feature_importance[i])
    
    # Select top_k features
    top_features = sorted(
        feature_importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    selected_features = [f[0] for f in top_features]
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_test_selected, selected_features, feature_importance_dict


def lime_feature_selection(X_train, X_test, y_train, top_k, n_samples=500):
    """
    Use LIME (Local Interpretable Model-agnostic Explanations) for feature selection.
    Explains predictions locally and aggregates feature importance.
    """
    from lime.lime_tabular import LimeTabularExplainer
    
    # Train a model for LIME analysis
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['No Landfall', 'Landfall'],
        mode='classification',
        random_state=42
    )
    
    # Sample instances to explain (for speed, use subset)
    sample_size = min(n_samples, len(X_train))
    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
    
    # Aggregate feature importance across explanations
    feature_importance_sum = np.zeros(X_train.shape[1])
    
    print(f"  Generating LIME explanations for {sample_size} samples...")
    
    for i, idx in enumerate(sample_indices):
        if i % 100 == 0:
            print(f"    Progress: {i}/{sample_size}")
        
        # Get explanation for this instance
        exp = explainer.explain_instance(
            X_train.values[idx],
            model.predict_proba,
            num_features=X_train.shape[1]
        )
        
        # Extract feature importance from explanation
        for feature_idx, importance in exp.as_list():
            # feature_idx is feature name, extract actual index
            feature_name = feature_idx.split('<=')[0].split('>')[0].strip()
            if feature_name in X_train.columns:
                col_idx = X_train.columns.get_loc(feature_name)
                feature_importance_sum[col_idx] += abs(importance)
    
    # Average importance across all explanations
    feature_importance = feature_importance_sum / sample_size
    
    # Create feature importance dict
    feature_importance_dict = dict(zip(X_train.columns, feature_importance))
    
    # Select top_k features
    top_features = sorted(
        feature_importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    selected_features = [f[0] for f in top_features]
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_test_selected, selected_features, feature_importance_dict


def train_with_selected_features(X_train, X_test, y_train, y_test,
                                  method_name, top_k, dataset_name, selected_features):
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
        model_dir = MODELS_DIR / "xai_fs" / dataset_name / method_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        import pickle
        model_filename = f"{model_name}_top{top_k}.pkl"
        with open(model_dir / model_filename, "wb") as f:
            pickle.dump(model, f)
        
        # Store results
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "method": method_name,
            "n_features": n_features,
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "train_time": train_time
        })
    
    return results


def run_xai_fs(dataset_name):
    """Run XAI-based feature selection methods."""
    
    print(f"\n{'='*60}")
    print(f"XAI-Based Feature Selection: {dataset_name}")
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
    
    # ==================== SHAP ====================
    print(f"\n{'='*60}")
    print("SHAP VALUES")
    print(f"{'='*60}\n")
    
    for top_k in TOP_K_VALUES:
        if top_k >= X_train.shape[1]:
            continue
        
        print(f"\n--- Top {top_k} features ---")
        
        # Select features using SHAP
        X_train_selected, X_test_selected, selected_features, importance = shap_feature_selection(
            X_train, X_test, y_train, top_k
        )
        
        print(f"Selected {len(selected_features)} features")
        
        # Save selected features and importance
        features_dir = FEATURES_DIR / dataset_name / "shap"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Get importance for selected features only
        selected_importance = {f: float(importance[f]) for f in selected_features}
        
        with open(features_dir / f"selected_top{top_k}.json", "w") as f:
            json.dump({
                "top_k": top_k,
                "n_features": len(selected_features),
                "selected_features": selected_features,
                "shap_importance": selected_importance
            }, f, indent=2)
        
        # Train models and collect results
        results = train_with_selected_features(
            X_train_selected, X_test_selected, y_train, y_test,
            "shap", top_k, dataset_name, selected_features
        )
        
        all_results.extend(results)
        
        # Print summary
        for r in results:
            print(f"  {r['model']:20s} - F1: {r['f1_score']:.4f}, AUC: {r['auc']:.4f}")
    
    # ==================== LIME ====================
    print(f"\n{'='*60}")
    print("LIME (Local Interpretable Model-agnostic Explanations)")
    print(f"{'='*60}\n")
    
    for top_k in TOP_K_VALUES:
        if top_k >= X_train.shape[1]:
            continue
        
        print(f"\n--- Top {top_k} features ---")
        
        # Select features using LIME
        X_train_selected, X_test_selected, selected_features, importance = lime_feature_selection(
            X_train, X_test, y_train, top_k
        )
        
        print(f"Selected {len(selected_features)} features")
        
        # Save selected features and importance
        features_dir = FEATURES_DIR / dataset_name / "lime"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Get importance for selected features only
        selected_importance = {f: float(importance[f]) for f in selected_features}
        
        with open(features_dir / f"selected_top{top_k}.json", "w") as f:
            json.dump({
                "top_k": top_k,
                "n_features": len(selected_features),
                "selected_features": selected_features,
                "lime_importance": selected_importance
            }, f, indent=2)
        
        # Train models and collect results
        results = train_with_selected_features(
            X_train_selected, X_test_selected, y_train, y_test,
            "lime", top_k, dataset_name, selected_features
        )
        
        all_results.extend(results)
        
        # Print summary
        for r in results:
            print(f"  {r['model']:20s} - F1: {r['f1_score']:.4f}, AUC: {r['auc']:.4f}")
    
    # Save all results
    results_dir = COMPARISONS_DIR / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / f"xai_fs_{dataset_name}.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ XAI FS complete for {dataset_name}")
    print(f"✓ Saved results to {results_dir / f'xai_fs_{dataset_name}.csv'}")
    print(f"{'='*60}\n")
    
    return results_df


if __name__ == "__main__":
    all_results = []
    
    for ds in DATASETS:
        results = run_xai_fs(ds)
        all_results.append(results)
    
    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(COMPARISONS_DIR / "tables" / "xai_fs_all.csv", index=False)
    
    print("\n" + "="*60)
    print("ALL XAI FEATURE SELECTION COMPLETE")
    print("="*60)
    print(f"\nSHAP Results:")
    print(combined[combined['method'] == 'shap'].groupby('n_features')['f1_score'].mean().to_string())
    print(f"\nLIME Results:")
    print(combined[combined['method'] == 'lime'].groupby('n_features')['f1_score'].mean().to_string())