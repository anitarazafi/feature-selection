import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
import time
import json
from pathlib import Path
import pickle
from src.utils.paths import SPLITS_DIR, MODELS_DIR, FEATURES_DIR, COMPARISONS_DIR
from src.utils.data_io import load_splits
from src.utils.load_models import load_models

# Feature selection configurations
N_FEATURES_TO_SELECT = [10, 20, 30, 40, 50]  # Different feature counts to try


def correlation_based_selection(X_train, X_test, n_features):
    """
    Select features based on correlation with target and remove highly correlated features.
    Strategy:
    1. Remove features with correlation > 0.95 with each other
    2. Select top n_features based on absolute correlation with target
    """
    # Calculate correlation with all features
    corr_matrix = X_train.corr().abs()
    
    # Find highly correlated pairs (upper triangle)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Drop features with correlation > 0.95
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > 0.95)]
    
    X_train_reduced = X_train.drop(columns=to_drop, errors='ignore')
    X_test_reduced = X_test.drop(columns=to_drop, errors='ignore')
    
    # If still more features than needed, select top n based on variance
    if X_train_reduced.shape[1] > n_features:
        variances = X_train_reduced.var()
        top_features = variances.nlargest(n_features).index.tolist()
        X_train_reduced = X_train_reduced[top_features]
        X_test_reduced = X_test_reduced[top_features]
    
    selected_features = X_train_reduced.columns.tolist()
    
    return X_train_reduced, X_test_reduced, selected_features


def variance_threshold_selection(X_train, X_test, n_features):
    """
    Select features with highest variance.
    """
    # Calculate variance for all features
    variances = X_train.var()
    
    # Select top n_features with highest variance
    top_features = variances.nlargest(n_features).index.tolist()
    
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    return X_train_selected, X_test_selected, top_features


def chi_square_selection(X_train, X_test, y_train, n_features):
    """
    Select features using chi-square test.
    Note: Chi-square requires non-negative features.
    """
    # Make all features non-negative (min-max scaling to [0, 1])
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Apply chi-square selection
    selector = SelectKBest(chi2, k=min(n_features, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names
    selected_features = X_train_scaled.columns[selector.get_support()].tolist()
    
    # Convert back to DataFrame
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    return X_train_selected, X_test_selected, selected_features


def train_with_selected_features(X_train, X_test, y_train, y_test, method_name, n_features, dataset_name):
    """
    Train all models with selected features and return results.
    """
    results = []
    MODELS = load_models()
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
        model_dir = MODELS_DIR / "traditional_fs" / dataset_name / method_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{model_name}_k{n_features}.pkl"
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


def run_traditional_fs(dataset_name):
    """
    Run all traditional feature selection methods.
    """
    print(f"\n{'='*60}")
    print(f"Traditional Feature Selection: {dataset_name}")
    print(f"{'='*60}\n")
    
    # Load preprocessed splits
    splits = load_splits(dataset_name)
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Original features: {X_train.shape[1]}\n")
    
    all_results = []
    
    # Feature selection methods
    fs_methods = {
        "correlation": correlation_based_selection,
        "variance": variance_threshold_selection,
        "chi_square": chi_square_selection
    }
    
    for method_name, fs_func in fs_methods.items():
        print(f"\n--- {method_name.upper()} ---")
        
        for n_features in N_FEATURES_TO_SELECT:
            if n_features >= X_train.shape[1]:
                continue  # Skip if requesting more features than available
            
            print(f"\nSelecting top {n_features} features...")
            
            # Select features
            if method_name == "chi_square":
                X_train_selected, X_test_selected, selected_features = fs_func(
                    X_train, X_test, y_train, n_features
                )
            else:
                X_train_selected, X_test_selected, selected_features = fs_func(
                    X_train, X_test, n_features
                )
            
            print(f"Selected {len(selected_features)} features")
            
            # Save selected features
            features_dir = FEATURES_DIR / dataset_name / method_name
            features_dir.mkdir(parents=True, exist_ok=True)
            
            with open(features_dir / f"selected_k{n_features}.json", "w") as f:
                json.dump(selected_features, f, indent=2)
            
            # Train models and collect results
            results = train_with_selected_features(
                X_train_selected, X_test_selected, y_train, y_test,
                method_name, len(selected_features), dataset_name
            )
            
            all_results.extend(results)
            
            # Print summary for this configuration
            for r in results:
                print(f"  {r['model']:20s} - F1: {r['f1_score']:.4f}, AUC: {r['auc']:.4f}")
    
    # Save all results
    results_dir = COMPARISONS_DIR / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / f"traditional_fs_{dataset_name}.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"Traditional FS complete for {dataset_name}")
    print(f"Saved results to {results_dir / f'traditional_fs_{dataset_name}.csv'}")
    print(f"{'='*60}\n")
    
    return results_df