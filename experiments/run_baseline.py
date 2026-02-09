import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import time
import json
from pathlib import Path
from src.utils.paths import MODELS_DIR, COMPARISONS_DIR
from src.utils.data_io import load_splits

# Datasets to process
DATASETS = ["ibtracs.last3years"]

# Models to train
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
}

def train_baseline(dataset_name):
    """
    Train baseline models (no feature selection).
    """
    print(f"\n{'='*50}")
    print(f"Training baseline models for: {dataset_name}")
    print(f"{'='*50}\n")
    # Load preprocessed splits
    splits = load_splits(dataset_name)
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}\n")
    results = []
    # Train each model
    for model_name, model in MODELS.items():
        print(f"Training {model_name}...")
        # Train and time it
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - AUC: {auc:.4f}")
        print(f"  - Training time: {train_time:.2f}s\n")
        # Save model
        model_dir = MODELS_DIR / "baseline" / dataset_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / f"{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
        # Store results
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "method": "baseline",
            "n_features": X_train.shape[1],
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "train_time": train_time
        })
    
    # Save results
    results_dir = COMPARISONS_DIR / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / f"baseline_{dataset_name}.csv", index=False)
    print(f"\nSaved results to {results_dir / f'baseline_{dataset_name}.csv'}")
    print(f"Saved models to {MODELS_DIR / 'baseline' / dataset_name}\n")
    
    return results_df

if __name__ == "__main__":
    all_results = []
    for ds in DATASETS:
        results = train_baseline(ds)
        all_results.append(results)
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(COMPARISONS_DIR / "tables" / "baseline_all.csv", index=False)
    print("\n" + "="*50)
    print("BASELINE TRAINING COMPLETE")
    print("="*50)
    print(combined.to_string(index=False))