import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import time
import json
from pathlib import Path
from src.utils.paths import MODELS_DIR, COMPARISONS_DIR
from src.utils.data_io import load_splits
from src.utils.load_models import load_models

def train_baseline(dataset_name):
    """
    Train baseline models (no feature selection).
    """
    print(f"\n{'='*60}")
    print(f"Training baseline models for: {dataset_name}")
    print(f"{'='*60}\n")
    
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

    MODELS = load_models()
    for model_name, model in MODELS.items():
        print(f"Training {model_name}...")
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"  - Training time: {train_time:.2f}s\n")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - AUC: {auc:.4f}")

        # Save predictions
        pred_dir = COMPARISONS_DIR / "predictions" / "baseline" / dataset_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        predictions = {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist(),
            "model": model_name,
            "n_features": X_train.shape[1]
        }
        with open(pred_dir / f"{model_name}_predictions.json", "w") as f:
            json.dump(predictions, f)

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
            "accuracy": round(accuracy, 4),      
            "precision": round(precision, 4),    
            "recall": round(recall, 4),          
            "f1_score": round(f1, 4),            
            "auc": round(auc, 4),                
            "train_time": round(train_time, 2)
        })
    
    # Save results
    results_dir = COMPARISONS_DIR / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / f"baseline_{dataset_name}.csv", index=False)
    print(f"\nSaved results to {results_dir / f'baseline_{dataset_name}.csv'}")
    print(f"Saved models to {MODELS_DIR / 'baseline' / dataset_name}\n")
    
    return results_df
