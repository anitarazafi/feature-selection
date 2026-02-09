import pandas as pd
from pathlib import Path
import pickle
from src.utils.paths import BASE_DIR, PREPROCESSED_DATA_DIR

def save_processed(X, y, dataset_name):
    out_dir = PREPROCESSED_DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "X.csv", index=False)
    y.to_csv(out_dir / "y.csv", index=False)
    print(f"Saved processed for {dataset_name} to {out_dir}")


def save_data_summary(X, y, dataset_name):
    out_dir = PREPROCESSED_DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "target_name": y.name,
        "target_distribution": y.value_counts(dropna=False).to_dict(),
        "missing_rate_per_feature": X.isna().mean().to_dict(),
        "dtypes": X.dtypes.astype(str).to_dict(),
    }

    pd.Series(summary).to_json(out_dir / "data_summary.json", indent=2)
    print(f"Saved summary for {dataset_name} to {out_dir}")
    

def save_splits(splits, dataset_name, encoder=None, scaler=None):
    out_dir = BASE_DIR / "data" / "splits" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save data splits as CSV
    for name, df in splits.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    # Save encoder if provided
    if encoder is not None:
        with open(out_dir / "encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)

    # Save scaler if provided
    if scaler is not None:
        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    
    print(f"Saved splits for {dataset_name} to {out_dir}")
    