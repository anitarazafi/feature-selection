import pandas as pd
import pickle
from src.utils.paths import BASE_DIR

def save_processed(X, y, cfg):
    processed_rel = cfg["paths"]["processed"]
    out_dir = BASE_DIR / processed_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "X.csv", index=False)
    y.to_csv(out_dir / "y.csv", index=False)
    print(f"\n{'='*60}")
    print(f"=Saved processed to {out_dir}")


def save_data_summary(X, y, cfg):
    processed_rel = cfg["paths"]["processed"]
    out_dir = BASE_DIR / processed_rel
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
    print(f"\n{'='*60}")
    print(f"=Saved summary to {out_dir}")
    

def save_splits(splits, cfg, encoder=None, scaler=None):
    processed_rel = cfg["paths"]["splits"]
    out_dir = BASE_DIR / processed_rel
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
    
    print(f"\n{'='*60}")
    print(f"=Saved splits for to {out_dir}")
    