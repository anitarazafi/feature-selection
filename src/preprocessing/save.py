from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

def save_processed(X, y, dataset_name):
    out_dir = BASE_DIR / "data" / "processed" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X.to_csv(out_dir / "X.csv", index=False)
    y.to_csv(out_dir / "y.csv", index=False)


def save_data_summary(X, y, dataset_name):
    out_dir = BASE_DIR / "data" / "processed" / dataset_name
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


def save_splits(splits, dataset_name):
    out_dir = BASE_DIR / "data" / "processed" / dataset_name / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)