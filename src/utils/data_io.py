import pickle
import pandas as pd

def load_splits(splits_dir):
    """
    Load saved splits, encoder, and scaler.
    """
    splits = {
        "X_train": pd.read_csv(splits_dir / "X_train.csv"),
        "X_val":   pd.read_csv(splits_dir / "X_val.csv"),
        "X_test":  pd.read_csv(splits_dir / "X_test.csv"),
        "y_train": pd.read_csv(splits_dir / "y_train.csv").squeeze(),
        "y_val":   pd.read_csv(splits_dir / "y_val.csv").squeeze(),
        "y_test":  pd.read_csv(splits_dir / "y_test.csv").squeeze(),
    }

    encoder_path = splits_dir / "encoder.pkl"
    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            splits["encoder"] = pickle.load(f)

    scaler_path = splits_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            splits["scaler"] = pickle.load(f)

    return splits