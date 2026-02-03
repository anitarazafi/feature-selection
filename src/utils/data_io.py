import pickle
import pandas as pd
from src.utils.paths import PREPROCESSED_DATA_DIR

def load_splits(dataset_name):
    """
    Load saved splits, encoder, and scaler.
    """
    splits_dir = PREPROCESSED_DATA_DIR / dataset_name / "splits"
    splits = {
        "X_train": pd.read_csv(splits_dir / "X_train.csv"),
        "X_test": pd.read_csv(splits_dir / "X_test.csv"),
        "y_train": pd.read_csv(splits_dir / "y_train.csv").squeeze(),
        "y_test": pd.read_csv(splits_dir / "y_test.csv").squeeze()
    }
    
    # Load encoder if exists
    encoder_path = splits_dir / "encoder.pkl"
    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            splits["encoder"] = pickle.load(f)
    
    # Load scaler if exists
    scaler_path = splits_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            splits["scaler"] = pickle.load(f)
    
    return splits