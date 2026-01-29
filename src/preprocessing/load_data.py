import yaml
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

def load_dataset(dataset_name):
    """
    Loads raw dataset and its configuration.

    Parameters
    ----------
    dataset_name : str

    Returns
    -------
    df : pandas.DataFrame
    cfg : dict
    """
    cfg_path = BASE_DIR / "configs" / "data" / f"{dataset_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = BASE_DIR / cfg["paths"]["raw"]
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)

    return df, cfg
