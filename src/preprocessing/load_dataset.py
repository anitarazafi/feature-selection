import yaml
import pandas as pd
from src.utils.paths import BASE_DIR, CONFIG_DIR

def load_dataset(dataset_name):
    """
    Loads raw dataset and its configuration.
    Returns: pandas.DataFrame, cfg
    """
    configs = CONFIG_DIR / "datasets.yaml"
    with open(configs, "r") as f:
        all_configs = yaml.safe_load(f)
    
    # Get specific dataset config
    if dataset_name not in all_configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in config. Available: {list(all_configs.keys())}")
    
    cfg = all_configs[dataset_name]
    # Load data
    data_path = BASE_DIR / cfg["paths"]["raw"]
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path, low_memory=False)
    
    return df, cfg