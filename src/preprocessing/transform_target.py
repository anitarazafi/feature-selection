import pandas as pd

def transform_target(y_raw, cfg, dataset_name):
    """
    Transform target variable based on dataset-specific requirements.
    Parameters:
    - y_raw: Raw target series
    - cfg: Dataset configuration
    - dataset_name: Name of the dataset
    Returns:
    - y: Transformed target series
    """
    transform_type = cfg.get("target_transform")
    
    if transform_type == "binary_landfall":
        # Convert to numeric and create binary target
        y_raw = pd.to_numeric(y_raw, errors="coerce")
        y = (y_raw == 0).astype(int)
        y.name = f"{cfg['target']}_bin"
    else:
        # No transformation
        y = y_raw
    
    return y