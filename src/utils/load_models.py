import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils.paths import CONFIG_DIR


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier
}

DEFAULT_PARAMS = {
    "logistic_regression": {"max_iter": 1000, "random_state": 42},
    "random_forest": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    "xgboost": {"random_state": 42, "n_jobs": -1, "eval_metric": "logloss"}
}

def load_models():
    """
    Load models based on config.
    """
    config_path = CONFIG_DIR / "models.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    active = config.get('models', list(MODEL_REGISTRY.keys()))
    hyperparams = config.get('hyperparameters', {})
    
    models = {}
    for model_name in active:
        if model_name not in MODEL_REGISTRY:
            print(f"Warning: Unknown model '{model_name}', skipping")
            continue
        
        # Get class
        model_class = MODEL_REGISTRY[model_name]
        
        # Merge default params with config overrides
        params = DEFAULT_PARAMS.get(model_name, {}).copy()
        params.update(hyperparams.get(model_name, {}))
        
        # Instantiate
        models[model_name] = model_class(**params)
    
    return models