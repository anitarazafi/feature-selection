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
        
        model_class = MODEL_REGISTRY[model_name]
        params = hyperparams.get(model_name, {})
        models[model_name] = model_class(**params)
    
    return models