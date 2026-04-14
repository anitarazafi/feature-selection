import yaml
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from src.utils.paths import CONFIG_DIR


MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "mlp": MLPClassifier
}

SEED_PARAM = {
    "random_forest": "random_state",
    "xgboost": "random_state",
    "mlp": "random_state",
}
 
def load_models(seed=42):
    """
    Load models based on config, with a configurable random seed.
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
 
        # Inject seed
        seed_key = SEED_PARAM.get(model_name)
        if seed_key:
            params[seed_key] = seed
 
        models[model_name] = model_class(**params)
    
    return models