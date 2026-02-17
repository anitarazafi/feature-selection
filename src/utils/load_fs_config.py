import yaml
from src.utils.paths import CONFIG_DIR

def load_fs_config():
    """
    Load feature selection configuration.
    """
    config_path = CONFIG_DIR / "fs.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)