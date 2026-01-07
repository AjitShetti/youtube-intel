import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            # Try looking in parent directory if not found (e.g. running from src/)
            parent_path = self.config_path.parent.parent / self.config_path.name
            if parent_path.exists():
                self.config_path = parent_path
            else:
                raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Inject env vars
        if "youtube" in config:
            config["youtube"]["api_key"] = os.getenv("YOUTUBE_API_KEY", config["youtube"].get("api_key"))

        return config

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

# Global config instance
try:
    # Assume config.yaml is in the project root
    PROJECT_ROOT = Path(__file__).parent.parent
    config_instance = Config(PROJECT_ROOT / "config.yaml")
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    config_instance = None

def get_config(key, default=None):
    if config_instance:
        return config_instance.get(key, default)
    return default
