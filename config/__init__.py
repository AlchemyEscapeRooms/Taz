"""Configuration management for AI Trading Bot."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager with environment variable support."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Replace environment variables
        self._replace_env_vars(self._config)

    def _replace_env_vars(self, config: Any) -> None:
        """Recursively replace ${VAR} with environment variables."""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, "")
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(config, list):
            for item in config:
                self._replace_env_vars(item)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)

    @property
    def raw(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        return self._config


# Global configuration instance
config = Config()
