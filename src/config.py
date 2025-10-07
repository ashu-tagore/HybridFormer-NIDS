"""
Configuration management for NIDS project.

This module provides utilities for loading and managing
experiment configurations from YAML files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for experiments.

    Loads configuration from YAML files and provides easy access
    to nested configuration values.

    Args:
        config_path: Path to YAML configuration file

    Example:
        >>> config = Config('configs/baseline.yaml')
        >>> print(config.get('training.batch_size'))
        64
        >>> print(config.model.hidden_dims)
        [256, 128, 64]
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self._load_config()
        self._validate_config()

        logger.info(f"Configuration loaded from {self.config_path}")

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        if self._config is None:
            self._config = {}

    def _validate_config(self):
        """Validate that required configuration fields exist."""
        required_sections = ['data', 'model', 'training']

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate data paths
        data_config = self._config.get('data', {})
        required_paths = [
            'train_features', 'train_labels',
            'val_features', 'val_labels',
            'test_features', 'test_labels'
        ]

        for path_key in required_paths:
            if path_key not in data_config:
                raise ValueError(f"Missing required data path: data.{path_key}")

        logger.info("Configuration validation passed")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'training.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get('training.batch_size')
            64
            >>> config.get('training.missing_key', 128)
            128
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to config sections.

        Example:
            >>> config.training.batch_size
            64
        """
        if name.startswith('_'):
            # Avoid infinite recursion for private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name in self._config:
            value = self._config[name]
            # Wrap dict values in DotDict for nested attribute access
            if isinstance(value, dict):
                return DotDict(value)
            return value

        raise AttributeError(f"Configuration has no section: {name}")

    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self._config.copy()

    def save(self, path: str):
        """
        Save configuration to YAML file.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {path}")


class DotDict:
    """
    Wrapper for dictionaries to allow dot notation access.

    Internal helper class for Config.
    """

    def __init__(self, data: Dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return DotDict(value)
            return value

        raise AttributeError(f"Configuration has no key: {name}")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict:
        return self._data.copy()