"""
NIDS Bachelor Project - Network Intrusion Detection System
Simplified structure for easier development and maintenance
"""

__version__ = "0.1.0"
__author__ = "NIDS Research Team"

# Import main components for easy access
from .dataset import NIDSDataset, get_data_loaders, compute_class_weights
from .baseline_model import BaselineFFN
from .trainer import Trainer
from .metrics import MetricsCalculator
from .config import Config
from .utils import (
    save_checkpoint,
    load_checkpoint,
    early_stopping_check,
    EarlyStopping,
    set_seed,
    get_device,
    print_model_summary
)

__all__ = [
    # Data
    'NIDSDataset',
    'get_data_loaders',
    'compute_class_weights',

    # Models
    'BaselineFFN',

    # Training
    'Trainer',
    'MetricsCalculator',

    # Config
    'Config',

    # Utils
    'save_checkpoint',
    'load_checkpoint',
    'early_stopping_check',
    'EarlyStopping',
    'set_seed',
    'get_device',
    'print_model_summary'
]