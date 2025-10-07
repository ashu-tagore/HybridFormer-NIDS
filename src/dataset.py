"""
PyTorch Dataset classes and DataLoader utilities for NIDS project.

This module provides custom Dataset classes for loading preprocessed
network intrusion detection data from pickle files, plus utilities
for creating DataLoaders and computing class weights.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class NIDSDataset(Dataset):
    """
    Standard dataset for loading preprocessed NIDS data.

    Loads the full 42-feature dataset from pickle files created during
    preprocessing (notebook 03).

    Args:
        features_path: Path to features pickle file (e.g., train_features.pkl)
        labels_path: Path to labels pickle file (e.g., train_labels.pkl)

    Returns:
        Tuple of (features, label) where:
            - features: torch.FloatTensor of shape (42,)
            - label: torch.LongTensor of shape (1,)

    Example:
        >>> dataset = NIDSDataset(
        ...     'data/processed/train_features.pkl',
        ...     'data/processed/train_labels.pkl'
        ... )
        >>> features, label = dataset[0]
        >>> print(features.shape, label.shape)
        torch.Size([42]) torch.Size([1])
    """

    def __init__(
        self,
        features_path: str,
        labels_path: str
    ):
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)

        # Validate paths exist
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        # Load data
        logger.info(f"Loading features from {self.features_path}")
        logger.info(f"Loading labels from {self.labels_path}")

        self._load_data()

        # Validate data
        self._validate_data()

        logger.info(f"Dataset loaded successfully: {len(self)} samples, {self.num_features} features")

    def _load_data(self):
        """Load pickle files into memory."""
        try:
            with open(self.features_path, 'rb') as f:
                self.features = pickle.load(f)

            with open(self.labels_path, 'rb') as f:
                self.labels = pickle.load(f)

            # Convert pandas DataFrame to numpy array if needed
            if hasattr(self.features, 'values'):
                self.features = self.features.values
            if hasattr(self.labels, 'values'):
                self.labels = self.labels.values

        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def _validate_data(self):
        """Validate loaded data integrity."""
        # Check types
        if not isinstance(self.features, np.ndarray):
            raise TypeError(f"Features must be numpy array, got {type(self.features)}")
        if not isinstance(self.labels, np.ndarray):
            raise TypeError(f"Labels must be numpy array, got {type(self.labels)}")

        # Check shapes
        if len(self.features.shape) != 2:
            raise ValueError(f"Features must be 2D array, got shape {self.features.shape}")
        if len(self.labels.shape) != 1:
            raise ValueError(f"Labels must be 1D array, got shape {self.labels.shape}")

        # Check matching lengths
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Features and labels length mismatch: "
                f"{len(self.features)} vs {len(self.labels)}"
            )

        # Check feature dimension (should be 42 after preprocessing)
        if self.features.shape[1] != 42:
            logger.warning(
                f"Expected 42 features, got {self.features.shape[1]}. "
                f"This might be okay if you modified preprocessing."
            )

        # Check label range (should be 0-9 for 10 classes)
        unique_labels = np.unique(self.labels)
        if unique_labels.min() < 0 or unique_labels.max() > 9:
            logger.warning(
                f"Unexpected label range: {unique_labels.min()} to {unique_labels.max()}"
            )

        logger.info(f"Data validation passed")
        logger.info(f"  Features shape: {self.features.shape}")
        logger.info(f"  Labels shape: {self.labels.shape}")
        logger.info(f"  Unique labels: {sorted(unique_labels.tolist())}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (features, label) as PyTorch tensors
        """
        # Get features and label
        features = self.features[idx]
        label = self.labels[idx]

        # Convert to PyTorch tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label])

        return features_tensor, label_tensor

    @property
    def num_features(self) -> int:
        """Return the number of features per sample."""
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        """Return the number of unique classes."""
        return len(np.unique(self.labels))

    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class_id -> count
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_class_names(self) -> dict:
        """
        Get mapping of class IDs to attack names.

        Returns:
            Dictionary mapping class_id -> attack_name
        """
        return {
            0: "Benign",
            1: "Analysis",
            2: "Backdoor",
            3: "DoS",
            4: "Exploits",
            5: "Fuzzers",
            6: "Generic",
            7: "Reconnaissance",
            8: "Shellcode",
            9: "Worms"
        }


def get_data_loaders(
    train_features_path: str,
    train_labels_path: str,
    val_features_path: str,
    val_labels_path: str,
    test_features_path: str,
    test_labels_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        train_features_path: Path to training features pickle
        train_labels_path: Path to training labels pickle
        val_features_path: Path to validation features pickle
        val_labels_path: Path to validation labels pickle
        test_features_path: Path to test features pickle
        test_labels_path: Path to test labels pickle
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pinned memory (faster GPU transfer)

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders

    Example:
        >>> loaders = get_data_loaders(
        ...     'data/processed/train_features.pkl',
        ...     'data/processed/train_labels.pkl',
        ...     'data/processed/val_features.pkl',
        ...     'data/processed/val_labels.pkl',
        ...     'data/processed/test_features.pkl',
        ...     'data/processed/test_labels.pkl',
        ...     batch_size=64
        ... )
        >>> for features, labels in loaders['train']:
        ...     print(features.shape)  # (64, 42)
        ...     break
    """
    logger.info("Creating datasets...")

    # Create datasets
    train_dataset = NIDSDataset(train_features_path, train_labels_path)
    val_dataset = NIDSDataset(val_features_path, val_labels_path)
    test_dataset = NIDSDataset(test_features_path, test_labels_path)

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    logger.info(f"DataLoaders created successfully")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = 10,
    method: str = 'balanced'
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.

    Even though SMOTE was applied during preprocessing, validation and test
    sets remain imbalanced. Class weights help the model focus on minority classes.

    Args:
        labels: Array of class labels (shape: [N,])
        num_classes: Total number of classes
        method: Weighting method ('balanced' or 'inverse_frequency')

    Returns:
        Tensor of class weights (shape: [num_classes,])

    Example:
        >>> labels = np.array([0, 0, 0, 1, 2])
        >>> weights = compute_class_weights(labels, num_classes=3)
        >>> print(weights)
        tensor([0.4167, 1.2500, 1.2500])
    """
    # Count samples per class
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = np.zeros(num_classes)
    for cls, count in zip(unique, counts):
        class_counts[int(cls)] = count

    # Replace zeros with 1 to avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    if method == 'balanced':
        # sklearn-style balanced weights: n_samples / (n_classes * n_samples_per_class)
        total_samples = len(labels)
        weights = total_samples / (num_classes * class_counts)
    elif method == 'inverse_frequency':
        # Simple inverse frequency
        weights = 1.0 / class_counts
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to sum to num_classes (optional, helps with loss scaling)
    weights = weights / weights.sum() * num_classes

    # Convert to PyTorch tensor
    weights_tensor = torch.FloatTensor(weights)

    logger.info(f"Class weights computed ({method}):")
    for i, w in enumerate(weights):
        logger.info(f"  Class {i}: {w:.4f} (count: {int(class_counts[i])})")

    return weights_tensor
