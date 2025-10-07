"""
Dataset and DataLoader utilities for NIDS project.
Supports both flat features (baseline) and branch-specific features (HybridFormer).
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NIDSDataset(Dataset):
    """
    Network Intrusion Detection Dataset.

    Supports two modes:
    1. Flat mode: Returns features as single tensor (for baseline models)
    2. Branch mode: Returns features as dictionary (for HybridFormer)

    Args:
        features: Feature tensor or path to .pkl file
        labels: Label tensor or path to .pkl file
        branch_allocation: Optional path to branch_feature_allocation.json
        mode: 'flat' or 'branch' (default: 'flat')
    """

    def __init__(
        self,
        features: Union[torch.Tensor, str, Path],
        labels: Union[torch.Tensor, str, Path],
        branch_allocation: Optional[Union[str, Path]] = None,
        mode: str = 'flat'
    ):
        self.mode = mode

        # Load features
        if isinstance(features, (str, Path)):
            logger.info(f"Loading features from {features}")
            with open(features, 'rb') as f:
                self.features = pickle.load(f)
        else:
            self.features = features

        # Load labels
        if isinstance(labels, (str, Path)):
            logger.info(f"Loading labels from {labels}")
            with open(labels, 'rb') as f:
                self.labels = pickle.load(f)
        else:
            self.labels = labels

        # Convert pandas DataFrame/Series to numpy if needed
        if isinstance(self.features, pd.DataFrame):
            logger.info("Converting features DataFrame to numpy array")
            self.features = self.features.values
        if isinstance(self.labels, pd.Series):
            logger.info("Converting labels Series to numpy array")
            self.labels = self.labels.values

        # Convert to tensors if needed
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.FloatTensor(self.features)
        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.LongTensor(self.labels)

        # Validate shapes
        assert len(self.features) == len(self.labels), \
            f"Feature count {len(self.features)} != label count {len(self.labels)}"

        logger.info(f"Loaded dataset: {len(self.features)} samples, {self.features.shape[1]} features")

        # Load branch allocation if in branch mode
        self.branch_indices = None
        if mode == 'branch':
            if branch_allocation is None:
                raise ValueError("branch_allocation must be provided in 'branch' mode")
            self._load_branch_allocation(branch_allocation)

    def _load_branch_allocation(self, allocation_path: Union[str, Path]):
        """Load feature allocation for each branch."""
        logger.info(f"Loading branch allocation from {allocation_path}")

        with open(allocation_path, 'r') as f:
            allocation = json.load(f)

        # Extract feature indices for each branch
        self.branch_indices = {
            'cnn': allocation['cnn_branch']['feature_indices'],
            'transformer': allocation['transformer_branch']['feature_indices'],
            'graph': allocation['graph_branch']['feature_indices']
        }

        # Validate indices
        total_features = self.features.shape[1]
        for branch_name, indices in self.branch_indices.items():
            if not indices:
                raise ValueError(f"No features allocated to {branch_name} branch")
            if max(indices) >= total_features:
                raise ValueError(
                    f"Invalid feature index {max(indices)} for {branch_name} "
                    f"(total features: {total_features})"
                )

        # Log allocation info
        logger.info("Branch feature allocation:")
        logger.info(f"  CNN: {len(self.branch_indices['cnn'])} features")
        logger.info(f"  Transformer: {len(self.branch_indices['transformer'])} features")
        logger.info(f"  Graph: {len(self.branch_indices['graph'])} features")

        # Verify no feature overlap (each feature should go to exactly one branch)
        all_indices = (self.branch_indices['cnn'] +
                      self.branch_indices['transformer'] +
                      self.branch_indices['graph'])
        if len(all_indices) != len(set(all_indices)):
            logger.warning("Warning: Some features are allocated to multiple branches")

        # Verify all features are used
        if set(all_indices) != set(range(total_features)):
            unused = set(range(total_features)) - set(all_indices)
            logger.warning(f"Warning: {len(unused)} features not allocated to any branch")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Get a single sample.

        Returns:
            If mode='flat': (features, label)
            If mode='branch': ({'cnn': tensor, 'transformer': tensor, 'graph': tensor}, label)
        """
        features = self.features[idx]
        label = self.labels[idx]

        if self.mode == 'flat':
            return features, label

        elif self.mode == 'branch':
            # Split features according to branch allocation
            branch_features = {
                'cnn': features[self.branch_indices['cnn']],
                'transformer': features[self.branch_indices['transformer']],
                'graph': features[self.branch_indices['graph']]
            }
            return branch_features, label

        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'flat' or 'branch'")


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 4,
    mode: str = 'flat',
    branch_allocation_path: Optional[Union[str, Path]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing processed data files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        mode: 'flat' or 'branch'
        branch_allocation_path: Path to branch_feature_allocation.json (required if mode='branch')

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    # Create datasets
    train_dataset = NIDSDataset(
        features=data_dir / 'train_features.pkl',
        labels=data_dir / 'train_labels.pkl',
        branch_allocation=branch_allocation_path,
        mode=mode
    )

    val_dataset = NIDSDataset(
        features=data_dir / 'val_features.pkl',
        labels=data_dir / 'val_labels.pkl',
        branch_allocation=branch_allocation_path,
        mode=mode
    )

    test_dataset = NIDSDataset(
        features=data_dir / 'test_features.pkl',
        labels=data_dir / 'test_labels.pkl',
        branch_allocation=branch_allocation_path,
        mode=mode
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created dataloaders with batch_size={batch_size}, mode={mode}")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def get_class_distribution(labels: torch.Tensor) -> Dict[int, int]:
    """
    Get class distribution for dataset analysis.

    Args:
        labels: Label tensor

    Returns:
        Dictionary mapping class index to count
    """
    unique, counts = torch.unique(labels, return_counts=True)
    distribution = {int(cls): int(count) for cls, count in zip(unique, counts)}
    return distribution


def compute_class_weights(labels: torch.Tensor, method: str = 'balanced') -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        labels: Label tensor
        method: 'balanced' (inverse frequency) or 'effective' (effective number of samples)

    Returns:
        Class weights tensor
    """
    unique, counts = torch.unique(labels, return_counts=True)
    num_classes = len(unique)

    if method == 'balanced':
        # Inverse frequency: weight = total_samples / (num_classes * class_count)
        total = len(labels)
        weights = total / (num_classes * counts.float())

    elif method == 'effective':
        # Effective number of samples: weight = (1 - beta) / (1 - beta^n)
        # where beta = (N-1)/N, N is total samples
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts.float())
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes

    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Computed class weights using '{method}' method:")
    for cls, weight in zip(unique, weights):
        logger.info(f"  Class {int(cls)}: {weight:.4f}")

    return weights


# ============================================================================
# Testing and Validation Functions
# ============================================================================

def test_dataset_modes():
    """Test that dataset works in both flat and branch modes."""
    print("\n" + "="*70)
    print("TESTING DATASET MODES")
    print("="*70)

    data_dir = Path('data/processed')

    # Test flat mode (backward compatibility)
    print("\n1. Testing FLAT mode (for baseline model)...")
    try:
        dataset_flat = NIDSDataset(
            features=data_dir / 'val_features.pkl',
            labels=data_dir / 'val_labels.pkl',
            mode='flat'
        )
        features, label = dataset_flat[0]
        print(f"✓ Flat mode works")
        print(f"  Features shape: {features.shape}")
        print(f"  Label: {label.item()}")
        assert features.dim() == 1, "Flat features should be 1D"
    except Exception as e:
        print(f"✗ Flat mode failed: {e}")
        return False

    # Test branch mode (for HybridFormer)
    print("\n2. Testing BRANCH mode (for HybridFormer)...")
    try:
        dataset_branch = NIDSDataset(
            features=data_dir / 'val_features.pkl',
            labels=data_dir / 'val_labels.pkl',
            branch_allocation=data_dir / 'branch_feature_allocation.json',
            mode='branch'
        )
        branch_features, label = dataset_branch[0]
        print(f"✓ Branch mode works")
        print(f"  CNN features shape: {branch_features['cnn'].shape}")
        print(f"  Transformer features shape: {branch_features['transformer'].shape}")
        print(f"  Graph features shape: {branch_features['graph'].shape}")
        print(f"  Label: {label.item()}")

        assert isinstance(branch_features, dict), "Branch features should be dict"
        assert 'cnn' in branch_features, "Missing CNN features"
        assert 'transformer' in branch_features, "Missing Transformer features"
        assert 'graph' in branch_features, "Missing Graph features"
    except Exception as e:
        print(f"✗ Branch mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test dataloader batching
    print("\n3. Testing DataLoader batching...")
    try:
        loader = DataLoader(dataset_branch, batch_size=64, shuffle=False)
        batch = next(iter(loader))
        features, labels = batch

        print(f"✓ DataLoader batching works")
        print(f"  Batch size: {labels.shape[0]}")
        print(f"  CNN batch shape: {features['cnn'].shape}")
        print(f"  Transformer batch shape: {features['transformer'].shape}")
        print(f"  Graph batch shape: {features['graph'].shape}")

        assert features['cnn'].shape[0] == 64, "Wrong batch size"
        assert features['cnn'].dim() == 2, "CNN features should be 2D (batch, features)"
    except Exception as e:
        print(f"✗ DataLoader batching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    return True


def test_create_dataloaders():
    """Test the create_dataloaders convenience function."""
    print("\n" + "="*70)
    print("TESTING DATALOADER CREATION")
    print("="*70)

    # Test branch mode dataloaders
    print("\nCreating dataloaders in BRANCH mode...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir='data/processed',
            batch_size=64,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            mode='branch',
            branch_allocation_path='data/processed/branch_feature_allocation.json'
        )

        print(f"✓ Dataloaders created successfully")

        # Test one batch from each loader
        print("\nTesting batch extraction...")
        for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
            batch = next(iter(loader))
            features, labels = batch
            print(f"  {name}: batch_size={labels.shape[0]}, "
                  f"cnn={features['cnn'].shape[1]}, "
                  f"trans={features['transformer'].shape[1]}, "
                  f"graph={features['graph'].shape[1]}")

        print("\n✓ All loaders working correctly")
        return True

    except Exception as e:
        print(f"✗ Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_weights():
    """Test class weight computation."""
    print("\n" + "="*70)
    print("TESTING CLASS WEIGHT COMPUTATION")
    print("="*70)

    data_dir = Path('data/processed')

    # Load validation labels for testing
    with open(data_dir / 'val_labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    # Convert pandas Series to numpy if needed
    if isinstance(labels, pd.Series):
        labels = labels.values

    labels = torch.LongTensor(labels)

    # Get class distribution
    print("\nClass distribution:")
    dist = get_class_distribution(labels)
    for cls, count in sorted(dist.items()):
        print(f"  Class {cls}: {count} samples")

    # Compute weights
    print("\nComputing class weights...")
    weights = compute_class_weights(labels, method='balanced')

    print("\n✓ Class weights computed successfully")
    return True


if __name__ == '__main__':
    """Run all tests when script is executed directly."""
    print("\n" + "="*70)
    print("RUNNING DATASET.PY TESTS")
    print("="*70)

    all_passed = True

    # Run tests
    all_passed &= test_dataset_modes()
    all_passed &= test_create_dataloaders()
    all_passed &= test_class_weights()

    if all_passed:
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓✓✓")
        print("Dataset.py is ready for HybridFormer!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ✗")
        print("Please fix errors before proceeding")
        print("="*70)
