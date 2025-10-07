# test_setup.py
import sys
from pathlib import Path

# Test 1: Import all modules
print("Testing imports...")
try:
    from src import (
        NIDSDataset,
        get_data_loaders,
        BaselineFFN,
        Trainer,
        MetricsCalculator,
        Config,
        set_seed
    )
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Load config
print("\nTesting configuration...")
try:
    config = Config('configs/baseline.yaml')
    print(f"✓ Config loaded!")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
except Exception as e:
    print(f"✗ Config error: {e}")
    sys.exit(1)

# Test 3: Load dataset
print("\nTesting dataset...")
try:
    dataset = NIDSDataset(
        'data/processed/train_features.pkl',
        'data/processed/train_labels.pkl'
    )
    print(f"✓ Dataset loaded!")
    print(f"  Samples: {len(dataset):,}")
    print(f"  Features: {dataset.num_features}")
    print(f"  Classes: {dataset.num_classes}")

    # Test getting one sample
    features, label = dataset[0]
    print(f"  Sample shape: features={features.shape}, label={label.shape}")
except Exception as e:
    print(f"✗ Dataset error: {e}")
    sys.exit(1)

# Test 4: Create model
print("\nTesting model...")
try:
    import torch
    model = BaselineFFN(
        input_dim=42,
        hidden_dims=[256, 128, 64],
        num_classes=10
    )
    print(f"✓ Model created!")

    # Test forward pass
    test_input = torch.randn(8, 42)
    test_output = model(test_input)
    print(f"  Forward pass: {test_input.shape} → {test_output.shape}")
except Exception as e:
    print(f"✗ Model error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYou're ready to train! Run:")
print("  python train.py")
print("="*60)
