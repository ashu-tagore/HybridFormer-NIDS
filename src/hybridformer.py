"""
HybridFormer - Multi-Modal Network Intrusion Detection System
Combines CNN, Transformer, and Graph branches with simple concatenation fusion.

Architecture:
    Input (B, 42) → Split into branches
    ├── CNN Branch (B, 10) → (B, 512)
    ├── Transformer Branch (B, 12) → (B, 256)
    └── Graph Branch (B, 20) → (B, 256)
    → Concatenate → (B, 1024)
    → Classification Head → (B, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Handle imports for both module usage and direct testing
try:
    from .cnn_branch import CNNBranch
    from .transformer_branch import TransformerBranch
    from .graph_branch import GraphBranch
except ImportError:
    from cnn_branch import CNNBranch
    from transformer_branch import TransformerBranch
    from graph_branch import GraphBranch


class HybridFormer(nn.Module):
    """
    HybridFormer with simple concatenation fusion.

    This is the baseline version that concatenates branch outputs
    before classification. Cross-attention fusion can be added later.

    Args:
        num_classes: Number of output classes (default: 10)
        dropout: Dropout rate for classification head (default: 0.3)
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes

        # Branch models
        self.cnn_branch = CNNBranch(in_features=10)
        self.transformer_branch = TransformerBranch(in_features=12)
        self.graph_branch = GraphBranch(in_features=20)

        # Get output dimensions from each branch
        cnn_dim = self.cnn_branch.get_output_dim()  # 512
        trans_dim = self.transformer_branch.get_output_dim()  # 256
        graph_dim = self.graph_branch.get_output_dim()  # 256

        # Total concatenated dimension
        self.fused_dim = cnn_dim + trans_dim + graph_dim  # 1024

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.67),  # 0.2

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.33),  # 0.1

            nn.Linear(128, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classifier weights using Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through HybridFormer.

        Args:
            x: Dictionary with keys 'cnn', 'transformer', 'graph'
               Each value is a tensor of shape (batch_size, num_features)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract branch-specific features
        cnn_features = x['cnn']  # (B, 10)
        trans_features = x['transformer']  # (B, 12)
        graph_features = x['graph']  # (B, 20)

        # Pass through each branch
        cnn_out = self.cnn_branch(cnn_features)  # (B, 512)
        trans_out = self.transformer_branch(trans_features)  # (B, 256)
        graph_out = self.graph_branch(graph_features)  # (B, 256)

        # Concatenate branch outputs
        fused = torch.cat([cnn_out, trans_out, graph_out], dim=1)  # (B, 1024)

        # Classification
        logits = self.classifier(fused)  # (B, 10)

        return logits

    def get_branch_outputs(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get outputs from each branch separately (for analysis).

        Args:
            x: Dictionary with branch-specific features

        Returns:
            Dictionary with branch outputs
        """
        with torch.no_grad():
            cnn_out = self.cnn_branch(x['cnn'])
            trans_out = self.transformer_branch(x['transformer'])
            graph_out = self.graph_branch(x['graph'])

        return {
            'cnn': cnn_out,
            'transformer': trans_out,
            'graph': graph_out
        }


# ============================================================================
# Testing and Validation Functions
# ============================================================================

def test_hybridformer_forward():
    """Test forward pass with different batch sizes."""
    print("\n" + "="*70)
    print("TESTING HYBRIDFORMER FORWARD PASS")
    print("="*70)

    model = HybridFormer(num_classes=10)
    model.eval()

    print(f"\nModel architecture:")
    print(f"  CNN output: 512")
    print(f"  Transformer output: 256")
    print(f"  Graph output: 256")
    print(f"  Fused dimension: {model.fused_dim}")
    print(f"  Number of classes: {model.num_classes}")

    # Test with different batch sizes
    batch_sizes = [1, 16, 64, 128]

    for batch_size in batch_sizes:
        x = {
            'cnn': torch.randn(batch_size, 10),
            'transformer': torch.randn(batch_size, 12),
            'graph': torch.randn(batch_size, 20)
        }

        try:
            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, 10), \
                f"Expected shape ({batch_size}, 10), got {output.shape}"

            print(f"✓ Batch size {batch_size:3d}: output {output.shape}")

        except (RuntimeError, AssertionError) as e:
            print(f"✗ Batch size {batch_size:3d} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ All forward passes successful")
    return True


def test_hybridformer_backward():
    """Test backward pass and gradient flow."""
    print("\n" + "="*70)
    print("TESTING HYBRIDFORMER BACKWARD PASS")
    print("="*70)

    model = HybridFormer(num_classes=10)
    model.train()

    # Create dummy data and target
    x = {
        'cnn': torch.randn(64, 10),
        'transformer': torch.randn(64, 12),
        'graph': torch.randn(64, 20)
    }
    target = torch.randint(0, 10, (64,))

    # Forward pass
    output = model(x)

    # Compute loss
    loss = F.cross_entropy(output, target)

    # Backward pass
    try:
        loss.backward()
        print(f"✓ Backward pass successful")
        print(f"  Loss: {loss.item():.4f}")

        # Check gradients in each branch
        print("\n  Gradient norms by component:")

        cnn_grads = sum(p.grad.norm().item() for p in model.cnn_branch.parameters()
                       if p.grad is not None)
        trans_grads = sum(p.grad.norm().item() for p in model.transformer_branch.parameters()
                         if p.grad is not None)
        graph_grads = sum(p.grad.norm().item() for p in model.graph_branch.parameters()
                         if p.grad is not None)
        classifier_grads = sum(p.grad.norm().item() for p in model.classifier.parameters()
                              if p.grad is not None)

        print(f"    CNN branch: {cnn_grads:.4f}")
        print(f"    Transformer branch: {trans_grads:.4f}")
        print(f"    Graph branch: {graph_grads:.4f}")
        print(f"    Classifier: {classifier_grads:.4f}")

        assert cnn_grads > 0, "CNN branch has no gradients"
        assert trans_grads > 0, "Transformer branch has no gradients"
        assert graph_grads > 0, "Graph branch has no gradients"
        assert classifier_grads > 0, "Classifier has no gradients"

        print("\n✓ All gradients computed successfully")
        return True

    except (RuntimeError, AssertionError) as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybridformer_on_real_data():
    """Test HybridFormer on real NIDS data."""
    print("\n" + "="*70)
    print("TESTING HYBRIDFORMER ON REAL DATA")
    print("="*70)

    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from dataset import create_dataloaders

        print("\nLoading real data...")
        _, val_loader, _ = create_dataloaders(
            data_dir='data/processed',
            batch_size=64,
            num_workers=0,
            mode='branch',
            branch_allocation_path='data/processed/branch_feature_allocation.json'
        )

        # Get one batch
        batch = next(iter(val_loader))
        features, labels = batch

        print(f"✓ Loaded real data")
        print(f"  CNN features: {features['cnn'].shape}")
        print(f"  Transformer features: {features['transformer'].shape}")
        print(f"  Graph features: {features['graph'].shape}")
        print(f"  Labels: {labels.shape}")

        # Test forward pass
        model = HybridFormer(num_classes=10)
        model.eval()

        with torch.no_grad():
            output = model(features)

        # Get predictions
        probabilities = F.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        print(f"\n✓ Forward pass on real data successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Logits range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
        print(f"  Predictions: {predictions[:10].tolist()}")
        print(f"  Ground truth: {labels[:10].tolist()}")

        # Quick accuracy check (untrained, so will be ~10% random)
        accuracy = (predictions == labels).float().mean().item()
        print(f"  Random accuracy: {accuracy*100:.2f}% (expected ~10% for untrained)")

        return True

    except (RuntimeError, ImportError, KeyError) as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybridformer_gpu():
    """Test HybridFormer on GPU if available."""
    print("\n" + "="*70)
    print("TESTING HYBRIDFORMER ON GPU")
    print("="*70)

    if not torch.cuda.is_available():
        print("✗ GPU not available, skipping GPU test")
        return True

    device = torch.device('cuda')
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")

    try:
        model = HybridFormer(num_classes=10).to(device)
        x = {
            'cnn': torch.randn(64, 10).to(device),
            'transformer': torch.randn(64, 12).to(device),
            'graph': torch.randn(64, 20).to(device)
        }

        with torch.no_grad():
            output = model(x)

        assert output.is_cuda, "Output should be on GPU"
        assert output.shape == (64, 10), f"Wrong output shape: {output.shape}"

        print(f"✓ GPU forward pass successful")
        print(f"  Output device: {output.device}")
        print(f"  Output shape: {output.shape}")

        return True

    except (RuntimeError, AssertionError) as e:
        print(f"✗ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def count_parameters(model):
    """Count trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def test_model_size():
    """Test and report model size."""
    print("\n" + "="*70)
    print("TESTING MODEL SIZE")
    print("="*70)

    model = HybridFormer(num_classes=10)

    total_params = count_parameters(model)
    cnn_params = count_parameters(model.cnn_branch)
    trans_params = count_parameters(model.transformer_branch)
    graph_params = count_parameters(model.graph_branch)
    classifier_params = count_parameters(model.classifier)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print(f"\nParameter breakdown:")
    print(f"  CNN branch: {cnn_params:,} ({cnn_params/total_params*100:.1f}%)")
    print(f"  Transformer branch: {trans_params:,} ({trans_params/total_params*100:.1f}%)")
    print(f"  Graph branch: {graph_params:,} ({graph_params/total_params*100:.1f}%)")
    print(f"  Classifier: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")

    print(f"\n✓ Model size analysis complete")
    return True


if __name__ == '__main__':
    """Run all tests when script is executed directly."""
    print("\n" + "="*70)
    print("RUNNING HYBRIDFORMER TESTS")
    print("="*70)

    all_passed = True

    # Run tests
    all_passed &= test_hybridformer_forward()
    all_passed &= test_hybridformer_backward()
    all_passed &= test_model_size()
    all_passed &= test_hybridformer_gpu()
    all_passed &= test_hybridformer_on_real_data()

    if all_passed:
        print("\n" + "="*70)
        print("ALL HYBRIDFORMER TESTS PASSED ✓✓✓")
        print("Ready for training!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ✗")
        print("Please fix errors before proceeding")
        print("="*70)
