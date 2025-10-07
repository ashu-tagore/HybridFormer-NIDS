"""
CNN Branch for HybridFormer.
Processes local feature patterns using 1D convolutions.

Input: (batch_size, 10) - CNN-specific features
Output: (batch_size, 512) - Rich feature representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNNBranch(nn.Module):
    """
    1D Convolutional Neural Network branch for local pattern extraction.

    Architecture:
        Input (B, 10) → Reshape → (B, 10, 1)
        ↓
        Conv1D Block 1: (B, 10, 1) → (B, 10, 64)
        ↓
        Conv1D Block 2: (B, 10, 64) → (B, 10, 128)
        ↓
        Conv1D Block 3: (B, 10, 128) → (B, 10, 256)
        ↓
        Global Pooling: (B, 10, 256) → (B, 512)

    Args:
        in_features: Number of input features (default: 10)
        channels: List of channel sizes for each conv block (default: [64, 128, 256])
        kernel_sizes: List of kernel sizes for each conv block (default: [3, 3, 3])
        dropout: Dropout rate (default: 0.1)
        use_batch_norm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        in_features: int = 10,
        channels: list = [64, 128, 256],
        kernel_sizes: list = [3, 3, 3],
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super(CNNBranch, self).__init__()

        self.in_features = in_features
        self.channels = channels
        self.output_dim = channels[-1] * 2  # *2 because we concat max and avg pooling

        assert len(channels) == len(kernel_sizes), \
            "Number of channels must match number of kernel sizes"

        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 1  # Single channel input (reshaped from features)

        for out_channels, kernel_size in zip(channels, kernel_sizes):
            block = self._make_conv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        # Global pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Initialize weights
        self._init_weights()

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        use_batch_norm: bool
    ) -> nn.Sequential:
        """
        Create a convolutional block with optional batch norm and dropout.

        Block structure:
            Conv1d → BatchNorm1d (optional) → ReLU → Dropout (optional)
        """
        layers = []

        # Convolution with padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_batch_norm  # No bias if using batch norm
        ))

        # Batch normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        # Activation
        layers.append(nn.ReLU(inplace=True))

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN branch.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, output_dim)
            where output_dim = channels[-1] * 2 (max + avg pooling)
        """
        batch_size = x.size(0)

        # Reshape: (batch_size, in_features) → (batch_size, in_features, 1)
        # This treats each feature as a "time step" in the sequence
        x = x.unsqueeze(-1)  # (B, 10, 1)

        # Transpose for Conv1d: (batch_size, 1, in_features)
        # Conv1d expects (batch, channels, sequence_length)
        x = x.transpose(1, 2)  # (B, 1, 10)

        # Pass through convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)  # (B, channels[i], 10)

        # Global pooling
        # x shape: (batch_size, channels[-1], in_features)
        max_pooled = self.global_max_pool(x)  # (B, channels[-1], 1)
        avg_pooled = self.global_avg_pool(x)  # (B, channels[-1], 1)

        # Remove the last dimension and concatenate
        max_pooled = max_pooled.squeeze(-1)  # (B, channels[-1])
        avg_pooled = avg_pooled.squeeze(-1)  # (B, channels[-1])

        # Concatenate max and average pooling
        output = torch.cat([max_pooled, avg_pooled], dim=1)  # (B, channels[-1] * 2)

        return output

    def get_output_dim(self) -> int:
        """Return the output dimension of this branch."""
        return self.output_dim


# ============================================================================
# Testing and Validation Functions
# ============================================================================

def test_cnn_branch_forward():
    """Test forward pass with different batch sizes."""
    print("\n" + "="*70)
    print("TESTING CNN BRANCH FORWARD PASS")
    print("="*70)

    model = CNNBranch(in_features=10)
    model.eval()

    print(f"\nModel architecture:")
    print(f"  Input features: {model.in_features}")
    print(f"  Channels: {model.channels}")
    print(f"  Output dimension: {model.output_dim}")

    # Test with different batch sizes
    batch_sizes = [1, 16, 64, 128]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 10)

        try:
            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, 512), \
                f"Expected shape ({batch_size}, 512), got {output.shape}"

            print(f"✓ Batch size {batch_size:3d}: input {x.shape} → output {output.shape}")

        except Exception as e:
            print(f"✗ Batch size {batch_size:3d} failed: {e}")
            return False

    print("\n✓ All forward passes successful")
    return True


def test_cnn_branch_backward():
    """Test backward pass and gradient flow."""
    print("\n" + "="*70)
    print("TESTING CNN BRANCH BACKWARD PASS")
    print("="*70)

    model = CNNBranch(in_features=10)
    model.train()

    # Create dummy data and target
    x = torch.randn(64, 10, requires_grad=True)
    target = torch.randn(64, 512)

    # Forward pass
    output = model(x)

    # Compute loss
    loss = F.mse_loss(output, target)

    # Backward pass
    try:
        loss.backward()
        print(f"✓ Backward pass successful")
        print(f"  Loss: {loss.item():.4f}")

        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm = {grad_norm:.4f}")

        assert has_grad, "No gradients computed"
        print("\n✓ Gradients computed successfully")
        return True

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False


def test_cnn_branch_on_real_data():
    """Test CNN branch on real NIDS data."""
    print("\n" + "="*70)
    print("TESTING CNN BRANCH ON REAL DATA")
    print("="*70)

    from pathlib import Path
    import sys

    # Add src to path
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
        cnn_features = features['cnn']

        print(f"✓ Loaded real data")
        print(f"  CNN features shape: {cnn_features.shape}")
        print(f"  Labels shape: {labels.shape}")

        # Test forward pass
        model = CNNBranch(in_features=10)
        model.eval()

        with torch.no_grad():
            output = model(cnn_features)

        print(f"\n✓ Forward pass on real data successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  Output mean: {output.mean():.4f}")
        print(f"  Output std: {output.std():.4f}")

        return True

    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_branch_gpu():
    """Test CNN branch on GPU if available."""
    print("\n" + "="*70)
    print("TESTING CNN BRANCH ON GPU")
    print("="*70)

    if not torch.cuda.is_available():
        print("✗ GPU not available, skipping GPU test")
        return True

    device = torch.device('cuda')
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")

    try:
        model = CNNBranch(in_features=10).to(device)
        x = torch.randn(64, 10).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.is_cuda, "Output should be on GPU"
        assert output.shape == (64, 512), f"Wrong output shape: {output.shape}"

        print(f"✓ GPU forward pass successful")
        print(f"  Input device: {x.device}")
        print(f"  Output device: {output.device}")
        print(f"  Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ GPU test failed: {e}")
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

    model = CNNBranch(in_features=10)

    total_params = count_parameters(model)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Breakdown by layer type
    conv_params = sum(p.numel() for n, p in model.named_parameters()
                     if 'conv' in n and p.requires_grad)
    bn_params = sum(p.numel() for n, p in model.named_parameters()
                   if 'batch_norm' in n and p.requires_grad)

    print(f"\nParameter breakdown:")
    print(f"  Convolutional layers: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
    print(f"  Batch norm layers: {bn_params:,} ({bn_params/total_params*100:.1f}%)")

    print(f"\n✓ Model size analysis complete")
    return True


if __name__ == '__main__':
    """Run all tests when script is executed directly."""
    print("\n" + "="*70)
    print("RUNNING CNN BRANCH TESTS")
    print("="*70)

    all_passed = True

    # Run tests
    all_passed &= test_cnn_branch_forward()
    all_passed &= test_cnn_branch_backward()
    all_passed &= test_model_size()
    all_passed &= test_cnn_branch_gpu()
    all_passed &= test_cnn_branch_on_real_data()

    if all_passed:
        print("\n" + "="*70)
        print("ALL CNN BRANCH TESTS PASSED ✓✓✓")
        print("Ready to integrate into HybridFormer!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ✗")
        print("Please fix errors before proceeding")
        print("="*70)
