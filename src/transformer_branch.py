"""
Transformer Branch for HybridFormer.
Processes sequential/temporal dependencies using multi-head attention.

Input: (batch_size, 12) - Transformer-specific features
Output: (batch_size, 256) - Rich feature representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for feature importance.

    Unlike standard sinusoidal positional encoding (used for sequences),
    this learns which features are most important via trainable embeddings.
    """

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch_size, num_features, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pos_embedding
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block.

    Architecture:
        Input → Multi-Head Attention → Add & Norm →
        Feed Forward → Add & Norm → Output
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attn_mask: Optional attention mask

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Multi-head self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)

        return x


class TransformerBranch(nn.Module):
    """
    Transformer encoder branch for sequential feature processing.

    Architecture:
        Input (B, 12) → Project → (B, 12*256) → Reshape → (B, 12, 256)
        ↓
        Positional Encoding
        ↓
        Transformer Blocks (4 layers)
        ↓
        Mean Pooling → (B, 256)

    Args:
        in_features: Number of input features (default: 12)
        d_model: Dimension of transformer model (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer blocks (default: 4)
        dim_feedforward: Dimension of feed-forward network (default: 512)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        in_features: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_features = in_features
        self.d_model = d_model
        self.output_dim = d_model

        # Input projection: (B, 12) → (B, 12*256)
        # This creates a rich representation for each feature
        self.input_projection = nn.Linear(in_features, d_model * in_features)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(in_features, d_model, dropout)

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output layer normalization
        self.output_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer branch.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, d_model)
        """
        batch_size = x.size(0)

        # Project input: (B, 12) → (B, 12*256)
        x = self.input_projection(x)  # (B, 3072)

        # Reshape to sequence: (B, 3072) → (B, 12, 256)
        # Each of the 12 features now has a 256-dimensional representation
        x = x.view(batch_size, self.in_features, self.d_model)  # (B, 12, 256)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, 12, 256)

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # (B, 12, 256)

        # Global mean pooling across sequence dimension
        # (B, 12, 256) → (B, 256)
        x = x.mean(dim=1)

        # Final normalization
        x = self.output_norm(x)

        return x

    def get_output_dim(self) -> int:
        """Return the output dimension of this branch."""
        return self.output_dim


# ============================================================================
# Testing and Validation Functions
# ============================================================================

def test_transformer_branch_forward():
    """Test forward pass with different batch sizes."""
    print("\n" + "="*70)
    print("TESTING TRANSFORMER BRANCH FORWARD PASS")
    print("="*70)

    model = TransformerBranch(in_features=12)
    model.eval()

    print(f"\nModel architecture:")
    print(f"  Input features: {model.in_features}")
    print(f"  Model dimension: {model.d_model}")
    print(f"  Output dimension: {model.output_dim}")

    # Test with different batch sizes
    batch_sizes = [1, 16, 64, 128]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 12)

        try:
            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, 256), \
                f"Expected shape ({batch_size}, 256), got {output.shape}"

            print(f"✓ Batch size {batch_size:3d}: input {x.shape} → output {output.shape}")

        except Exception as e:
            print(f"✗ Batch size {batch_size:3d} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ All forward passes successful")
    return True


def test_transformer_branch_backward():
    """Test backward pass and gradient flow."""
    print("\n" + "="*70)
    print("TESTING TRANSFORMER BRANCH BACKWARD PASS")
    print("="*70)

    model = TransformerBranch(in_features=12)
    model.train()

    # Create dummy data and target
    x = torch.randn(64, 12, requires_grad=True)
    target = torch.randn(64, 256)

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
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if len(grad_norms) <= 5:  # Print first 5
                    print(f"  {name[:50]}: grad_norm = {grad_norm:.4f}")

        assert len(grad_norms) > 0, "No gradients computed"
        print(f"  ... and {len(grad_norms) - 5} more layers")
        print("\n✓ Gradients computed successfully")
        return True

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_branch_on_real_data():
    """Test Transformer branch on real NIDS data."""
    print("\n" + "="*70)
    print("TESTING TRANSFORMER BRANCH ON REAL DATA")
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
        transformer_features = features['transformer']

        print(f"✓ Loaded real data")
        print(f"  Transformer features shape: {transformer_features.shape}")
        print(f"  Labels shape: {labels.shape}")

        # Test forward pass
        model = TransformerBranch(in_features=12)
        model.eval()

        with torch.no_grad():
            output = model(transformer_features)

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


def test_transformer_branch_gpu():
    """Test Transformer branch on GPU if available."""
    print("\n" + "="*70)
    print("TESTING TRANSFORMER BRANCH ON GPU")
    print("="*70)

    if not torch.cuda.is_available():
        print("✗ GPU not available, skipping GPU test")
        return True

    device = torch.device('cuda')
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")

    try:
        model = TransformerBranch(in_features=12).to(device)
        x = torch.randn(64, 12).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.is_cuda, "Output should be on GPU"
        assert output.shape == (64, 256), f"Wrong output shape: {output.shape}"

        print(f"✓ GPU forward pass successful")
        print(f"  Input device: {x.device}")
        print(f"  Output device: {output.device}")
        print(f"  Output shape: {output.shape}")

        return True

    except Exception as e:
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

    model = TransformerBranch(in_features=12)

    total_params = count_parameters(model)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print(f"\n✓ Model size analysis complete")
    return True


if __name__ == '__main__':
    """Run all tests when script is executed directly."""
    print("\n" + "="*70)
    print("RUNNING TRANSFORMER BRANCH TESTS")
    print("="*70)

    all_passed = True

    # Run tests
    all_passed &= test_transformer_branch_forward()
    all_passed &= test_transformer_branch_backward()
    all_passed &= test_model_size()
    all_passed &= test_transformer_branch_gpu()
    all_passed &= test_transformer_branch_on_real_data()

    if all_passed:
        print("\n" + "="*70)
        print("ALL TRANSFORMER BRANCH TESTS PASSED ✓✓✓")
        print("Ready to integrate into HybridFormer!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ✗")
        print("Please fix errors before proceeding")
        print("="*70)
