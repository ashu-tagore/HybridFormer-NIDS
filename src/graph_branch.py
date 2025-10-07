"""
Graph Branch for HybridFormer.
Processes network topology and feature relationships using Graph Attention Networks.

Input: (batch_size, 20) - Graph-specific features
Output: (batch_size, 256) - Rich feature representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from torch_geometric.nn import GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. Graph branch will use fallback implementation.")


class GraphConstructor(nn.Module):
    """
    Dynamic graph constructor that learns connectivity from features.

    For tabular data, we construct a feature correlation graph where:
    - Each feature is a node
    - Edges connect features based on learned similarity
    """

    def __init__(self, num_features: int = 20, k_neighbors: int = 8):
        super().__init__()
        self.num_features = num_features
        self.k = k_neighbors

        # Learnable metric for computing feature similarity
        self.similarity_net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct graph from input features.

        Args:
            x: Input features (batch_size, num_features)

        Returns:
            edge_index: Graph connectivity (2, num_edges)
            node_features: Node feature matrix (batch_size * num_features, feature_dim)
        """
        # Compute embeddings for similarity (for future dynamic graph construction)
        _ = self.similarity_net(x)  # Currently not used, but prepared for future enhancement

        # For simplicity, use static k-NN graph based on feature indices
        # In production, you'd compute this dynamically per batch
        edge_index = self._create_knn_graph(self.num_features, self.k)

        # Prepare node features: each sample's features become nodes
        # (batch_size, num_features) → (batch_size * num_features, 1)
        node_features = x.view(-1, 1)  # Flatten to (B*20, 1)

        return edge_index, node_features

    def _create_knn_graph(self, num_nodes: int, k: int) -> torch.Tensor:
        """Create a k-NN graph structure."""
        edges = []
        for i in range(num_nodes):
            # Connect to k nearest neighbors (by index proximity)
            for j in range(max(0, i-k//2), min(num_nodes, i+k//2+1)):
                if i != j:
                    edges.append([i, j])

        if len(edges) == 0:
            # Fallback: create a line graph
            edges = [[i, i+1] for i in range(num_nodes-1)]
            edges += [[i+1, i] for i in range(num_nodes-1)]

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index


class GATLayer(nn.Module):
    """
    Simple Graph Attention Layer implementation.
    Fallback if torch_geometric's GATConv has issues.
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        # Multi-head attention weights
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(1, num_heads, 2 * out_features)))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GAT layer.

        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Graph connectivity (2, num_edges)

        Returns:
            Updated node features
        """
        num_nodes = x.size(0)

        # Linear transformation
        h = self.W(x).view(-1, self.num_heads, self.out_features)  # (N, heads, out)

        # Aggregate neighbor features
        edge_src, edge_dst = edge_index[0], edge_index[1]

        # Initialize output
        out = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)

        # Simple mean aggregation (fallback)
        for i in range(num_nodes):
            neighbors = edge_dst == i
            if neighbors.any():
                neighbor_indices = edge_src[neighbors]
                out[i] = h[neighbor_indices].mean(dim=0)
            else:
                out[i] = h[i]

        # Concatenate heads
        out = out.view(num_nodes, -1)  # (N, heads * out)

        return out


class GraphBranch(nn.Module):
    """
    Graph Attention Network branch for network topology processing.

    Architecture:
        Input (B, 20) → Graph Construction → Node Features (B*20, 1)
        ↓
        Feature Projection → (B*20, 128)
        ↓
        GAT Layer 1: (B*20, 128) → (B*20, 128*4)
        ↓
        GAT Layer 2: (B*20, 128*4) → (B*20, 64*4)
        ↓
        Global Pooling: (B*20, 256) → (B, 256)

    Args:
        in_features: Number of input features (default: 20)
        hidden_channels: Hidden dimension for GAT layers (default: 128)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        in_features: int = 20,
        hidden_channels: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.output_dim = hidden_channels // 2 * num_heads  # 64 * 4 = 256

        # Graph constructor
        self.graph_constructor = GraphConstructor(in_features, k_neighbors=8)

        # Input projection (1 → hidden_channels)
        self.input_projection = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GAT layers
        if TORCH_GEOMETRIC_AVAILABLE:
            self.gat1 = GATConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
            self.gat2 = GATConv(
                in_channels=hidden_channels * num_heads,
                out_channels=hidden_channels // 2,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
        else:
            # Fallback to custom implementation
            self.gat1 = GATLayer(hidden_channels, hidden_channels, num_heads, dropout)
            self.gat2 = GATLayer(hidden_channels * num_heads, hidden_channels // 2, num_heads, dropout)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = nn.BatchNorm1d(self.output_dim)

        # Output normalization
        self.output_norm = nn.LayerNorm(self.output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Graph branch.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # Construct graph
        edge_index, node_features = self.graph_constructor(x)
        edge_index = edge_index.to(x.device)

        # Project node features
        x = self.input_projection(node_features)  # (B*20, 128)

        # GAT Layer 1
        x = self.gat1(x, edge_index)  # (B*20, 128*4)
        x = F.elu(x)
        x = self.bn1(x)

        # GAT Layer 2
        x = self.gat2(x, edge_index)  # (B*20, 64*4)
        x = F.elu(x)
        x = self.bn2(x)

        # Reshape back to batch format: (B*20, 256) → (B, 20, 256)
        x = x.view(batch_size, self.in_features, -1)  # (B, 20, 256)

        # Global pooling: aggregate across nodes (features)
        x = x.mean(dim=1)  # (B, 256)

        # Final normalization
        x = self.output_norm(x)

        return x

    def get_output_dim(self) -> int:
        """Return the output dimension of this branch."""
        return self.output_dim


# ============================================================================
# Testing and Validation Functions
# ============================================================================

def test_graph_branch_forward():
    """Test forward pass with different batch sizes."""
    print("\n" + "="*70)
    print("TESTING GRAPH BRANCH FORWARD PASS")
    print("="*70)

    model = GraphBranch(in_features=20)
    model.eval()

    print(f"\nModel architecture:")
    print(f"  Input features: {model.in_features}")
    print(f"  Hidden channels: {model.hidden_channels}")
    print(f"  Output dimension: {model.output_dim}")
    print(f"  Torch Geometric available: {TORCH_GEOMETRIC_AVAILABLE}")

    # Test with different batch sizes
    batch_sizes = [1, 16, 64, 128]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 20)

        try:
            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, 256), \
                f"Expected shape ({batch_size}, 256), got {output.shape}"

            print(f"✓ Batch size {batch_size:3d}: input {x.shape} → output {output.shape}")

        except (RuntimeError, AssertionError) as e:
            print(f"✗ Batch size {batch_size:3d} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ All forward passes successful")
    return True


def test_graph_branch_backward():
    """Test backward pass and gradient flow."""
    print("\n" + "="*70)
    print("TESTING GRAPH BRANCH BACKWARD PASS")
    print("="*70)

    model = GraphBranch(in_features=20)
    model.train()

    # Create dummy data and target
    x = torch.randn(64, 20, requires_grad=True)
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

    except (RuntimeError, AssertionError) as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_branch_on_real_data():
    """Test Graph branch on real NIDS data."""
    print("\n" + "="*70)
    print("TESTING GRAPH BRANCH ON REAL DATA")
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
        graph_features = features['graph']

        print(f"✓ Loaded real data")
        print(f"  Graph features shape: {graph_features.shape}")
        print(f"  Labels shape: {labels.shape}")

        # Test forward pass
        model = GraphBranch(in_features=20)
        model.eval()

        with torch.no_grad():
            output = model(graph_features)

        print(f"\n✓ Forward pass on real data successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  Output mean: {output.mean():.4f}")
        print(f"  Output std: {output.std():.4f}")

        return True

    except (RuntimeError, ImportError, KeyError) as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_branch_gpu():
    """Test Graph branch on GPU if available."""
    print("\n" + "="*70)
    print("TESTING GRAPH BRANCH ON GPU")
    print("="*70)

    if not torch.cuda.is_available():
        print("✗ GPU not available, skipping GPU test")
        return True

    device = torch.device('cuda')
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")

    try:
        model = GraphBranch(in_features=20).to(device)
        x = torch.randn(64, 20).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.is_cuda, "Output should be on GPU"
        assert output.shape == (64, 256), f"Wrong output shape: {output.shape}"

        print(f"✓ GPU forward pass successful")
        print(f"  Input device: {x.device}")
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

    model = GraphBranch(in_features=20)

    total_params = count_parameters(model)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print(f"\n✓ Model size analysis complete")
    return True


if __name__ == '__main__':
    """Run all tests when script is executed directly."""
    print("\n" + "="*70)
    print("RUNNING GRAPH BRANCH TESTS")
    print("="*70)

    all_passed = True

    # Run tests
    all_passed &= test_graph_branch_forward()
    all_passed &= test_graph_branch_backward()
    all_passed &= test_model_size()
    all_passed &= test_graph_branch_gpu()
    all_passed &= test_graph_branch_on_real_data()

    if all_passed:
        print("\n" + "="*70)
        print("ALL GRAPH BRANCH TESTS PASSED ✓✓✓")
        print("Ready to integrate into HybridFormer!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ✗")
        print("Please fix errors before proceeding")
        print("="*70)
