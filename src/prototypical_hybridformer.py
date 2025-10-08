"""
Prototypical HybridFormer - HybridFormer with Prototypical Learning.

Combines multi-modal fusion with prototype-based classification for rare classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union

# Handle imports for both module usage and direct testing
try:
    from src.cnn_branch import CNNBranch
    from src.transformer_branch import TransformerBranch
    from src.graph_branch import GraphBranch
except ImportError:
    from cnn_branch import CNNBranch
    from transformer_branch import TransformerBranch
    from graph_branch import GraphBranch


class PrototypicalHybridFormer(nn.Module):
    """
    HybridFormer with Prototypical Learning.

    Key Innovation:
    - Standard classification learns decision boundaries
    - Prototypical learning learns class prototypes in embedding space
    - MUCH better for rare classes (Worms: 36 samples, Analysis: 58 samples)

    Architecture:
        Input → HybridFormer Backbone → Features (1024-dim)
        ↓
        ├─→ Embedding Projection → Prototypes (128-dim)
        │   └─→ Distance-based Classification
        └─→ Traditional Classification Head → Logits

        Final: Weighted combination of both predictions
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
        embedding_dim: int = 128,
        use_dual_head: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_dual_head = use_dual_head

        # Backbone: CNN + Transformer + Graph branches
        self.cnn_branch = CNNBranch(in_features=10)
        self.transformer_branch = TransformerBranch(in_features=12)
        self.graph_branch = GraphBranch(in_features=20)

        # Fused feature dimension
        self.fused_dim = 512 + 256 + 256  # 1024

        # Embedding projection for prototypical learning
        self.embedding_projection = nn.Sequential(
            nn.Linear(self.fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Learnable class prototypes (updated during training)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # Traditional classification head (if using dual head)
        if use_dual_head:
            self.classifier = nn.Sequential(
                nn.Linear(self.fused_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.67),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout * 0.33),

                nn.Linear(128, num_classes)
            )
            self._init_classifier_weights()

        # Learnable fusion weight (how much to trust prototypes vs classifier)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Dict with 'cnn', 'transformer', 'graph' features
            return_embeddings: If True, return (logits, proto_logits, embeddings)

        Returns:
            logits: Final classification logits
            OR
            (logits, proto_logits, embeddings) if return_embeddings=True
        """
        # Extract branch features
        cnn_out = self.cnn_branch(x['cnn'])
        trans_out = self.transformer_branch(x['transformer'])
        graph_out = self.graph_branch(x['graph'])

        # Concatenate
        fused = torch.cat([cnn_out, trans_out, graph_out], dim=1)  # (B, 1024)

        # Project to embedding space
        embeddings = self.embedding_projection(fused)  # (B, 128)

        # L2 normalize embeddings and prototypes
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)

        # Compute distances to prototypes (negative distance = similarity)
        # Lower distance = higher similarity
        distances = torch.cdist(embeddings_norm, prototypes_norm)  # (B, num_classes)
        proto_logits = -distances  # Convert to similarity scores

        # Traditional classification (if dual head)
        if self.use_dual_head:
            class_logits = self.classifier(fused)  # (B, num_classes)

            # Weighted fusion
            alpha = torch.sigmoid(self.fusion_weight)  # Constrain to [0, 1]
            final_logits = alpha * proto_logits + (1 - alpha) * class_logits
        else:
            final_logits = proto_logits

        if return_embeddings:
            return final_logits, proto_logits, embeddings_norm
        else:
            return final_logits

    def get_prototypes(self) -> torch.Tensor:
        """Get normalized prototypes."""
        return F.normalize(self.prototypes, p=2, dim=1)


class PrototypicalLoss(nn.Module):
    """
    Combined loss for prototypical learning.

    Loss = classification_loss + prototype_separation + compactness
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        separation_weight: float = 0.1,
        compactness_weight: float = 0.2
    ):
        super().__init__()
        self.base_criterion = base_criterion  # Focal loss or CE
        self.separation_weight = separation_weight
        self.compactness_weight = compactness_weight

    def forward(
        self,
        final_logits: torch.Tensor,
        proto_logits: torch.Tensor,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            final_logits: Final classification logits
            proto_logits: Prototype-based logits
            embeddings: Normalized embeddings
            prototypes: Normalized prototypes
            targets: Ground truth labels

        Returns:
            total_loss, loss_dict
        """
        # 1. Classification loss (main task)
        class_loss = self.base_criterion(final_logits, targets)

        # 2. Prototype classification loss
        proto_class_loss = self.base_criterion(proto_logits, targets)

        # 3. Prototype separation loss (prototypes should be far apart)
        proto_distances = torch.cdist(prototypes, prototypes)
        # Encourage minimum distance of 2.0 between different prototypes
        mask = 1 - torch.eye(len(prototypes), device=prototypes.device)
        separation_loss = F.relu(2.0 - proto_distances).mul(mask).mean()

        # 4. Compactness loss (samples should be close to their class prototype)
        target_prototypes = prototypes[targets]  # (B, embedding_dim)
        compactness_loss = F.mse_loss(embeddings, target_prototypes)

        # Total loss
        total_loss = (
            class_loss +
            proto_class_loss +
            self.separation_weight * separation_loss +
            self.compactness_weight * compactness_loss
        )

        # Loss breakdown for logging
        loss_dict = {
            'total': total_loss.item(),
            'classification': class_loss.item(),
            'proto_classification': proto_class_loss.item(),
            'separation': separation_loss.item(),
            'compactness': compactness_loss.item()
        }

        return total_loss, loss_dict


# ============================================================================
# Testing
# ============================================================================

def test_prototypical_hybridformer():
    """Test PrototypicalHybridFormer."""
    print("\n" + "="*70)
    print("TESTING PROTOTYPICAL HYBRIDFORMER")
    print("="*70)

    model = PrototypicalHybridFormer(num_classes=10, use_dual_head=True)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")

    # Test forward pass
    batch_size = 64
    x = {
        'cnn': torch.randn(batch_size, 10),
        'transformer': torch.randn(batch_size, 12),
        'graph': torch.randn(batch_size, 20)
    }

    # Test inference mode
    print("\n1. Testing inference mode...")
    with torch.no_grad():
        logits = model(x)
    print(f"   Output shape: {logits.shape}")
    expected_shape = (batch_size, 10)
    assert logits.shape == expected_shape, "Wrong shape: {} vs expected {}".format(
        logits.shape, expected_shape
    )
    print("   ✓ Inference mode works")

    # Test training mode with embeddings
    print("\n2. Testing training mode with embeddings...")
    model.train()
    logits, proto_logits, embeddings = model(x, return_embeddings=True)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Proto logits shape: {proto_logits.shape}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print("   ✓ Training mode works")

    # Test loss computation
    print("\n3. Testing prototypical loss...")
    try:
        from src.losses import FocalLoss
    except ImportError:
        from losses import FocalLoss

    base_criterion = FocalLoss(alpha=1.0, gamma=2.0)
    proto_loss = PrototypicalLoss(base_criterion)

    targets = torch.randint(0, 10, (batch_size,))
    prototypes = model.get_prototypes()

    total_loss, loss_dict = proto_loss(
        logits, proto_logits, embeddings, prototypes, targets
    )

    print(f"   Total loss: {total_loss.item():.4f}")
    print("   Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"     {key}: {value:.4f}")

    # Test backward
    total_loss.backward()
    print("   ✓ Backward pass works")

    # Check prototype gradients
    if model.prototypes.grad is not None:
        print(f"   Prototype grad norm: {model.prototypes.grad.norm().item():.4f}")
        print("   ✓ Prototypes are being updated")

    print("\n" + "="*70)
    print("ALL PROTOTYPICAL HYBRIDFORMER TESTS PASSED ✓")
    print("="*70)


if __name__ == '__main__':
    test_prototypical_hybridformer()
