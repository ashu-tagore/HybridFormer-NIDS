"""
Loss functions for NIDS training.

Includes:
1. FocalLoss - Standard focal loss for class imbalance
2. ClassBalancedFocalLoss - Focal loss with class-specific weighting
3. PrototypicalFocalLoss - Combined focal + prototypical learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    The focal loss focuses training on hard examples by down-weighting
    easy examples. This is particularly useful for imbalanced datasets.

    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the true class.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter for modulating loss (gamma >= 0)
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)

        Returns:
            Computed focal loss
        """
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t: probability of true class
        p_t = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.

    Combines focal loss with class-specific weights computed from class frequencies.
    Useful when you have severe class imbalance and want to give more weight
    to minority classes.

    Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (2019)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            class_weights: Weights for each class, shape (num_classes,)
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            reduction: Reduction method
        """
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class-balanced focal loss.

        Args:
            inputs: Predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)

        Returns:
            Computed class-balanced focal loss
        """
        # Compute cross-entropy with class weights
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            reduction='none'
        )

        # Compute p_t
        p_t = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PrototypicalFocalLoss(nn.Module):
    """
    Combined Focal Loss + Prototypical Loss for enhanced minority class learning.

    This loss combines three components:
    1. Focal Loss on classifier outputs - handles class imbalance
    2. Prototypical Loss on proto_logits - learns discriminative features
    3. Separation Loss on prototypes - ensures prototype distinctiveness

    The prototypical component encourages the model to learn embeddings where
    samples cluster around class prototypes, improving minority class detection.

    Architecture requirements:
    - Model must have dual heads (classifier + prototypical)
    - Model must return: (outputs, proto_logits, embeddings)
    - Model must have get_prototypes() method

    Note: The loss only needs outputs, proto_logits, and prototypes.
    Embeddings are computed internally by the model to generate proto_logits.
    """

    def __init__(
        self,
        num_classes: int = 10,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        proto_weight: float = 0.3,
        separation_weight: float = 0.1,
        temperature: float = 0.1
    ):
        """
        Args:
            num_classes: Number of classes
            focal_alpha: Focal loss alpha (class weighting)
            focal_gamma: Focal loss gamma (focusing parameter)
            proto_weight: Weight for prototypical loss component
            separation_weight: Weight for prototype separation loss
            temperature: Temperature for prototypical distance scaling
        """
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.proto_weight = proto_weight
        self.separation_weight = separation_weight
        self.temperature = temperature

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def prototypical_loss(
        self,
        proto_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prototypical loss from prototype-based logits.

        The model computes proto_logits from embedding distances to prototypes,
        so we just need to apply cross-entropy loss on these logits.

        Args:
            proto_logits: Logits from prototypical head [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Prototypical loss value
        """
        # Cross-entropy on prototype logits (already computed by model)
        proto_loss = F.cross_entropy(proto_logits, targets)
        return proto_loss

    def separation_loss(self, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Encourage prototypes to be well-separated from each other.

        Uses reciprocal of distances to penalize prototypes that are too close.
        When prototypes are far apart, penalty is small. When close, penalty is large.

        Args:
            prototypes: Class prototypes [num_classes, embedding_dim]

        Returns:
            Separation loss (penalty for close prototypes)
        """
        # Compute pairwise squared distances between prototypes
        proto_expanded_1 = prototypes.unsqueeze(0)  # [1, C, D]
        proto_expanded_2 = prototypes.unsqueeze(1)  # [C, 1, D]

        pairwise_distances = torch.sum(
            (proto_expanded_1 - proto_expanded_2) ** 2,
            dim=2
        )  # [C, C]

        # Create mask to exclude diagonal (distance to self)
        mask = ~torch.eye(
            self.num_classes,
            device=prototypes.device,
            dtype=torch.bool
        )

        # Get off-diagonal distances
        off_diag_distances = pairwise_distances[mask]

        # Use reciprocal of distances as penalty (add small epsilon for stability)
        # When prototypes are close (small distance), penalty is high
        # When prototypes are far (large distance), penalty is low
        epsilon = 1e-6
        separation = torch.mean(1.0 / (off_diag_distances + epsilon))

        return separation

    def forward(
        self,
        outputs: torch.Tensor,
        proto_logits: torch.Tensor,
        prototypes: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            outputs: Classifier outputs [batch_size, num_classes]
            proto_logits: Prototypical head outputs [batch_size, num_classes]
            prototypes: Class prototypes [num_classes, embedding_dim]
            targets: Ground truth labels [batch_size]

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components for logging
        """
        # 1. Focal loss on main classifier outputs
        focal = self.focal_loss(outputs, targets)

        # 2. Prototypical loss on proto_logits (computed by model)
        proto = self.prototypical_loss(proto_logits, targets)

        # 3. Separation loss on prototypes
        separation = self.separation_loss(prototypes)

        # 4. Combined weighted loss
        total_loss = (
            focal +
            self.proto_weight * proto +
            self.separation_weight * separation
        )

        # Return both total loss and components for logging
        loss_dict = {
            'total_loss': total_loss,
            'ce_loss': focal,
            'proto_loss': proto,
            'separation_loss': separation
        }

        return total_loss, loss_dict


# ============================================================================
# Utility Functions
# ============================================================================

def compute_class_weights(
    class_counts: torch.Tensor,
    mode: str = 'balanced'
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.

    Args:
        class_counts: Number of samples per class, shape (num_classes,)
        mode: Weighting scheme - 'balanced', 'sqrt', or 'log'
            - 'balanced': weight = total_samples / (num_classes * class_count)
            - 'sqrt': weight = sqrt(total_samples / class_count)
            - 'log': weight = log(total_samples / class_count)

    Returns:
        Class weights tensor of shape (num_classes,)
    """
    num_classes = len(class_counts)
    total_samples = class_counts.sum()

    if mode == 'balanced':
        # Inverse frequency weighting
        weights = total_samples / (num_classes * class_counts)
    elif mode == 'sqrt':
        # Square root of inverse frequency (less aggressive)
        weights = torch.sqrt(total_samples / class_counts)
    elif mode == 'log':
        # Log of inverse frequency (even less aggressive)
        weights = torch.log(total_samples / class_counts + 1.0)
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")

    # Normalize weights to have mean = 1.0
    weights = weights / weights.mean()

    return weights


# ============================================================================
# Testing Functions
# ============================================================================

def test_focal_loss():
    """Test FocalLoss implementation."""
    print("\n" + "="*70)
    print("TESTING FOCAL LOSS")
    print("="*70)

    batch_size = 64
    num_classes = 10

    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Test different gamma values
    for gamma in [0.0, 1.0, 2.0, 5.0]:
        loss_fn = FocalLoss(alpha=1.0, gamma=gamma)
        loss = loss_fn(logits, targets)

        print(f"Gamma={gamma:.1f}: Loss={loss.item():.4f}")

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    print("✓ FocalLoss test passed")
    return True


def test_class_balanced_focal_loss():
    """Test ClassBalancedFocalLoss implementation."""
    print("\n" + "="*70)
    print("TESTING CLASS-BALANCED FOCAL LOSS")
    print("="*70)

    batch_size = 64
    num_classes = 10

    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Create class weights (simulate imbalanced dataset)
    class_counts = torch.tensor([1000, 500, 200, 100, 50, 25, 12, 6, 3, 1], dtype=torch.float)
    class_weights = compute_class_weights(class_counts, mode='sqrt')

    print(f"Class weights: {class_weights.tolist()}")

    # Test loss
    loss_fn = ClassBalancedFocalLoss(class_weights=class_weights, gamma=2.0)
    loss = loss_fn(logits, targets)

    print(f"Loss: {loss.item():.4f}")

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("✓ ClassBalancedFocalLoss test passed")
    return True


def test_prototypical_focal_loss():
    """Test PrototypicalFocalLoss implementation."""
    print("\n" + "="*70)
    print("TESTING PROTOTYPICAL FOCAL LOSS")
    print("="*70)

    batch_size = 64
    num_classes = 10
    embedding_dim = 128

    # Create dummy data with gradient tracking enabled
    outputs = torch.randn(batch_size, num_classes, requires_grad=True)
    proto_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    prototypes = torch.randn(num_classes, embedding_dim, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Test loss
    loss_fn = PrototypicalFocalLoss(
        num_classes=num_classes,
        focal_gamma=2.0,
        proto_weight=0.3,
        separation_weight=0.1
    )

    total_loss, loss_dict = loss_fn(
        outputs, proto_logits, prototypes, targets
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"  CE loss: {loss_dict['ce_loss'].item():.4f}")
    print(f"  Proto loss: {loss_dict['proto_loss'].item():.4f}")
    print(f"  Separation loss: {loss_dict['separation_loss'].item():.4f}")

    # Validate loss components
    assert total_loss.item() > 0, "Total loss should be positive"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"

    for key, value in loss_dict.items():
        assert not torch.isnan(value), f"{key} should not be NaN"

    # Test backward pass
    total_loss.backward()

    # Verify gradients were computed
    assert outputs.grad is not None, "Outputs should have gradients"
    assert proto_logits.grad is not None, "Proto logits should have gradients"
    assert prototypes.grad is not None, "Prototypes should have gradients"

    print("✓ Backward pass successful")
    print("✓ All gradients computed")

    print("✓ PrototypicalFocalLoss test passed")
    return True


def test_compute_class_weights():
    """Test class weight computation."""
    print("\n" + "="*70)
    print("TESTING CLASS WEIGHT COMPUTATION")
    print("="*70)

    # Simulate imbalanced dataset
    class_counts = torch.tensor([1000, 500, 200, 100, 50, 25, 12, 6, 3, 1], dtype=torch.float)

    print(f"Class counts: {class_counts.tolist()}")

    for mode in ['balanced', 'sqrt', 'log']:
        weights = compute_class_weights(class_counts, mode=mode)
        print(f"\n{mode.capitalize()} weights:")
        print(f"  {weights.tolist()}")
        print(f"  Mean: {weights.mean():.4f} (should be ~1.0)")
        print(f"  Range: [{weights.min():.4f}, {weights.max():.4f}]")

        # Verify properties
        assert torch.all(weights > 0), "All weights should be positive"
        assert abs(weights.mean().item() - 1.0) < 0.01, "Mean should be close to 1.0"

    print("\n✓ Class weight computation test passed")
    return True


if __name__ == '__main__':
    """Run all tests when script is executed directly."""
    print("\n" + "="*70)
    print("RUNNING LOSS FUNCTION TESTS")
    print("="*70)

    all_passed = True

    all_passed &= test_focal_loss()
    all_passed &= test_class_balanced_focal_loss()
    all_passed &= test_prototypical_focal_loss()
    all_passed &= test_compute_class_weights()

    if all_passed:
        print("\n" + "="*70)
        print("ALL LOSS TESTS PASSED ✓✓✓")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ✗")
        print("="*70)
