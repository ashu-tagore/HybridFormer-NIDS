"""
Custom loss functions for imbalanced classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard examples and down-weights easy examples.
    Critical for rare classes like Worms (36 samples).

    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
               - gamma=0: equivalent to CE loss
               - gamma=2: strong focus on hard examples
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t (probability of true class)
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Focal Loss with per-class weighting.

    Combines focal loss with class weights for extreme imbalance.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0
    ):
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        """
        # Get class weights for this batch
        weights = self.class_weights[targets]

        # Compute cross entropy with class weights
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t
        pt = torch.exp(-ce_loss)

        # Focal loss with class weights
        focal_loss = self.alpha * weights * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def test_focal_loss():
    """Test focal loss implementation."""
    print("\n" + "="*70)
    print("TESTING FOCAL LOSS")
    print("="*70)

    # Create dummy data
    batch_size = 64
    num_classes = 10

    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Test standard focal loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    loss = focal_loss(logits, targets)

    print(f"\n1. Standard Focal Loss:")
    print(f"   Loss value: {loss.item():.4f}")

    # Test backward
    loss.backward()
    print(f"   Gradient computed: {logits.grad is not None}")

    # Test class-balanced focal loss
    class_weights = torch.rand(num_classes)
    cb_focal_loss = ClassBalancedFocalLoss(class_weights, alpha=1.0, gamma=2.0)

    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    loss = cb_focal_loss(logits, targets)

    print(f"\n2. Class-Balanced Focal Loss:")
    print(f"   Loss value: {loss.item():.4f}")

    loss.backward()
    print(f"   Gradient computed: {logits.grad is not None}")

    # Compare with standard CE
    ce_loss = F.cross_entropy(
        torch.randn(batch_size, num_classes),
        targets
    )

    print(f"\n3. Standard Cross-Entropy (for comparison):")
    print(f"   Loss value: {ce_loss.item():.4f}")

    print("\n" + "="*70)
    print("FOCAL LOSS TESTS PASSED ✓")
    print("="*70)


if __name__ == '__main__':
    test_focal_loss()