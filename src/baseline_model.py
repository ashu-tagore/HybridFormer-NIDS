"""
Baseline feedforward neural network for NIDS.

This module implements a simple fully-connected neural network
to establish performance baseline before implementing HybridFormer.
"""

import torch
import torch.nn as nn
from typing import List
import logging

logger = logging.getLogger(__name__)


class BaselineFFN(nn.Module):
    """
    Baseline Feedforward Neural Network for network intrusion detection.

    Simple fully-connected architecture with dropout regularization.
    Serves as baseline to compare against HybridFormer.

    Architecture:
        Input(42) → Linear(42, 256) → ReLU → Dropout
                 → Linear(256, 128) → ReLU → Dropout
                 → Linear(128, 64) → ReLU
                 → Linear(64, 10) → Output(10)

    Args:
        input_dim: Number of input features (default: 42)
        hidden_dims: List of hidden layer dimensions (default: [256, 128, 64])
        num_classes: Number of output classes (default: 10)
        dropout: Dropout rate (default: 0.3)
        activation: Activation function ('relu', 'gelu', 'tanh')

    Example:
        >>> model = BaselineFFN(input_dim=42, num_classes=10)
        >>> x = torch.randn(64, 42)  # Batch of 64 samples
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([64, 10])
    """

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 10,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super(BaselineFFN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        self.layers = self._build_layers()

        # Initialize weights
        self._initialize_weights()

        logger.info(f"Baseline model initialized:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        logger.info(f"  Output dim: {num_classes}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Activation: {activation}")

    def _build_layers(self) -> nn.ModuleList:
        """Build network layers dynamically."""
        layers = nn.ModuleList()

        # Input layer
        prev_dim = self.input_dim

        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (helps with training stability)
            layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self.activation)

            # Dropout (skip on last hidden layer for better final features)
            if i < len(self.hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no activation - CrossEntropyLoss includes softmax)
        layers.append(nn.Linear(prev_dim, self.num_classes))

        return layers

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Pass through all layers sequentially
        for layer in self.layers:
            x = layer(x)

        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings from the last hidden layer.

        Useful for visualization and analysis.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Embeddings of shape (batch_size, last_hidden_dim)
        """
        # Pass through all layers except the final output layer
        for layer in self.layers[:-1]:
            x = layer(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (argmax of logits).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)

        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities
