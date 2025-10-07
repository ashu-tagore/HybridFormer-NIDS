"""
Training loop orchestrator for NIDS project.

This module provides the main Trainer class that handles:
- Training loop with validation
- Early stopping
- Checkpoint saving
- TensorBoard logging
- Progress tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import logging

from .metrics import MetricsCalculator
from .utils import save_checkpoint, EarlyStopping, get_device

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main training orchestrator for NIDS models.

    Handles complete training workflow including:
    - Training and validation loops
    - Metric computation
    - Early stopping
    - Model checkpointing
    - TensorBoard logging

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        device: Device to train on ('cuda' or 'cpu')
        config: Configuration dictionary

    Example:
        >>> trainer = Trainer(
        ...     model=my_model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=nn.CrossEntropyLoss(),
        ...     optimizer=optimizer,
        ...     device='cuda',
        ...     config=config
        ... )
        >>> history = trainer.train(epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = get_device(prefer_cuda=(device == 'cuda'))
        self.config = config or {}

        # Move model to device
        self.model.to(self.device)

        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator(
            num_classes=self.config.get('num_classes', 10)
        )

        # Setup early stopping
        patience = self.config.get('early_stopping_patience', 10)
        self.early_stopping = EarlyStopping(
            patience=patience,
            mode='max'  # Maximize validation F1
        )

        # Setup TensorBoard
        self.use_tensorboard = self.config.get('use_tensorboard', True)
        if self.use_tensorboard:
            log_dir = self.config.get('tensorboard_dir', 'runs/baseline')
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1_macro': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0

        # Save directory
        self.save_dir = Path(self.config.get('save_dir', 'saved_models'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Trainer initialized successfully")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1} [Train]",
            leave=False
        )

        for batch_idx, (features, labels) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device).squeeze()

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(features)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping (helps with stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log batch metrics to TensorBoard
            if self.use_tensorboard and batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Batch/train_loss', loss.item(), global_step)

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics_calc.compute_metrics(all_labels, all_predictions)

        return {
            'loss': avg_loss,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro']
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1} [Val]  ",
            leave=False
        )

        with torch.no_grad():
            for features, labels in pbar:
                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device).squeeze()

                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_calc.compute_metrics(all_labels, all_predictions)

        return {
            'loss': avg_loss,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'all_metrics': metrics
        }

    def train(self, epochs: int) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            epochs: Number of epochs to train

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])
            self.history['learning_rates'].append(current_lr)

            # Log to TensorBoard
            if self.use_tensorboard:
                self.writer.add_scalar('Epoch/train_loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/val_loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/train_acc', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Epoch/val_acc', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Epoch/val_f1_macro', val_metrics['f1_macro'], epoch)
                self.writer.add_scalar('Epoch/learning_rate', current_lr, epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_epoch = epoch

                self._save_best_model(epoch, val_metrics)
                print(f"  ✓ New best model! F1: {self.best_val_f1:.4f}")

            # Early stopping check
            if self.early_stopping(val_metrics['f1_macro']):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch+1}")
                break

        # Training complete
        logger.info("Training complete!")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch+1}")

        # Close TensorBoard writer
        if self.use_tensorboard:
            self.writer.close()

        return self.history

    def _save_best_model(self, epoch: int, metrics: Dict[str, float]):
        """Save the best model checkpoint."""
        save_path = self.save_dir / 'baseline_best.pth'

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            path=save_path,
            scheduler=self.scheduler,
            extra_state={'best_val_f1': self.best_val_f1}
        )

    def save_final_model(self):
        """Save final model state."""
        save_path = self.save_dir / 'baseline_final.pth'

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=len(self.history['train_loss']),
            metrics={'val_f1_macro': self.history['val_f1_macro'][-1]},
            path=save_path,
            scheduler=self.scheduler
        )

        logger.info(f"Final model saved to {save_path}")
