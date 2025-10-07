"""
Trainer class for NIDS models.
Handles both baseline (flat features) and HybridFormer (dict features).
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from tqdm import tqdm
import time

from src.metrics import compute_metrics
from src.utils import EarlyStopping, save_checkpoint

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for NIDS models.

    Supports both:
    - Baseline models with flat features (batch_size, num_features)
    - HybridFormer with dict features {'cnn': ..., 'transformer': ..., 'graph': ...}
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device: str = 'cuda',
        config: dict = None,
        scheduler=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config or {}

        # Device setup
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        logger.info(f"Using {self.device} device" +
                   (f": {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

        # Logging setup
        log_dir = self.config.get('logging', {}).get('log_dir', 'runs/baseline')
        self.writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logging to {log_dir}")

        # Save directory
        self.save_dir = Path(self.config.get('logging', {}).get('save_dir', 'saved_models'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        early_stop_config = self.config.get('training', {}).get('early_stopping', {})
        if early_stop_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get('patience', 10),
                mode=early_stop_config.get('mode', 'max')
            )
        else:
            self.early_stopping = None

        # Gradient clipping
        grad_clip_config = self.config.get('training', {}).get('grad_clip', {})
        self.grad_clip_enabled = grad_clip_config.get('enabled', False)
        self.grad_clip_max_norm = grad_clip_config.get('max_norm', 1.0)

        # Tracking
        self.current_epoch = 0
        self.best_val_metric = float('-inf')
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

        logger.info("Trainer initialized successfully")

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (features, labels) in enumerate(pbar):
            # Move to device - handle both flat and dict features
            if isinstance(features, dict):
                features = {k: v.to(self.device) for k, v in features.items()}
            else:
                features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip_enabled:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_max_norm
                )

            # Optimizer step
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to tensorboard
            log_interval = self.config.get('logging', {}).get('log_interval', 100)
            if batch_idx % log_interval == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)

        # Compute epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)

        return {
            'loss': epoch_loss,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1']
        }

    def validate(self) -> dict:
        """Validate on validation set."""
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validating"):
                # Move to device - handle both flat and dict features
                if isinstance(features, dict):
                    features = {k: v.to(self.device) for k, v in features.items()}
                else:
                    features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # Track metrics
                running_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_preds.append(predictions.cpu())
                all_labels.append(labels.cpu())

        # Compute validation metrics
        val_loss = running_loss / len(self.val_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)

        return {
            'loss': val_loss,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'per_class_f1': metrics['per_class_f1'],
            'confusion_matrix': metrics['confusion_matrix']
        }

    def train(self, epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Learning rate step
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Train/LearningRate', current_lr, epoch)

            # Epoch time
            epoch_time = time.time() - epoch_start_time

            # Log metrics
            self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Train/MacroF1', train_metrics['macro_f1'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/MacroF1', val_metrics['macro_f1'], epoch)

            # Print epoch summary
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                f"Val Macro F1: {val_metrics['macro_f1']*100:.2f}%"
            )

            # Print per-class F1 scores
            class_names = self.config.get('evaluation', {}).get('class_names', {})
            if class_names:
                logger.info("Per-class F1 scores:")
                for class_idx, f1_score in enumerate(val_metrics['per_class_f1']):
                    class_name = class_names.get(class_idx, f"Class {class_idx}")
                    logger.info(f"  {class_name}: {f1_score*100:.2f}%")

            # Save checkpoint
            metric_for_best = val_metrics.get('macro_f1', val_metrics['accuracy'])

            if metric_for_best > self.best_val_metric:
                self.best_val_metric = metric_for_best
                logger.info(f"New best model! Macro F1: {metric_for_best*100:.2f}%")

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    path=self.save_dir / 'best_model.pth'
                )

            # Save last model
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=val_metrics,
                path=self.save_dir / 'last_model.pth'
            )

            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(metric_for_best)
                if self.early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        logger.info("Training completed!")
        logger.info(f"Best validation macro F1: {self.best_val_metric*100:.2f}%")

        self.writer.close()
