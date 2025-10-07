"""
Training script for NIDS Baseline Model

Usage:
    python train.py
    python train.py --config configs/baseline.yaml
    python train.py --epochs 30 --batch-size 128

This script trains the baseline feedforward neural network
and saves the best model based on validation F1 score.
"""

import argparse
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

from src import (
    Config,
    get_data_loaders,
    compute_class_weights,
    BaselineFFN,
    Trainer,
    set_seed,
    get_device,
    print_model_summary
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Baseline NIDS Model')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/baseline.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (overrides config)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    return parser.parse_args()


def main():
    """Main training function."""

    # Print header
    print("\n" + "="*70)
    print(" "*20 + "BASELINE MODEL TRAINING")
    print("="*70)
    print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Parse arguments
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = Config(args.config)
    print("✓ Configuration loaded\n")

    # Override config with command line arguments
    epochs = args.epochs if args.epochs else config.training.epochs
    batch_size = args.batch_size if args.batch_size else config.training.batch_size
    device = args.device if args.device else config.get('device', 'cuda')
    learning_rate = args.lr if args.lr else config.training.learning_rate

    print("Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"  Early stopping patience: {config.training.early_stopping_patience}\n")

    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"✓ Random seed set to {seed}\n")

    # Create data loaders
    print("="*70)
    print("LOADING DATA")
    print("="*70)

    loaders = get_data_loaders(
        train_features_path=config.data.train_features,
        train_labels_path=config.data.train_labels,
        val_features_path=config.data.val_features,
        val_labels_path=config.data.val_labels,
        test_features_path=config.data.test_features,
        test_labels_path=config.data.test_labels,
        batch_size=batch_size,
        num_workers=config.get('dataloader.num_workers', 4),
        pin_memory=config.get('dataloader.pin_memory', True)
    )

    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    train_dataset = loaders['train_dataset']

    print(f"\n✓ Data loaders created successfully")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Training batches: {len(train_loader):,}")
    print(f"  Validation batches: {len(val_loader):,}")
    print(f"  Test batches: {len(test_loader):,}\n")

    # Compute class weights
    print("Computing class weights for imbalanced data...")
    class_weights = None
    if config.get('training.class_weights', True):
        class_weights = compute_class_weights(
            train_dataset.labels,
            num_classes=config.model.num_classes,
            method=config.get('training.class_weight_method', 'balanced')
        )
        print("✓ Class weights computed\n")

    # Create model
    print("="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    model = BaselineFFN(
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        activation=config.get('model.activation', 'relu')
    )

    print_model_summary(model)

    # Setup training components
    print("Setting up training components...")

    # Get device
    device_obj = get_device(prefer_cuda=(device == 'cuda'))

    # Loss function
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device_obj),
            label_smoothing=config.get('loss.label_smoothing', 0.0)
        )
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('loss.label_smoothing', 0.0)
        )

    print(f"✓ Loss function: CrossEntropyLoss")

    # Optimizer
    optimizer_name = config.get('optimizer.name', 'adam').lower()
    weight_decay = config.get('optimizer.weight_decay', 0.0001)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get('optimizer.betas', [0.9, 0.999]),
            eps=config.get('optimizer.eps', 1e-8)
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"✓ Optimizer: {optimizer_name.upper()}")

    # Learning rate scheduler
    scheduler = None
    scheduler_name = config.get('scheduler.name', 'reduce_on_plateau')

    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('scheduler.mode', 'min'),
            factor=config.get('scheduler.factor', 0.5),
            patience=config.get('scheduler.patience', 5),
            min_lr=config.get('scheduler.min_lr', 1e-6)
        )
        print(f"✓ Scheduler: ReduceLROnPlateau\n")

    # Create trainer
    trainer_config = {
        'num_classes': config.model.num_classes,
        'early_stopping_patience': config.training.early_stopping_patience,
        'use_tensorboard': config.get('logging.tensorboard', True),
        'tensorboard_dir': config.get('logging.tensorboard_dir', 'runs/baseline'),
        'save_dir': config.get('logging.save_dir', 'saved_models')
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=trainer_config
    )

    print("✓ Trainer initialized\n")

    # Start training
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"\nTraining for maximum {epochs} epochs...")
    print(f"Progress will be displayed below.\n")

    try:
        history = trainer.train(epochs=epochs)

        # Training complete
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)

        print(f"\nBest Model:")
        print(f"  Validation F1 Score: {trainer.best_val_f1:.4f}")
        print(f"  Achieved at Epoch: {trainer.best_epoch + 1}")

        print(f"\nFinal Metrics:")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Train Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"  Val Accuracy: {history['val_acc'][-1]:.4f}")
        print(f"  Val F1 (Macro): {history['val_f1_macro'][-1]:.4f}")

        # Save final model
        print(f"\nSaving final model...")
        trainer.save_final_model()

        print(f"\nModel saved to: {trainer.save_dir}")
        print(f"  Best model: baseline_best.pth")
        print(f"  Final model: baseline_final.pth")

        print(f"\nTensorBoard logs: {trainer_config['tensorboard_dir']}")
        print(f"To view training progress, run:")
        print(f"  tensorboard --logdir {trainer_config['tensorboard_dir']}")

        print("\n" + "="*70)
        print("SUCCESS! Training completed successfully.")
        print("="*70)
        print(f"Training ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model state...")
        trainer.save_final_model()
        print("Model saved. You can resume training later.")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        print("Check logs for details.")
        raise


if __name__ == '__main__':
    main()
