"""
Main training script for HybridFormer NIDS model.
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.hybridformer import HybridFormer
from src.dataset import create_dataloaders
from src.trainer import Trainer
from src.utils import set_seed
import src.metrics as metrics_module


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create HybridFormer model from config."""
    model = HybridFormer(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    return model


def create_optimizer(model, config):
    """Create optimizer from config."""
    optimizer_config = config['training']['optimizer']

    if optimizer_config['type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            betas=tuple(optimizer_config['betas'])
        )
    elif optimizer_config['type'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")

    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler from config."""
    scheduler_config = config['training']['scheduler']

    if scheduler_config['type'].lower() == 'cosine':
        warmup_epochs = scheduler_config['warmup_epochs']
        total_epochs = config['training']['epochs']

        # Warmup + cosine annealing
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    return scheduler


def create_criterion(config):
    """Create loss function from config."""
    loss_config = config['training']['loss']

    if loss_config['type'].lower() == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_config['type']}")

    return criterion


def main(config_args):
    """Main training function."""
    # Load configuration
    config = load_config(config_args.config)
    print(f"Loaded config from {config_args.config}")
    print(f"Experiment: {config['experiment']['name']}")

    # Set random seed for reproducibility
    set_seed(config['training']['seed'])

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        mode=config['data']['mode'],
        branch_allocation_path=config['data']['branch_allocation']
    )

    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    # Create model
    print("\nCreating model...")
    model = create_model(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Create optimizer, scheduler, and criterion
    optimizer = create_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)
    criterion = create_criterion(config)

    print(f"Optimizer: {config['training']['optimizer']['type']}")
    print(f"Learning rate: {config['training']['optimizer']['lr']}")
    print(f"Scheduler: {config['training']['scheduler']['type']}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config['training']['device'],
        config=config,
        scheduler=scheduler
    )

    # Train model
    print("\nStarting training...")
    print("="*70)
    trainer.train(epochs=config['training']['epochs'])

    # Evaluate on test set
    print("\n" + "="*70)
    print("Evaluating on test set...")
    print("="*70)

    # Load best model
    best_model_path = Path(config['logging']['save_dir']) / 'best_model.pth'
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    device = torch.device(config['training']['device'])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch

            # Move to device
            if isinstance(features, dict):
                features = {k: v.to(device) for k, v in features.items()}
            else:
                features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)

            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    test_metrics = metrics_module.compute_metrics(all_preds, all_labels)

    print("\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1: {test_metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1: {test_metrics['weighted_f1']*100:.2f}%")

    print("\nPer-Class F1 Scores:")
    class_names = config['evaluation']['class_names']
    for class_idx, f1_score in enumerate(test_metrics['per_class_f1']):
        class_name = class_names[class_idx]
        print(f"  {class_name}: {f1_score*100:.2f}%")

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HybridFormer NIDS model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/hybridformer.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()
    main(args)
