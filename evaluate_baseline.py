"""
Evaluate the trained baseline model and show detailed metrics.
"""

import torch
import numpy as np
from src import Config, get_data_loaders, BaselineFFN, MetricsCalculator, get_device

print("="*70)
print("BASELINE MODEL EVALUATION")
print("="*70)

# Load config
config = Config('configs/baseline.yaml')

# Load data
print("\nLoading validation data...")
loaders = get_data_loaders(
    train_features_path=config.data.train_features,
    train_labels_path=config.data.train_labels,
    val_features_path=config.data.val_features,
    val_labels_path=config.data.val_labels,
    test_features_path=config.data.test_features,
    test_labels_path=config.data.test_labels,
    batch_size=64,
    num_workers=0
)

val_loader = loaders['val']
print(f"✓ Loaded {len(loaders['val_dataset']):,} validation samples")

# Load model
print("\nLoading trained model...")
device = get_device()
model = BaselineFFN(
    input_dim=42,
    hidden_dims=[256, 128, 64],
    num_classes=10
)

# Load checkpoint
checkpoint = torch.load('saved_models/baseline_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("✓ Model loaded")

# Evaluate
print("\nEvaluating model...")
all_predictions = []
all_labels = []

with torch.no_grad():
    for features, labels in val_loader:
        features = features.to(device)
        labels = labels.to(device).squeeze()

        logits = model(features)
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate metrics
metrics_calc = MetricsCalculator(num_classes=10)
metrics = metrics_calc.compute_metrics(all_labels, all_predictions)

print("\n" + "="*70)
print("DETAILED RESULTS")
print("="*70)

# Overall metrics
print(f"\nOverall Performance:")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  F1 (Macro): {metrics['f1_macro']:.4f}")

# Per-class performance
print(f"\nPer-Class Performance:")
print("-"*70)
print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
print("-"*70)

class_names = {
    0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS",
    4: "Exploits", 5: "Fuzzers", 6: "Generic",
    7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}

for i in range(10):
    name = class_names[i]
    prec = metrics[f'precision_{name}']
    rec = metrics[f'recall_{name}']
    f1 = metrics[f'f1_{name}']
    supp = metrics[f'support_{name}']
    print(f"{name:<20} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {supp:<10}")

# Prediction distribution
print(f"\n" + "="*70)
print("PREDICTION DISTRIBUTION")
print("="*70)
unique, counts = np.unique(all_predictions, return_counts=True)
print(f"\nWhat the model predicted:")
for cls, count in zip(unique, counts):
    percentage = (count / len(all_predictions)) * 100
    print(f"  {class_names[cls]:<20}: {count:>6,} ({percentage:>5.2f}%)")

# Actual distribution
print(f"\nActual distribution:")
unique, counts = np.unique(all_labels, return_counts=True)
for cls, count in zip(unique, counts):
    percentage = (count / len(all_labels)) * 100
    print(f"  {class_names[cls]:<20}: {count:>6,} ({percentage:>5.2f}%)")

print("\n" + "="*70)
