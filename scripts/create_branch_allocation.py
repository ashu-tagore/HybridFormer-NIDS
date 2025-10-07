"""
Create branch feature allocation for HybridFormer.
Intelligently assigns 42 features to CNN, Transformer, and Graph branches.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

def analyze_features():
    """Load and analyze feature characteristics."""
    print("Loading validation data for feature analysis...")

    data_dir = Path('data/processed')

    with open(data_dir / 'val_features.pkl', 'rb') as f:
        features = pickle.load(f)

    if isinstance(features, pd.DataFrame):
        feature_names = features.columns.tolist()
        features = features.values
    else:
        # Create generic names if no column names
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]

    print(f"Loaded {len(feature_names)} features")

    return features, feature_names


def allocate_features_intelligent(features, feature_names):
    """
    Intelligently allocate features to branches based on characteristics.

    CNN Branch (10 features): Local patterns, statistical features
    - Features with high variance (capture different patterns)
    - Features that represent packet sizes, rates, etc.

    Transformer Branch (12 features): Sequential/temporal dependencies
    - Features with temporal characteristics
    - Flow-based features

    Graph Branch (20 features): Network topology, relationships
    - Features related to network connections
    - Protocol and service features
    - Features with strong correlations to others
    """

    num_features = len(feature_names)
    assert num_features == 42, f"Expected 42 features, got {num_features}"

    # Compute feature statistics
    variances = np.var(features, axis=0)
    means = np.mean(features, axis=0)

    # Compute pairwise correlations (for graph features)
    correlations = np.corrcoef(features.T)
    avg_correlations = np.abs(correlations).mean(axis=1)

    # Score each feature for each branch
    cnn_scores = variances / (variances.max() + 1e-8)  # High variance
    transformer_scores = 1.0 - (means / (means.max() + 1e-8))  # Balanced distribution
    graph_scores = avg_correlations  # High correlation with others

    # Create allocation
    allocated = set()
    allocation = {
        'cnn': [],
        'transformer': [],
        'graph': []
    }

    # Allocate to CNN (10 features with highest CNN scores)
    cnn_candidates = [(i, cnn_scores[i]) for i in range(num_features)]
    cnn_candidates.sort(key=lambda x: x[1], reverse=True)
    for i, score in cnn_candidates[:10]:
        allocation['cnn'].append(i)
        allocated.add(i)

    # Allocate to Transformer (12 features with highest transformer scores)
    trans_candidates = [(i, transformer_scores[i]) for i in range(num_features) if i not in allocated]
    trans_candidates.sort(key=lambda x: x[1], reverse=True)
    for i, score in trans_candidates[:12]:
        allocation['transformer'].append(i)
        allocated.add(i)

    # Allocate remaining to Graph (20 features)
    remaining = [i for i in range(num_features) if i not in allocated]
    allocation['graph'] = remaining

    # Verify allocation
    assert len(allocation['cnn']) == 10, f"CNN should have 10 features, got {len(allocation['cnn'])}"
    assert len(allocation['transformer']) == 12, f"Transformer should have 12 features, got {len(allocation['transformer'])}"
    assert len(allocation['graph']) == 20, f"Graph should have 20 features, got {len(allocation['graph'])}"

    total_allocated = len(allocation['cnn']) + len(allocation['transformer']) + len(allocation['graph'])
    assert total_allocated == 42, f"Should allocate 42 features, got {total_allocated}"

    return allocation


def allocate_features_simple():
    """
    Simple allocation strategy: just split features sequentially.
    Use this if intelligent allocation has issues.
    """
    allocation = {
        'cnn': list(range(0, 10)),           # Features 0-9
        'transformer': list(range(10, 22)),   # Features 10-21
        'graph': list(range(22, 42))          # Features 22-41
    }
    return allocation


def create_allocation_file(strategy='intelligent'):
    """Create the branch_feature_allocation.json file."""

    print("\n" + "="*70)
    print("CREATING BRANCH FEATURE ALLOCATION")
    print("="*70)

    if strategy == 'intelligent':
        print("\nUsing INTELLIGENT allocation strategy...")
        try:
            features, feature_names = analyze_features()
            allocation = allocate_features_intelligent(features, feature_names)
            print("✓ Intelligent allocation successful")
        except Exception as e:
            print(f"✗ Intelligent allocation failed: {e}")
            print("Falling back to simple allocation...")
            allocation = allocate_features_simple()
    else:
        print("\nUsing SIMPLE allocation strategy...")
        allocation = allocate_features_simple()

    # Create the full structure
    allocation_data = {
        'cnn_branch': {
            'num_features': len(allocation['cnn']),
            'feature_indices': allocation['cnn'],
            'description': 'Local patterns, statistical features'
        },
        'transformer_branch': {
            'num_features': len(allocation['transformer']),
            'feature_indices': allocation['transformer'],
            'description': 'Sequential/temporal dependencies'
        },
        'graph_branch': {
            'num_features': len(allocation['graph']),
            'feature_indices': allocation['graph'],
            'description': 'Network topology, relationships'
        },
        'metadata': {
            'total_features': 42,
            'allocation_strategy': strategy,
            'creation_date': pd.Timestamp.now().isoformat()
        }
    }

    # Save to file
    output_path = Path('data/processed/branch_feature_allocation.json')
    with open(output_path, 'w') as f:
        json.dump(allocation_data, f, indent=2)

    print(f"\n✓ Saved allocation to {output_path}")

    # Print summary
    print("\nAllocation Summary:")
    print(f"  CNN Branch: {len(allocation['cnn'])} features")
    print(f"    Indices: {allocation['cnn']}")
    print(f"  Transformer Branch: {len(allocation['transformer'])} features")
    print(f"    Indices: {allocation['transformer']}")
    print(f"  Graph Branch: {len(allocation['graph'])} features")
    print(f"    Indices: {allocation['graph']}")

    print("\n" + "="*70)
    print("ALLOCATION FILE CREATED SUCCESSFULLY ✓")
    print("="*70)

    return allocation_data


if __name__ == '__main__':
    # Try intelligent allocation, fall back to simple if it fails
    create_allocation_file(strategy='intelligent')
