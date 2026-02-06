"""
Ultra-simple Phase 1 runner - Just loads and analyzes data
"""
import sys
import os
import numpy as np
import pandas as pd
# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    print("=" * 70)
    print("PHASE 1: ULTRA-SIMPLE VERSION")
    print("=" * 70)
    # Import just what we need
    from phase1_exploration.data_loader import DataLoader
    # Initialize loader
    loader = DataLoader(data_dir='../data')
    print("\n1. Loading datasets...")
    # Try to load from Phase 0
    if os.path.exists('../data/synthetic'):
        datasets = loader.load_from_directory('../data/synthetic', pattern="*dataset*.csv")
        print(f"   ✓ Loaded {len(datasets)} datasets from Phase 0")
    else:
        # Load some sklearn datasets
        loader.load_from_sklearn('digits')
        loader.load_from_sklearn('iris')
        print("   ✓ Loaded sklearn datasets (digits, iris)")
    # Show what we loaded
    print("\n2. Dataset Summary:")
    print("-" * 40)
    for name in loader.list_datasets():
        data = loader.get_dataset(name)
        print(f"\n{name.upper()}:")
        print(f"  • Samples: {data['n_samples']:,}")
        print(f"  • Features: {data['n_features']}")
        print(f"  • Classes: {data['n_classes']}")
        if 'label_stats' in data:
            print(f"  • Imbalance ratio: {data['label_stats']['imbalance_ratio']:.2f}")
            print(f"  • Balanced: {'Yes' if data['label_stats']['is_balanced'] else 'No'}")
    # Save for Phase 2
    print("\n3. Saving for Phase 2...")
    os.makedirs('../data/phase2_ready', exist_ok=True)
    for name in loader.list_datasets():
        data = loader.get_dataset(name)
        # Create DataFrame
        X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
        df = X_df.copy()
        df['target'] = data['y']
        # Save
        filename = f'../data/phase2_ready/{name}_cleaned.csv'
        df.to_csv(filename, index=False)
        print(f"  ✓ {name}: {len(df)} samples → {filename}")
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE!")
    print("=" * 70)
    print("\nOutput:")
    print("  • Clean datasets in: data/phase2_ready/")
    print("  • Ready for Phase 2: Noise Detection")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print(f"\nCurrent directory: {os.getcwd()}")
