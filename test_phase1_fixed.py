"""
Simple test to verify Phase 1 modules work
"""
import sys
import os
# Add src to path
sys.path.append('src')
try:
    # Test DataLoader
    print("Testing DataLoader...")
    from phase1_exploration.data_loader import DataLoader
    loader = DataLoader()
    # Test loading sklearn dataset
    print("  Loading digits dataset...")
    digits_data = loader.load_from_sklearn('digits')
    print(f"  ✓ Loaded digits: {digits_data['n_samples']} samples")
    print(f"  ✓ Has label_stats: {'label_stats' in digits_data}")
    if 'label_stats' in digits_data:
        print(f"  ✓ Imbalance ratio: {digits_data['label_stats']['imbalance_ratio']:.2f}")
    # Test ExploratoryAnalyzer
    print("\nTesting ExploratoryAnalyzer...")
    from phase1_exploration.exploratory_analysis import ExploratoryAnalyzer
    analyzer = ExploratoryAnalyzer()
    label_stats = analyzer.analyze_label_distribution(
        digits_data['y'],
        digits_data['target_names'],
        'Digits'
    )
    print(f"  ✓ Analyzed labels: {label_stats['n_classes']} classes")
    print(f"  ✓ Imbalance ratio: {label_stats['imbalance_ratio']:.2f}")
    # Test DataVisualizer
    print("\nTesting DataVisualizer...")
    from phase1_exploration.visualization import DataVisualizer
    visualizer = DataVisualizer()
    # Create test datasets
    test_datasets = {
        'digits': digits_data,
        'test': {
            'X': [[1, 2], [3, 4], [5, 6], [7, 8]],
            'y': [0, 0, 1, 1],
            'label_stats': {'imbalance_ratio': 1.0}
        }
    }
    print("  Creating comparison plot...")
    fig = visualizer.plot_class_comparison(test_datasets)
    print("  ✓ Comparison plot created successfully")
    # Save test figure
    import matplotlib.pyplot as plt
    fig.savefig('test_comparison.png', dpi=100, bbox_inches='tight')
    print("  ✓ Figure saved as test_comparison.png")
    plt.close(fig)
    # Clean up
    if os.path.exists('test_comparison.png'):
        os.remove('test_comparison.png')
    print("\n" + "="*60)
    print("✅ ALL PHASE 1 MODULES WORKING CORRECTLY!")
    print("="*60)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
