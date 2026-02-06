"""
Phase 1: Data Loading & Exploration - Main Runner
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1_exploration.data_loader import DataLoader
from phase1_exploration.exploratory_analysis import ExploratoryAnalyzer
from phase1_exploration.visualization import DataVisualizer
def run_phase_1(data_dir: str = 'data'):
    """Execute complete Phase 1 workflow."""
    print("=" * 70)
    print("LABEL QUALITY & NOISE ESTIMATION ENGINE - PHASE 1")
    print("DATA LOADING & EXPLORATION")
    print("=" * 70)
    # Step 1: Load data
    print("\nSTEP 1: Loading Datasets")
    print("-" * 40)
    loader = DataLoader(data_dir=data_dir)
    # Try to load Phase 0 datasets
    phase0_dir = os.path.join(data_dir, 'synthetic')
    if os.path.exists(phase0_dir):
        print("Loading Phase 0 datasets...")
        csv_datasets = loader.load_from_directory(phase0_dir, pattern="*dataset*.csv")
        # Also load noisy datasets if they exist
        noisy_dir = os.path.join(phase0_dir, 'digits')
        if os.path.exists(noisy_dir):
            noisy_datasets = loader.load_from_directory(noisy_dir, pattern="*.csv")
            print(f"Loaded {len(noisy_datasets)} noisy datasets")
    else:
        print("Phase 0 data not found. Loading sklearn datasets instead...")
        # Load some sklearn datasets as fallback
        loader.load_from_sklearn('digits')
        loader.load_from_sklearn('iris')
        loader.load_from_sklearn('wine')
    # Display summary
    print("\nLoaded Datasets Summary:")
    print("-" * 40)
    summary_df = loader.get_dataset_summary()
    print(summary_df.to_string(index=False))
    # Step 2: Exploratory Analysis
    print("\n\nSTEP 2: Exploratory Analysis")
    print("-" * 40)
    analyzer = ExploratoryAnalyzer()
    # Analyze each dataset
    reports = {}
    for dataset_name in loader.list_datasets():
        print(f"\nAnalyzing: {dataset_name}")
        dataset_info = loader.get_dataset(dataset_name)
        # Generate report
        report = analyzer.generate_dataset_report(dataset_info)
        reports[dataset_name] = report
        # Create visualization
        fig = analyzer.plot_comprehensive_report(
            dataset_info, 
            save_path=f'../reports/phase1_{dataset_name}_report.png'
        )
    # Step 3: Advanced Visualization
    print("\n\nSTEP 3: Advanced Visualization")
    print("-" * 40)
    visualizer = DataVisualizer()
    # Create dataset comparison
    all_datasets = {name: loader.get_dataset(name) for name in loader.list_datasets()}
    comparison_fig = visualizer.plot_class_comparison(all_datasets)
    visualizer.save_visualization(
        comparison_fig, 
        'phase1_dataset_comparison',
        directory='../reports/visualizations'
    )
    # Create label quality indicators for the largest dataset
    if loader.list_datasets():
        largest_dataset = max(loader.list_datasets(), 
                            key=lambda x: loader.get_dataset(x)['n_samples'])
        dataset_info = loader.get_dataset(largest_dataset)
        quality_fig = visualizer.plot_label_quality_indicators(
            dataset_info['X'],
            dataset_info['y'],
            dataset_name=largest_dataset
        )
        visualizer.save_visualization(
            quality_fig,
            f'phase1_{largest_dataset}_quality_indicators',
            directory='../reports/visualizations'
        )
    # Step 4: Generate Final Report
    print("\n\nSTEP 4: Generating Final Report")
    print("-" * 40)
    # Create summary statistics
    total_samples = sum([loader.get_dataset(name)['n_samples'] 
                        for name in loader.list_datasets()])
    total_features = sum([loader.get_dataset(name)['n_features'] 
                         for name in loader.list_datasets()])
    print(f"\n📊 Phase 1 Summary:")
    print(f"   • Total datasets analyzed: {len(loader.list_datasets())}")
    print(f"   • Total samples: {total_samples:,}")
    print(f"   • Total features: {total_features}")
    print(f"   • Reports generated: {len(reports)}")
    print(f"   • Visualizations saved: 2+")
    # Save metadata
    import json
    os.makedirs('../reports/metadata', exist_ok=True)
    metadata = {
        'phase': 1,
        'datasets_analyzed': loader.list_datasets(),
        'total_samples': total_samples,
        'total_features': total_features,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open('../reports/metadata/phase1_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE!")
    print("=" * 70)
    print("\nOutputs generated:")
    print("  • Dataset reports in reports/phase1_*_report.png")
    print("  • Visualizations in reports/visualizations/")
    print("  • Metadata in reports/metadata/phase1_metadata.json")
    print("\nNext steps:")
    print("  1. Run notebooks/1_phase1_exploration.ipynb for interactive exploration")
    print("  2. Proceed to Phase 2: Noise Detection - Baseline")
    print("=" * 70)
    return {
        'loader': loader,
        'analyzer': analyzer,
        'visualizer': visualizer,
        'reports': reports,
        'metadata': metadata
    }
if __name__ == "__main__":
    # Import pandas for timestamp
    import pandas as pd
    results = run_phase_1(data_dir='../data')
