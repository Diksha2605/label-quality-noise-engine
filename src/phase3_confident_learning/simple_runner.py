"""
Phase 3: Simple Runner - No CleanLab Required
With proper JSON serialization
"""
import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
def convert_to_serializable(obj):
    """Convert numpy objects to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):  # Handle NaN values
        return None
    else:
        return obj
def run_simple_phase_3():
    """Run simple Phase 3 without CleanLab."""
    print("=" * 80)
    print("PHASE 3: SIMPLE NOISE ESTIMATION")
    print("No CleanLab Required")
    print("=" * 80)
    try:
        from phase3_confident_learning.simple_estimator import SimpleNoiseEstimator
    except ImportError as e:
        print(f"Error importing SimpleNoiseEstimator: {e}")
        print("Make sure the module exists in phase3_confident_learning/")
        return
    # Create output directory
    output_dir = 'reports/phase3_simple'
    os.makedirs(output_dir, exist_ok=True)
    # Step 1: Load or create data
    print("\n1. Loading or creating data...")
    datasets = {}
    # Try to load from Phase 2
    phase2_dir = 'data/phase2_ready'
    if os.path.exists(phase2_dir):
        import glob
        csv_files = glob.glob(os.path.join(phase2_dir, "*.csv"))
        for csv_file in csv_files[:2]:  # Process up to 2 files
            try:
                df = pd.read_csv(csv_file)
                dataset_name = os.path.basename(csv_file).replace('.csv', '')
                if 'target' in df.columns:
                    X = df.drop('target', axis=1).values
                    y = df['target'].values
                    datasets[dataset_name] = {
                        'X': X,
                        'y': y,
                        'n_samples': len(df),
                        'n_features': X.shape[1],
                        'n_classes': len(np.unique(y))
                    }
                    print(f"  ✓ {dataset_name}: {len(df)} samples")
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {e}")
    # If no data found, create synthetic
    if not datasets:
        print("  No data found. Creating synthetic dataset...")
        from sklearn.datasets import make_classification
        X, y_clean = make_classification(
            n_samples=300,
            n_features=15,
            n_classes=3,
            n_informative=8,
            random_state=42
        )
        # Add noise
        np.random.seed(42)
        noise_mask = np.random.rand(len(y_clean)) < 0.15
        y_noisy = y_clean.copy()
        for i in np.where(noise_mask)[0]:
            other_classes = [c for c in np.unique(y_clean) if c != y_clean[i]]
            y_noisy[i] = np.random.choice(other_classes)
        datasets['synthetic_with_noise'] = {
            'X': X,
            'y': y_noisy,
            'y_clean': y_clean,
            'n_samples': len(y_noisy),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y_clean)),
            'true_noise_rate': float(noise_mask.mean())
        }
        print(f"  ✓ Created synthetic dataset with {noise_mask.sum()} noisy samples")
    # Step 2: Analyze each dataset
    print(f"\n2. Analyzing {len(datasets)} datasets...")
    all_results = {}
    for dataset_name, data in datasets.items():
        print(f"\n📊 Dataset: {dataset_name}")
        print(f"  • Samples: {data['n_samples']}")
        print(f"  • Features: {data['n_features']}")
        print(f"  • Classes: {data['n_classes']}")
        X = data['X']
        y = data['y']
        # Initialize and fit estimator
        estimator = SimpleNoiseEstimator(
            cv_n_folds=5,
            confidence_threshold=0.5,
            random_state=42
        )
        results = estimator.fit(X, y)
        # Convert numpy types to Python native types for JSON
        serializable_results = convert_to_serializable(results)
        all_results[dataset_name] = serializable_results
        # Save results
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        # Save ranked errors
        if 'ranked_label_errors' in results and not results['ranked_label_errors'].empty:
            errors_file = os.path.join(dataset_dir, 'ranked_label_errors.csv')
            results['ranked_label_errors'].to_csv(errors_file, index=False)
            print(f"  ✓ Saved {len(results['ranked_label_errors'])} ranked errors")
        # Save quality scores
        if 'label_quality_scores' in results:
            scores_file = os.path.join(dataset_dir, 'label_quality_scores.csv')
            quality_df = pd.DataFrame({
                'sample_index': range(len(results['label_quality_scores'])),
                'label': y,
                'quality_score': results['label_quality_scores']
            })
            quality_df.to_csv(scores_file, index=False)
            print(f"  ✓ Saved quality scores")
    # Step 3: Generate summary
    print("\n3. Generating summary report...")
    summary_data = []
    for dataset_name, results in all_results.items():
        summary_data.append({
            'dataset': dataset_name,
            'n_samples': results.get('n_samples', 0),
            'n_classes': results.get('n_classes', 0),
            'label_errors_found': results.get('n_label_errors', 0),
            'label_error_rate': float(results.get('label_error_rate', 0)),
            'avg_label_quality': float(results.get('avg_label_quality', 0)),
            'median_label_quality': float(results.get('median_label_quality', 0))
        })
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print("\nSummary:")
    print("-" * 40)
    print(summary_df.to_string(index=False))
    # Step 4: Final report
    total_samples = sum([r.get('n_samples', 0) for r in all_results.values()])
    total_errors = sum([r.get('n_label_errors', 0) for r in all_results.values()])
    overall_error_rate = total_errors / total_samples if total_samples > 0 else 0
    final_report = {
        'phase': 3,
        'timestamp': datetime.now().isoformat(),
        'implementation': 'simple (no CleanLab)',
        'datasets_analyzed': list(datasets.keys()),
        'total_samples': int(total_samples),
        'total_label_errors_found': int(total_errors),
        'overall_label_error_rate': float(overall_error_rate),
        'summary': convert_to_serializable(summary_df.to_dict('records')),
        'output_files': {
            'summary': 'summary.csv',
            'dataset_results': [f'{name}/' for name in datasets.keys()]
        }
    }
    report_file = os.path.join(output_dir, 'final_report.json')
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2, cls=NumpyEncoder)
    print(f"\n✓ Final report saved to {report_file}")
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE (Simple Version)")
    print("=" * 80)
    print(f"\n📊 RESULTS:")
    print(f"  • Datasets analyzed: {len(datasets)}")
    print(f"  • Total samples: {total_samples:,}")
    print(f"  • Label errors identified: {total_errors:,} ({overall_error_rate:.1%})")
    print(f"\n📁 OUTPUT:")
    print(f"  • Directory: {output_dir}/")
    print(f"  • Summary: {summary_file}")
    print(f"  • Dataset results: {output_dir}/*/")
    print(f"\n🎯 NEXT STEPS:")
    print("  1. Review ranked_label_errors.csv for potential issues")
    print("  2. Examine label_quality_scores.csv to prioritize cleaning")
    print("  3. Proceed to Phase 4: Dataset Health & Class Profiling")
    print("=" * 80)
    return all_results, summary_df, final_report
if __name__ == "__main__":
    run_simple_phase_3()
