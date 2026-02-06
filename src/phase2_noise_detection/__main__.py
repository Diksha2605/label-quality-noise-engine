"""
Phase 2: Noise Detection - Baseline - Main Runner
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import json
from datetime import datetime
from phase2_noise_detection.baseline_detection import BaselineNoiseDetector
from phase2_noise_detection.cross_validation import CrossValidationAnalyzer
from phase2_noise_detection.disagreement_analysis import DisagreementAnalyzer
def load_phase2_data(data_dir: str = 'data/phase2_ready'):
    """Load datasets prepared in Phase 1."""
    import glob
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Phase 2 data directory not found: {data_dir}")
    datasets = {}
    csv_files = glob.glob(os.path.join(data_dir, "*_cleaned.csv"))
    print(f"Loading datasets from {data_dir}...")
    print(f"Found {len(csv_files)} dataset files")
    for csv_file in csv_files:
        try:
            # Load CSV
            df = pd.read_csv(csv_file)
            # Extract dataset name
            dataset_name = os.path.splitext(os.path.basename(csv_file))[0].replace('_cleaned', '')
            # Separate features and target
            feature_columns = [col for col in df.columns if col != 'target']
            X = df[feature_columns].values
            y = df['target'].values
            datasets[dataset_name] = {
                'X': X,
                'y': y,
                'feature_names': feature_columns,
                'dataframe': df,
                'n_samples': len(df),
                'n_features': len(feature_columns),
                'n_classes': len(np.unique(y)),
                'filepath': csv_file
            }
            print(f"  ✓ {dataset_name}: {len(df)} samples, {len(feature_columns)} features")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file}: {e}")
    return datasets
def run_phase_2(data_dir: str = 'data/phase2_ready', 
                output_dir: str = 'reports/phase2'):
    """Execute complete Phase 2 workflow."""
    print("=" * 70)
    print("LABEL QUALITY & NOISE ESTIMATION ENGINE - PHASE 2")
    print("NOISE DETECTION - BASELINE")
    print("=" * 70)
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'detection_results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    # Step 1: Load datasets
    print("\nSTEP 1: Loading datasets from Phase 1...")
    print("-" * 40)
    datasets = load_phase2_data(data_dir)
    if not datasets:
        print("No datasets found! Running Phase 1 first or using sample data...")
        # Create sample dataset
        from sklearn.datasets import load_digits
        digits = load_digits()
        datasets = {
            'digits': {
                'X': digits.data,
                'y': digits.target,
                'feature_names': [f'pixel_{i}' for i in range(digits.data.shape[1])],
                'n_samples': digits.data.shape[0],
                'n_features': digits.data.shape[1],
                'n_classes': len(np.unique(digits.target))
            }
        }
        print("  Created sample digits dataset")
    print(f"\nLoaded {len(datasets)} datasets for analysis")
    # Step 2: Baseline Noise Detection
    print("\n\nSTEP 2: Baseline Noise Detection")
    print("-" * 40)
    all_results = {}
    for dataset_name, dataset_info in datasets.items():
        print(f"\nAnalyzing dataset: {dataset_name}")
        print(f"  • Samples: {dataset_info['n_samples']:,}")
        print(f"  • Features: {dataset_info['n_features']}")
        print(f"  • Classes: {dataset_info['n_classes']}")
        X = dataset_info['X']
        y = dataset_info['y']
        # Initialize detectors
        baseline_detector = BaselineNoiseDetector(
            model_type='logistic',
            n_folds=5,
            random_state=42
        )
        cv_analyzer = CrossValidationAnalyzer(
            n_folds=5,
            random_state=42
        )
        disagreement_analyzer = DisagreementAnalyzer(
            random_state=42
        )
        # 2A: Run baseline detection
        print("\n  Running baseline detection...")
        baseline_results = baseline_detector.compute_disagreement_scores(X, y)
        # 2B: Run CV analysis
        print("\n  Running cross-validation analysis...")
        cv_results = cv_analyzer.run_cv_analysis(X, y, model_type='logistic')
        # 2C: Run disagreement analysis
        print("\n  Running disagreement analysis...")
        disagreement_report = disagreement_analyzer.generate_disagreement_report(
            X, y, cv_results, dataset_name=dataset_name
        )
        # 2D: Rank suspicious samples
        suspicious_ranking = baseline_detector.rank_suspicious_samples(
            baseline_results, top_k=20
        )
        # 2E: Class analysis
        class_analysis = baseline_detector.analyze_by_class(X, y, baseline_results)
        # Save results
        dataset_results = {
            'baseline_results': baseline_results,
            'cv_results': cv_results,
            'disagreement_report': disagreement_report,
            'suspicious_ranking': suspicious_ranking.to_dict('records') if not suspicious_ranking.empty else [],
            'class_analysis': class_analysis.to_dict('records') if not class_analysis.empty else [],
            'dataset_info': dataset_info
        }
        all_results[dataset_name] = dataset_results
        # Save to files
        results_file = os.path.join(output_dir, 'detection_results', f'{dataset_name}_results.json')
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in dataset_results.items():
            if key in ['baseline_results', 'cv_results']:
                # Convert numpy arrays in these dicts
                serializable_value = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_value[k] = v.tolist()
                    elif isinstance(v, np.generic):
                        serializable_value[k] = v.item()
                    else:
                        serializable_value[k] = v
                serializable_results[key] = serializable_value
            else:
                serializable_results[key] = value
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"  ✓ Results saved to {results_file}")
        # Save suspicious samples to CSV
        if not suspicious_ranking.empty:
            csv_file = os.path.join(output_dir, 'detection_results', f'{dataset_name}_suspicious_samples.csv')
            suspicious_ranking.to_csv(csv_file, index=False)
            print(f"  ✓ Suspicious samples saved to {csv_file}")
    # Step 3: Generate Summary Report
    print("\n\nSTEP 3: Generating Summary Report")
    print("-" * 40)
    summary_data = []
    for dataset_name, results in all_results.items():
        baseline = results['baseline_results']
        cv = results['cv_results']
        summary_data.append({
            'dataset': dataset_name,
            'n_samples': baseline.get('n_samples', len(results['dataset_info']['y'])),
            'n_features': results['dataset_info']['n_features'],
            'n_classes': results['dataset_info']['n_classes'],
            'disagreement_rate': baseline.get('disagreement_rate', 0),
            'n_suspicious': baseline.get('n_suspicious', 0),
            'overall_accuracy': cv.get('overall_accuracy', 0),
            'overall_confidence': cv.get('overall_confidence', 0),
            'avg_confidence_suspicious': baseline.get('avg_confidence_suspicious', 0),
            'avg_confidence_correct': baseline.get('avg_confidence_correct', 0)
        })
    summary_df = pd.DataFrame(summary_data)
    # Save summary
    summary_file = os.path.join(output_dir, 'phase2_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print("\nPhase 2 Summary:")
    print("-" * 40)
    print(summary_df.to_string(index=False))
    # Step 4: Generate Visualizations
    print("\n\nSTEP 4: Generating Visualizations")
    print("-" * 40)
    # For each dataset, create visualizations
    for dataset_name, dataset_info in datasets.items():
        if dataset_name in all_results:
            print(f"\nCreating visualizations for {dataset_name}...")
            X = dataset_info['X']
            y = dataset_info['y']
            results = all_results[dataset_name]
            # Initialize analyzers
            cv_analyzer = CrossValidationAnalyzer(n_folds=5, random_state=42)
            disagreement_analyzer = DisagreementAnalyzer(random_state=42)
            # Create visualizations
            try:
                # CV visualization
                import matplotlib.pyplot as plt
                cv_fig = cv_analyzer.visualize_cv_results(
                    results['cv_results'],
                    dataset_name=dataset_name
                )
                cv_viz_file = os.path.join(output_dir, 'visualizations', f'{dataset_name}_cv_analysis.png')
                plt.savefig(cv_viz_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ CV visualization saved")
                # Disagreement visualization
                disagreement_analyzer.visualize_disagreement_analysis(
                    X, y, results['cv_results'],
                    dataset_name=dataset_name
                )
                disagree_viz_file = os.path.join(output_dir, 'visualizations', f'{dataset_name}_disagreement_analysis.png')
                plt.savefig(disagree_viz_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Disagreement visualization saved")
            except Exception as e:
                print(f"  ⚠ Could not create visualizations for {dataset_name}: {e}")
    # Step 5: Generate Final Report
    print("\n\nSTEP 5: Generating Final Report")
    print("-" * 40)
    # Calculate overall statistics
    total_samples = sum([info['n_samples'] for info in datasets.values()])
    total_suspicious = sum([res['baseline_results'].get('n_suspicious', 0) 
                          for res in all_results.values()])
    avg_disagreement = total_suspicious / total_samples if total_samples > 0 else 0
    final_report = {
        'phase': 2,
        'timestamp': datetime.now().isoformat(),
        'datasets_analyzed': list(datasets.keys()),
        'total_samples': total_samples,
        'total_suspicious_labels': total_suspicious,
        'average_disagreement_rate': avg_disagreement,
        'output_files': {
            'summary': 'phase2_summary.csv',
            'detection_results': [f'{name}_results.json' for name in datasets.keys()],
            'suspicious_samples': [f'{name}_suspicious_samples.csv' for name in datasets.keys()],
            'visualizations': [f'{name}_cv_analysis.png' for name in datasets.keys()] + 
                            [f'{name}_disagreement_analysis.png' for name in datasets.keys()]
        },
        'recommendations': {
            'high_disagreement': 'Consider reviewing labels for datasets with >20% disagreement rate',
            'low_confidence': 'Samples with low prediction confidence are good candidates for relabeling',
            'next_steps': 'Proceed to Phase 3 for more advanced noise estimation using Confident Learning'
        }
    }
    # Save final report
    report_file = os.path.join(output_dir, 'phase2_final_report.json')
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    print(f"\n✓ Final report saved to {report_file}")
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE!")
    print("=" * 70)
    print("\nOutput Summary:")
    print(f"  • Analyzed {len(datasets)} datasets")
    print(f"  • Total samples: {total_samples:,}")
    print(f"  • Total suspicious labels detected: {total_suspicious:,} ({avg_disagreement:.1%})")
    print(f"\nOutput Files:")
    print(f"  • Summary: {summary_file}")
    print(f"  • Detection results: {output_dir}/detection_results/")
    print(f"  • Visualizations: {output_dir}/visualizations/")
    print(f"  • Final report: {report_file}")
    print("\nNext Steps:")
    print("  1. Review suspicious_samples.csv files for potential label errors")
    print("  2. Examine visualizations to understand disagreement patterns")
    print("  3. Proceed to Phase 3: Confident Learning Module (Advanced noise estimation)")
    print("=" * 70)
    return all_results, summary_df, final_report
if __name__ == "__main__":
    # Run Phase 2
    results, summary, report = run_phase_2(
        data_dir='../data/phase2_ready',
        output_dir='../reports/phase2'
    )
