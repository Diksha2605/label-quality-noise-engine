"""
Phase 3: Confident Learning Module - Main Runner
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import json
from datetime import datetime
from phase3_confident_learning.noise_estimation import ConfidentLearningNoiseEstimator
from phase3_confident_learning.joint_distribution import JointDistributionEstimator
from phase3_confident_learning.confidence_scoring import ConfidenceScorer
def load_data_for_phase3():
    """Load data for Phase 3 analysis."""
    # Try multiple possible data sources
    data_sources = [
        'data/phase2_ready',  # Phase 2 output
        'data/synthetic',     # Phase 0 output
        'data/processed'      # General processed data
    ]
    datasets = {}
    for data_dir in data_sources:
        if os.path.exists(data_dir):
            import glob
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dataset_name = os.path.basename(csv_file).replace('.csv', '')
                    # Check for target column
                    target_columns = ['target', 'label', 'class', 'y']
                    target_col = None
                    for col in target_columns:
                        if col in df.columns:
                            target_col = col
                            break
                    if target_col is None:
                        continue  # Skip files without target
                    # Separate features and target
                    feature_cols = [col for col in df.columns if col != target_col]
                    X = df[feature_cols].values
                    y = df[target_col].values
                    datasets[dataset_name] = {
                        'X': X,
                        'y': y,
                        'feature_names': feature_cols,
                        'dataframe': df,
                        'n_samples': len(df),
                        'n_features': X.shape[1],
                        'n_classes': len(np.unique(y)),
                        'source_file': csv_file
                    }
                    print(f"  ✓ {dataset_name}: {len(df)} samples from {data_dir}")
                except Exception as e:
                    print(f"  ✗ Error loading {csv_file}: {e}")
    return datasets
def run_phase_3():
    """Execute complete Phase 3 workflow."""
    print("=" * 80)
    print("LABEL QUALITY & NOISE ESTIMATION ENGINE - PHASE 3")
    print("CONFIDENT LEARNING MODULE (CORE)")
    print("=" * 80)
    # Create output directory
    output_dir = 'reports/phase3'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'noise_estimates'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'quality_scores'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    # Step 1: Load data
    print("\nSTEP 1: Loading data for analysis...")
    print("-" * 40)
    datasets = load_data_for_phase3()
    if not datasets:
        print("No datasets found in standard locations. Creating synthetic dataset...")
        # Create synthetic dataset with noise
        from sklearn.datasets import make_classification
        X, y_clean = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=4,
            n_informative=12,
            random_state=42
        )
        # Add 20% noise
        np.random.seed(42)
        noise_mask = np.random.rand(len(y_clean)) < 0.2
        y_noisy = y_clean.copy()
        for i in np.where(noise_mask)[0]:
            other_classes = [c for c in np.unique(y_clean) if c != y_clean[i]]
            y_noisy[i] = np.random.choice(other_classes)
        datasets = {
            'synthetic_with_noise': {
                'X': X,
                'y': y_noisy,
                'y_clean': y_clean,  # Store clean for evaluation
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y_clean)),
                'true_noise_rate': noise_mask.mean()
            }
        }
        print(f"  Created synthetic dataset with {noise_mask.sum()} noisy samples ({noise_mask.mean():.1%} noise)")
    print(f"\nLoaded {len(datasets)} datasets for analysis")
    # Initialize estimators
    print("\nSTEP 2: Initializing estimators...")
    print("-" * 40)
    confident_estimator = ConfidentLearningNoiseEstimator(
        cv_n_folds=5,
        prune_method='prune_by_noise_rate',
        random_state=42
    )
    joint_estimator = JointDistributionEstimator(random_state=42)
    confidence_scorer = ConfidenceScorer(random_state=42)
    # Step 3: Analyze each dataset
    print("\nSTEP 3: Analyzing datasets with Confident Learning...")
    print("-" * 40)
    all_results = {}
    for dataset_name, data in datasets.items():
        print(f"\n📊 Dataset: {dataset_name}")
        print(f"  • Samples: {data['n_samples']:,}")
        print(f"  • Features: {data['n_features']}")
        print(f"  • Classes: {data['n_classes']}")
        X = data['X']
        y = data['y']
        # 3A: Confident Learning noise estimation
        print("\n  A. Confident Learning Noise Estimation...")
        cl_results = confident_estimator.fit(X, y)
        # 3B: Joint distribution estimation
        print("\n  B. Joint Distribution Estimation...")
        # Get predicted probabilities for joint estimation
        from sklearn.model_selection import cross_val_predict
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, random_state=42)
        pred_probs = cross_val_predict(
            model, X_scaled, y, cv=5, method='predict_proba', n_jobs=-1
        )
        joint_dist = joint_estimator.estimate_confident_joint(y, pred_probs)
        noise_matrices = joint_estimator.estimate_noise_matrices(joint_dist)
        noise_analysis = joint_estimator.analyze_noise_patterns(noise_matrices['noise_matrix'])
        # 3C: Confidence scoring
        print("\n  C. Confidence Scoring...")
        quality_report = confidence_scorer.generate_quality_report(
            y, pred_probs, dataset_name=dataset_name
        )
        # 3D: If we have clean labels, evaluate
        evaluation = None
        if 'y_clean' in data:
            print("\n  D. Evaluation (synthetic noise)...")
            evaluation = confident_estimator.evaluate_on_synthetic_noise(
                X, data['y_clean'], y
            )
        # Compile results
        dataset_results = {
            'confident_learning': cl_results,
            'joint_distribution': {
                'joint_matrix': joint_dist.tolist() if hasattr(joint_dist, 'tolist') else joint_dist,
                'noise_matrices': {
                    'noise_matrix': noise_matrices['noise_matrix'].tolist(),
                    'inverse_noise_matrix': noise_matrices['inverse_noise_matrix'].tolist()
                },
                'noise_analysis': noise_analysis.to_dict('records')
            },
            'confidence_scoring': quality_report,
            'evaluation': evaluation,
            'dataset_info': {
                'name': dataset_name,
                'n_samples': data['n_samples'],
                'n_features': data['n_features'],
                'n_classes': data['n_classes']
            }
        }
        all_results[dataset_name] = dataset_results
        # Save individual dataset results
        dataset_output_dir = os.path.join(output_dir, 'noise_estimates', dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        # Save JSON results
        results_file = os.path.join(dataset_output_dir, 'confident_learning_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON
            serializable_results = {}
            for key, value in dataset_results.items():
                if key == 'confident_learning':
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
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"  ✓ Results saved to {results_file}")
        # Save ranked label errors to CSV
        if not cl_results['ranked_label_errors'].empty:
            errors_file = os.path.join(dataset_output_dir, 'ranked_label_errors.csv')
            cl_results['ranked_label_errors'].to_csv(errors_file, index=False)
            print(f"  ✓ Ranked label errors saved to {errors_file}")
        # Save quality scores
        if 'label_quality_scores' in quality_report:
            scores_file = os.path.join(output_dir, 'quality_scores', f'{dataset_name}_quality_scores.csv')
            pd.DataFrame({
                'sample_index': range(len(quality_report['label_quality_scores'])),
                'label': y,
                'quality_score': quality_report['label_quality_scores']
            }).to_csv(scores_file, index=False)
            print(f"  ✓ Quality scores saved to {scores_file}")
    # Step 4: Generate summary report
    print("\n\nSTEP 4: Generating summary report...")
    print("-" * 40)
    summary_data = []
    for dataset_name, results in all_results.items():
        cl_results = results['confident_learning']
        quality_report = results['confidence_scoring']
        summary_data.append({
            'dataset': dataset_name,
            'n_samples': results['dataset_info']['n_samples'],
            'n_classes': results['dataset_info']['n_classes'],
            'label_errors_found': cl_results['n_label_errors'],
            'label_error_rate': cl_results['label_error_rate'],
            'avg_label_quality': quality_report['summary']['avg_label_quality'],
            'avg_self_confidence': np.mean(cl_results.get('label_quality_scores', [0])),
            'noise_matrix_diag_mean': np.mean(np.diag(results['joint_distribution']['noise_matrices']['noise_matrix']))
        })
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'phase3_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print("\nPhase 3 Summary:")
    print("-" * 40)
    print(summary_df.to_string(index=False))
    # Step 5: Generate visualizations
    print("\n\nSTEP 5: Generating visualizations...")
    print("-" * 40)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Visualization 1: Label error rates across datasets
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        datasets_list = summary_df['dataset'].tolist()
        error_rates = summary_df['label_error_rate'].tolist()
        quality_scores = summary_df['avg_label_quality'].tolist()
        x = range(len(datasets_list))
        width = 0.35
        bars1 = ax1.bar([i - width/2 for i in x], error_rates, width, 
                       label='Label Error Rate', color='coral', alpha=0.8)
        bars2 = ax1.bar([i + width/2 for i in x], quality_scores, width,
                       label='Avg Label Quality', color='skyblue', alpha=0.8)
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Rate / Score')
        ax1.set_title('Label Quality Analysis Across Datasets')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets_list, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualizations', 'dataset_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Visualization 1: Dataset comparison saved")
        # Visualization 2: Noise matrix for first dataset
        if all_results:
            first_dataset = list(all_results.keys())[0]
            noise_matrix = np.array(all_results[first_dataset]['joint_distribution']['noise_matrices']['noise_matrix'])
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            im = ax2.imshow(noise_matrix, cmap='Blues', interpolation='nearest')
            plt.colorbar(im, ax=ax2, label='Probability')
            # Add text annotations
            for i in range(noise_matrix.shape[0]):
                for j in range(noise_matrix.shape[1]):
                    text = ax2.text(j, i, f'{noise_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
            ax2.set_xlabel('Observed Label (s)')
            ax2.set_ylabel('True Label (y)')
            ax2.set_title(f'Noise Matrix P(s|y) - {first_dataset}')
            ax2.set_xticks(range(noise_matrix.shape[1]))
            ax2.set_yticks(range(noise_matrix.shape[0]))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', f'{first_dataset}_noise_matrix.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Visualization 2: Noise matrix for {first_dataset} saved")
    except Exception as e:
        print(f"⚠ Could not create visualizations: {e}")
    # Step 6: Final report
    print("\n\nSTEP 6: Generating final report...")
    print("-" * 40)
    total_samples = sum([results['dataset_info']['n_samples'] for results in all_results.values()])
    total_errors = sum([results['confident_learning']['n_label_errors'] for results in all_results.values()])
    overall_error_rate = total_errors / total_samples if total_samples > 0 else 0
    final_report = {
        'phase': 3,
        'timestamp': datetime.now().isoformat(),
        'datasets_analyzed': list(all_results.keys()),
        'total_samples': total_samples,
        'total_label_errors_found': total_errors,
        'overall_label_error_rate': overall_error_rate,
        'summary_statistics': summary_df.to_dict('records'),
        'output_files': {
            'summary': 'phase3_summary.csv',
            'noise_estimates': [f'noise_estimates/{name}/confident_learning_results.json' for name in all_results.keys()],
            'quality_scores': [f'quality_scores/{name}_quality_scores.csv' for name in all_results.keys()],
            'visualizations': ['visualizations/dataset_comparison.png'] + 
                            [f'visualizations/{list(all_results.keys())[0]}_noise_matrix.png' if all_results else '']
        },
        'key_insights': {
            'high_error_warning': 'Datasets with >20% label error rate should be reviewed',
            'quality_threshold': 'Samples with quality score < 0.5 are high priority for review',
            'noise_patterns': 'Check noise matrices for systematic label confusion patterns'
        },
        'next_steps': [
            'Review ranked_label_errors.csv files for specific label issues',
            'Use quality scores to prioritize data cleaning efforts',
            'Proceed to Phase 4: Dataset Health & Class Profiling'
        ]
    }
    # Save final report
    report_file = os.path.join(output_dir, 'phase3_final_report.json')
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    print(f"\n✓ Final report saved to {report_file}")
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE!")
    print("=" * 80)
    print(f"\n📊 COMPREHENSIVE RESULTS:")
    print(f"  • Datasets analyzed: {len(all_results)}")
    print(f"  • Total samples processed: {total_samples:,}")
    print(f"  • Total label errors identified: {total_errors:,} ({overall_error_rate:.1%})")
    print(f"  • Average label error rate: {summary_df['label_error_rate'].mean():.3f}")
    print(f"  • Average label quality: {summary_df['avg_label_quality'].mean():.3f}")
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Summary: {summary_file}")
    print(f"  • Noise estimates: {output_dir}/noise_estimates/")
    print(f"  • Quality scores: {output_dir}/quality_scores/")
    print(f"  • Visualizations: {output_dir}/visualizations/")
    print(f"  • Final report: {report_file}")
    print(f"\n🎯 NEXT STEPS:")
    print("  1. Examine noise matrices to understand label confusion patterns")
    print("  2. Review ranked_label_errors.csv for specific label issues")
    print("  3. Use quality scores to prioritize data cleaning")
    print("  4. Proceed to Phase 4: Dataset Health & Class Profiling")
    print("=" * 80)
    return all_results, summary_df, final_report
def main():
    """Main function to run Phase 3."""
    try:
        # Check if cleanlab is installed
        import cleanlab
        print(f"✓ CleanLab version: {cleanlab.__version__}")
    except ImportError:
        print("\n⚠ CleanLab not installed. Installing required packages...")
        import subprocess
        import sys
        # Try to install cleanlab
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cleanlab"])
            print("✓ CleanLab installed successfully")
        except:
            print("❌ Could not install CleanLab. Please install manually: pip install cleanlab")
            return
    # Run Phase 3
    print("\nStarting Phase 3: Confident Learning Module...")
    results, summary, report = run_phase_3()
    return results, summary, report
if __name__ == "__main__":
    main()
