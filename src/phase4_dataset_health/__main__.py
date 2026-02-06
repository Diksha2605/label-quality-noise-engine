"""
Phase 4: Dataset Health Profiling - Main Runner
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4_dataset_health.health_metrics import DatasetHealthMetrics
from phase4_dataset_health.class_profiling import ClassProfiler
from phase4_dataset_health.quality_scoring import QualityScorer
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')
def run_phase_4(data_dir: str = 'data/phase2_ready'):
    """
    Execute complete Phase 4 workflow.
    Args:
        data_dir: Directory containing cleaned datasets from Phase 2
    """
    print("=" * 70)
    print("LABEL QUALITY & NOISE ESTIMATION ENGINE - PHASE 4")
    print("DATASET HEALTH PROFILING")
    print("=" * 70)
    # Step 1: Load datasets
    print("\nSTEP 1: Loading datasets for health profiling...")
    print("-" * 40)
    import glob
    dataset_files = glob.glob(os.path.join(data_dir, "*_cleaned.csv"))
    if not dataset_files:
        print("⚠ No cleaned datasets found. Creating sample datasets...")
        from sklearn.datasets import load_digits, load_iris
        dataset_files = []
        # Create sample datasets
        os.makedirs(data_dir, exist_ok=True)
        # Digits dataset
        digits = load_digits()
        digits_df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
        digits_df['target'] = digits.target
        digits_file = os.path.join(data_dir, 'digits_cleaned.csv')
        digits_df.to_csv(digits_file, index=False)
        dataset_files.append(digits_file)
        print(f"  ✓ Created digits dataset: {len(digits_df)} samples")
        # Iris dataset
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target
        iris_file = os.path.join(data_dir, 'iris_cleaned.csv')
        iris_df.to_csv(iris_file, index=False)
        dataset_files.append(iris_file)
        print(f"  ✓ Created iris dataset: {len(iris_df)} samples")
    print(f"\nFound {len(dataset_files)} datasets:")
    for file in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(file))[0].replace('_cleaned', '')
        df = pd.read_csv(file)
        print(f"  • {dataset_name}: {len(df)} samples, {df.shape[1]-1} features")
    # Step 2: Initialize components
    print("\n\nSTEP 2: Initializing health profiling components...")
    print("-" * 40)
    health_metrics = DatasetHealthMetrics()
    class_profiler = ClassProfiler()
    quality_scorer = QualityScorer()
    print("  ✓ Health Metrics initialized")
    print("  ✓ Class Profiler initialized")
    print("  ✓ Quality Scorer initialized")
    # Step 3: Process each dataset
    print("\n\nSTEP 3: Processing datasets...")
    print("-" * 40)
    all_reports = {}
    for dataset_file in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0].replace('_cleaned', '')
        print(f"\n📁 Processing: {dataset_name}")
        print("-" * 30)
        # Load dataset
        df = pd.read_csv(dataset_file)
        # Separate features and target
        target_col = 'target'
        if target_col not in df.columns:
            # Try to find target column
            potential_targets = ['target', 'label', 'class', 'y']
            for col in potential_targets:
                if col in df.columns:
                    target_col = col
                    break
        if target_col not in df.columns:
            print(f"  ⚠ Could not find target column in {dataset_name}. Skipping...")
            continue
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        print(f"  • Samples: {X.shape[0]}")
        print(f"  • Features: {X.shape[1]}")
        print(f"  • Classes: {len(np.unique(y))}")
        # 3.1 Health Metrics
        print(f"\n  📊 Computing health metrics...")
        health_report = health_metrics.generate_health_report(X, y, dataset_name)
        # 3.2 Class Profiling
        print(f"\n  👥 Profiling classes...")
        class_report = class_profiler.generate_class_report(
            X, y, dataset_name,
            target_names=[f'Class {i}' for i in np.unique(y)]
        )
        # 3.3 Quality Scoring
        print(f"\n  ⭐ Computing quality scores...")
        quality_report = quality_scorer.compute_dataset_quality_score(X, y, dataset_name)
        # 3.4 Identify low-quality samples
        print(f"\n  🔍 Identifying low-quality samples...")
        # 3.4 Identify low-quality samples
        print(f"\n  🔍 Identifying low-quality samples...")
        low_quality_samples = quality_scorer.identify_low_quality_samples(
            X, y, threshold=0.4, top_k=10
        )
        # Handle empty DataFrame
        if not low_quality_samples.empty:
            low_quality_list = low_quality_samples.to_dict('records')
        else:
            low_quality_list = []
            X, y, threshold=0.4, top_k=10
        )
        # Compile all results
        dataset_report = {
            'dataset_info': {
                'name': dataset_name,
                'file': dataset_file,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y))
            },
            'health_report': health_report,
            'class_report': class_report,
            'quality_report': quality_report,
'low_quality_samples': low_quality_list,
        }
        all_reports[dataset_name] = dataset_report
        print(f"\n  ✅ {dataset_name} analysis complete!")
    # Step 4: Generate comparative analysis
    print("\n\nSTEP 4: Comparative Analysis")
    print("-" * 40)
    if len(all_reports) > 1:
        print("\n📈 Comparing datasets:")
        # Create comparison DataFrame
        comparison_data = []
        for name, report in all_reports.items():
            quality = report['quality_report']
            comparison_data.append({
                'Dataset': name,
                'Overall Quality': quality['overall_score'],
                'Quality Grade': quality['quality_grade'],
                'Dataset Health': quality['component_scores']['dataset_health'],
                'Class Quality': quality['component_scores']['class_quality'],
                'Sample Quality': quality['component_scores']['sample_quality'],
                'Label Confidence': quality['component_scores']['label_confidence'],
                'Samples': report['dataset_info']['n_samples'],
                'Features': report['dataset_info']['n_features'],
                'Classes': report['dataset_info']['n_classes']
            })
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        # Identify best and worst datasets
        best_dataset = df_comparison.loc[df_comparison['Overall Quality'].idxmax()]
        worst_dataset = df_comparison.loc[df_comparison['Overall Quality'].idxmin()]
        print(f"\n🏆 Best dataset: {best_dataset['Dataset']} "
              f"(score: {best_dataset['Overall Quality']:.3f})")
        print(f"⚠️  Worst dataset: {worst_dataset['Dataset']} "
              f"(score: {worst_dataset['Overall Quality']:.3f})")
        # Save comparison
        os.makedirs('../reports/phase4', exist_ok=True)
        df_comparison.to_csv('../reports/phase4/dataset_comparison.csv', index=False)
        print(f"  ✓ Comparison saved to: reports/phase4/dataset_comparison.csv")
    # Step 5: Generate summary and save reports
    print("\n\nSTEP 5: Generating final reports...")
    print("-" * 40)
    # Create reports directory
    os.makedirs('../reports/phase4', exist_ok=True)
    os.makedirs('../reports/phase4/detailed', exist_ok=True)
    # Save individual reports
    for name, report in all_reports.items():
        # Save detailed report
        report_file = f'../reports/phase4/detailed/{name}_health_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        # Save low-quality samples
        if report['low_quality_samples']:
            low_quality_df = pd.DataFrame(report['low_quality_samples'])
            low_quality_file = f'../reports/phase4/detailed/{name}_low_quality_samples.csv'
            low_quality_df.to_csv(low_quality_file, index=False)
    # Generate summary report
    summary = {
        'phase': 4,
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets_analyzed': list(all_reports.keys()),
        'total_datasets': len(all_reports),
        'dataset_summary': [
            {
                'name': name,
                'n_samples': report['dataset_info']['n_samples'],
                'n_features': report['dataset_info']['n_features'],
                'n_classes': report['dataset_info']['n_classes'],
                'overall_quality': report['quality_report']['overall_score'],
                'quality_grade': report['quality_report']['quality_grade']
            }
            for name, report in all_reports.items()
        ]
    }
    summary_file = '../reports/phase4/summary_report.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📋 Reports saved:")
    print(f"  • Summary: reports/phase4/summary_report.json")
    print(f"  • Detailed: reports/phase4/detailed/*.json")
    print(f"  • Low-quality samples: reports/phase4/detailed/*_low_quality_samples.csv")
    if 'df_comparison' in locals():
        print(f"  • Comparison: reports/phase4/dataset_comparison.csv")
    # Step 6: Recommendations
    print("\n\nSTEP 6: Key Recommendations")
    print("-" * 40)
    all_recommendations = []
    for name, report in all_reports.items():
        quality = report['quality_report']
        recommendations = quality.get('recommendations', [])
        if recommendations:
            print(f"\n📌 For {name}:")
            for i, rec in enumerate(recommendations[:3], 1):  # Top 3 per dataset
                print(f"  {i}. {rec}")
                all_recommendations.append(f"{name}: {rec}")
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE!")
    print("=" * 70)
    print("\n🎯 What we accomplished:")
    print("  • Comprehensive health assessment for each dataset")
    print("  • Detailed class-by-class profiling")
    print("  • Overall quality scoring with component breakdown")
    print("  • Identification of low-quality samples")
    print("  • Comparative analysis across datasets")
    print("  • Specific recommendations for improvement")
    print("\n📂 Outputs generated in:")
    print("  • reports/phase4/ - All health profiling reports")
    print("\n➡️  Next steps:")
    print("  1. Review low-quality samples identified in the reports")
    print("  2. Address recommendations for dataset improvement")
    print("  3. Proceed to Phase 5: Synthetic Testing")
    print("\n" + "=" * 70)
    return {
        'health_metrics': health_metrics,
        'class_profiler': class_profiler,
        'quality_scorer': quality_scorer,
        'all_reports': all_reports,
        'summary': summary
    }
if __name__ == "__main__":
    results = run_phase_4(data_dir='../data/phase2_ready')
