"""
Simple Phase 2 Runner - Compatibility version
"""
import sys
import os
import numpy as np
import pandas as pd
import json
# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
def run_simple_phase_2():
    """Run a simple version of Phase 2."""
    print("=" * 70)
    print("PHASE 2: SIMPLE NOISE DETECTION")
    print("=" * 70)
    # Try to import the simple version
    try:
        from phase2_noise_detection.baseline_simple import BaselineNoiseDetector
        print("✓ Loaded compatibility version of BaselineNoiseDetector")
    except:
        print("⚠ Could not load BaselineNoiseDetector, using fallback...")
        # Define a minimal version
        class SimpleDetector:
            def __init__(self, n_folds=3, random_state=42):
                self.n_folds = n_folds
                self.random_state = random_state
            def compute_disagreement_scores(self, X, y):
                from sklearn.model_selection import cross_val_predict
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, random_state=self.random_state)
                y_pred = cross_val_predict(model, X, y, cv=self.n_folds)
                disagreement = (y_pred != y).astype(float)
                return {
                    'disagreement_scores': disagreement,
                    'predicted_labels': y_pred,
                    'confidence_scores': np.ones(len(y)) * 0.5,  # Placeholder
                    'suspicious_mask': disagreement == 1,
                    'n_suspicious': disagreement.sum(),
                    'disagreement_rate': disagreement.mean()
                }
        BaselineNoiseDetector = SimpleDetector
    # Load data
    print("\n1. Loading data...")
    data_dir = '../data/phase2_ready'
    if not os.path.exists(data_dir):
        print(f"⚠ Data directory not found: {data_dir}")
        print("Creating sample data...")
        from sklearn.datasets import load_digits
        digits = load_digits()
        datasets = {
            'digits': {
                'X': digits.data[:100],  # Smaller sample
                'y': digits.target[:100],
                'name': 'digits'
            }
        }
    else:
        # Load CSV files
        import glob
        datasets = {}
        csv_files = glob.glob(os.path.join(data_dir, "*_cleaned.csv"))
        for csv_file in csv_files[:2]:  # Process just first 2 files
            try:
                df = pd.read_csv(csv_file)
                name = os.path.basename(csv_file).replace('_cleaned.csv', '')
                X = df[[col for col in df.columns if col != 'target']].values
                y = df['target'].values
                datasets[name] = {'X': X, 'y': y, 'name': name}
                print(f"  ✓ {name}: {len(df)} samples")
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {e}")
    if not datasets:
        print("No datasets found!")
        return
    # Analyze each dataset
    print("\n2. Analyzing datasets...")
    all_results = {}
    for name, data in datasets.items():
        print(f"\nDataset: {name}")
        print(f"  • Samples: {len(data['y'])}")
        print(f"  • Features: {data['X'].shape[1]}")
        detector = BaselineNoiseDetector(n_folds=3, random_state=42)
        results = detector.compute_disagreement_scores(data['X'], data['y'])
        all_results[name] = results
        print(f"  • Suspicious labels: {results['n_suspicious']} ({results['disagreement_rate']:.1%})")
    # Save results
    print("\n3. Saving results...")
    output_dir = '../reports/phase2_simple'
    os.makedirs(output_dir, exist_ok=True)
    # Create summary
    summary = []
    for name, results in all_results.items():
        summary.append({
            'dataset': name,
            'n_samples': len(datasets[name]['y']),
            'n_suspicious': results['n_suspicious'],
            'disagreement_rate': results['disagreement_rate']
        })
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary saved to {summary_file}")
    # Save detailed results
    for name, results in all_results.items():
        # Save suspicious samples
        suspicious_idx = np.where(results['suspicious_mask'])[0]
        if len(suspicious_idx) > 0:
            suspicious_df = pd.DataFrame({
                'sample_index': suspicious_idx,
                'predicted_label': results['predicted_labels'][suspicious_idx],
                'confidence': results['confidence_scores'][suspicious_idx]
            })
            suspicious_file = os.path.join(output_dir, f'{name}_suspicious.csv')
            suspicious_df.to_csv(suspicious_file, index=False)
            print(f"  ✓ {name}: {len(suspicious_idx)} suspicious samples saved")
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE (Simple Version)")
    print("=" * 70)
    print(f"\nAnalyzed {len(datasets)} datasets")
    print(f"Total suspicious labels: {sum([r['n_suspicious'] for r in all_results.values()])}")
    return all_results
if __name__ == "__main__":
    run_simple_phase_2()
