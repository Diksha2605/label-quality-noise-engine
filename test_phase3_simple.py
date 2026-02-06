"""
Test the simple Phase 3 implementation
"""
import sys
import os
sys.path.append('src')
try:
    print("Testing Phase 3 Simple Implementation...")
    # Test the simple estimator
    from phase3_confident_learning.simple_estimator import SimpleNoiseEstimator
    print("✓ SimpleNoiseEstimator imported successfully!")
    # Create a small test dataset
    from sklearn.datasets import make_classification
    import numpy as np
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=3,
        n_informative=6,
        random_state=42
    )
    print(f"\nTest dataset: {X.shape[0]} samples, {X.shape[1]} features")
    # Test the estimator
    estimator = SimpleNoiseEstimator(
        cv_n_folds=3,
        confidence_threshold=0.5,
        random_state=42
    )
    results = estimator.fit(X, y)
    print(f"\nResults:")
    print(f"  • Label errors found: {results['n_label_errors']} ({results['label_error_rate']:.1%})")
    print(f"  • Average label quality: {results['avg_label_quality']:.3f}")
    if not results['ranked_label_errors'].empty:
        print(f"  • Ranked errors: {len(results['ranked_label_errors'])} samples")
    print("\n" + "="*60)
    print("✅ PHASE 3 SIMPLE IMPLEMENTATION WORKING!")
    print("="*60)
    print("\nTo run full Phase 3 (simple version):")
    print("  python src/phase3_confident_learning/simple_estimator.py")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
