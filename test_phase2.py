"""
Simple test for Phase 2
"""
import sys
import os
sys.path.append('src')
try:
    print("Testing Phase 2 modules...")
    # Test imports
    from phase2_noise_detection.baseline_detection import BaselineNoiseDetector
    from phase2_noise_detection.cross_validation import CrossValidationAnalyzer
    from phase2_noise_detection.disagreement_analysis import DisagreementAnalyzer
    print("✓ All Phase 2 modules imported successfully!")
    # Create a small test dataset
    from sklearn.datasets import make_classification
    import numpy as np
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=3,
        n_informative=5,
        random_state=42
    )
    print(f"\\nTest dataset: {X.shape[0]} samples, {X.shape[1]} features")
    # Test BaselineNoiseDetector
    print("\\n1. Testing BaselineNoiseDetector...")
    detector = BaselineNoiseDetector(n_folds=3, random_state=42)
    results = detector.compute_disagreement_scores(X, y)
    print(f"   ✓ Computed disagreement scores")
    print(f"     • Suspicious: {results['n_suspicious']} ({results['disagreement_rate']:.1%})")
    # Test CrossValidationAnalyzer
    print("\\n2. Testing CrossValidationAnalyzer...")
    cv_analyzer = CrossValidationAnalyzer(n_folds=3, random_state=42)
    cv_results = cv_analyzer.run_cv_analysis(X, y)
    print(f"   ✓ CV analysis complete")
    print(f"     • Accuracy: {cv_results['overall_accuracy']:.3f}")
    # Test DisagreementAnalyzer
    print("\\n3. Testing DisagreementAnalyzer...")
    disagreement_analyzer = DisagreementAnalyzer(random_state=42)
    report = disagreement_analyzer.generate_disagreement_report(X, y, cv_results)
    print(f"   ✓ Disagreement report generated")
    print(f"     • Recommendations: {len(report['recommendations'])}")
    print("\\n" + "="*60)
    print("✅ PHASE 2 MODULES WORKING CORRECTLY!")
    print("="*60)
    print("\\nTo run full Phase 2:")
    print("  python src/phase2_noise_detection/__main__.py")
    print("  or")
    print("  jupyter notebook notebooks/2_phase2_baseline.ipynb")
except Exception as e:
    print(f"\\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
