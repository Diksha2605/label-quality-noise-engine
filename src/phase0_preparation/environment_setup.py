"""
Main script for Phase 0: Preparation
Coordinates dataset preparation and noise injection.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import from current directory
from dataset_preparation import DataPreparation, main as prep_main
from noise_injection import NoiseInjector
def run_phase_0():
    """Execute complete Phase 0 workflow."""
    print("=" * 70)
    print("LABEL QUALITY & NOISE ESTIMATION ENGINE - PHASE 0")
    print("=" * 70)
    print("\nStep 1: Preparing clean datasets...")
    print("-" * 40)
    # Load and save clean datasets
    all_clean_datasets = prep_main()
    print("\nStep 2: Creating noisy datasets for testing...")
    print("-" * 40)
    # Test noise injection on synthetic dataset
    print("\nTesting noise injection on synthetic dataset...")
    from sklearn.datasets import make_classification
    # Create a test dataset with valid parameters
    X_test, y_test = make_classification(
        n_samples=500,
        n_features=15,
        n_classes=4,
        n_informative=5,  # 2^5=32 >= 4*2=8
        n_redundant=3,
        n_clusters_per_class=2,
        random_state=42
    )
    # Inject different types of noise
    injector = NoiseInjector(random_state=42)
    noise_types = ['random', 'class_dependent', 'hard']
    noise_rates = [0.1, 0.15, 0.05]
    for noise_type, noise_rate in zip(noise_types, noise_rates):
        if noise_type == 'random':
            y_noisy, noise_mask = injector.inject_random_noise(y_test, noise_rate)
        elif noise_type == 'class_dependent':
            y_noisy, noise_mask = injector.inject_class_dependent_noise(y_test)
        elif noise_type == 'hard':
            y_noisy, noise_mask = injector.inject_hard_example_noise(X_test, y_test, noise_rate)
        actual_rate = noise_mask.mean()
        print(f"  ✓ {noise_type}: Target={noise_rate:.3f}, Actual={actual_rate:.3f}, "
              f"Noisy samples={noise_mask.sum()}")
    print("\nStep 3: Creating comprehensive noisy dataset collection...")
    print("-" * 40)
    # Create a comprehensive noisy dataset for digits
    print("\nCreating comprehensive noisy dataset (Digits)...")
    from sklearn.datasets import load_digits
    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target
    noisy_digits = injector.create_noisy_dataset(
        X_digits, y_digits,
        noise_types=['random', 'class_dependent', 'hard'],
        noise_rates=[0.2, 0.25, 0.15],
        dataset_name='digits'
    )
    # Save the noisy digits datasets
    injector.save_noisy_datasets(noisy_digits, base_path='../data/synthetic/digits')
    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE!")
    print("=" * 70)
    print("\nGenerated datasets:")
    print("  • Clean datasets (digits, iris, wine, synthetic) in data/synthetic/")
    print("  • Noisy versions (random, class-dependent, hard) in data/synthetic/digits/")
    print("  • Ready for Phase 1: Data Loading & Exploration")
    print("\nNext steps:")
    print("  1. Run notebooks/0_phase0_preparation.ipynb to explore the datasets")
    print("  2. Proceed to src/phase1_exploration/ for Phase 1")
    print("=" * 70)
    return {
        'clean_datasets': all_clean_datasets,
        'noisy_digits': noisy_digits
    }
if __name__ == "__main__":
    results = run_phase_0()
