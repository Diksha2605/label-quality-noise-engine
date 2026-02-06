"""
Phase 0: Environment Setup and Dataset Preparation
Goal: Set up environment, understand labels and noise.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
class DataPreparation:
    """Prepare clean datasets and inject synthetic noise."""
    @staticmethod
    def load_sklearn_datasets():
        """Load standard sklearn datasets for testing."""
        datasets = {
            'digits': load_digits(),
            'iris': load_iris(),
            'wine': load_wine()
        }
        prepared_data = {}
        for name, data in datasets.items():
            X = data.data
            y = data.target
            feature_names = data.feature_names if hasattr(data, 'feature_names') else [f'feature_{i}' for i in range(X.shape[1])]
            target_names = data.target_names if hasattr(data, 'target_names') else [f'class_{i}' for i in np.unique(y)]
            prepared_data[name] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'target_names': target_names,
                'description': f'{name.capitalize()} dataset from sklearn'
            }
        return prepared_data
    @staticmethod
    def create_synthetic_dataset(n_samples=1000, n_features=20, n_classes=3, random_state=42):
        """Create a synthetic classification dataset with valid parameters."""
        from sklearn.datasets import make_classification
        # Calculate valid n_informative
        # Constraint: 2**n_informative >= n_classes * n_clusters_per_class
        n_clusters_per_class = 2
        # Find minimum n_informative that satisfies the condition
        min_informative = 1
        while 2**min_informative < n_classes * n_clusters_per_class:
            min_informative += 1
        # Use 70% of features as informative, but at least the minimum required
        n_informative = max(min_informative, int(n_features * 0.7))
        # Ensure n_informative doesn't exceed n_features
        n_informative = min(n_informative, n_features)
        # Calculate n_redundant
        n_redundant = max(0, int(n_features * 0.2))
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            random_state=random_state,
            flip_y=0.05  # Add some inherent noise
        )
        return {
            'X': X,
            'y': y,
            'feature_names': [f'feature_{i}' for i in range(n_features)],
            'target_names': [f'class_{i}' for i in range(n_classes)],
            'description': f'Synthetic dataset: {n_samples} samples, {n_features} features, {n_classes} classes',
            'n_informative': n_informative,
            'n_redundant': n_redundant
        }
    @staticmethod
    def save_dataset_to_csv(data_dict, base_path='data/synthetic'):
        """Save datasets to CSV files."""
        import os
        os.makedirs(base_path, exist_ok=True)
        for name, data in data_dict.items():
            # Create DataFrame for features
            X_df = pd.DataFrame(data['X'], columns=data['feature_names'])
            # Add target column
            df = X_df.copy()
            df['target'] = data['y']
            df['target_name'] = [data['target_names'][i] for i in data['y']]
            # Save to CSV
            filename = f'{base_path}/{name}_dataset.csv'
            df.to_csv(filename, index=False)
            print(f"✓ Saved {name} dataset to {filename} ({len(df)} samples)")
        return True
def main():
    """Main execution for Phase 0."""
    print("=" * 60)
    print("Phase 0: Environment Setup and Dataset Preparation")
    print("=" * 60)
    # Initialize data preparation
    prep = DataPreparation()
    # 1. Load sklearn datasets
    print("\n1. Loading sklearn datasets...")
    sklearn_data = prep.load_sklearn_datasets()
    for name, data in sklearn_data.items():
        print(f"   ✓ {name}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features, {len(np.unique(data['y']))} classes")
    # 2. Create synthetic dataset
    print("\n2. Creating synthetic dataset...")
    synthetic_data = prep.create_synthetic_dataset()
    print(f"   ✓ Synthetic: {synthetic_data['X'].shape[0]} samples, "
          f"{synthetic_data['X'].shape[1]} features, "
          f"{len(np.unique(synthetic_data['y']))} classes")
    # 3. Combine all datasets
    all_data = {**sklearn_data, 'synthetic': synthetic_data}
    # 4. Save to CSV
    print("\n3. Saving datasets to CSV...")
    prep.save_dataset_to_csv(all_data)
    print("\n" + "=" * 60)
    print("Phase 0 Complete! Ready for Phase 1: Data Exploration")
    print("=" * 60)
    return all_data
if __name__ == "__main__":
    all_datasets = main()
