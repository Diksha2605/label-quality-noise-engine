"""
Phase 1: Data Loading & Exploration
Goal: Build robust dataset loader and explore labels
"""
import numpy as np
import pandas as pd
import os
import yaml
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')
class DataLoader:
    """Robust dataset loader for LQNE project."""
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize DataLoader.
        Args:
            data_dir: Base directory for datasets
        """
        self.data_dir = data_dir
        self.loaded_datasets = {}
    def load_from_csv(self, filepath: str, 
                      target_column: str = 'target',
                      feature_columns: Optional[List[str]] = None,
                      drop_columns: Optional[List[str]] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Load dataset from CSV file.
        Args:
            filepath: Path to CSV file
            target_column: Name of target/label column
            feature_columns: List of feature column names (if None, auto-detect)
            drop_columns: Columns to drop from features
            **kwargs: Additional arguments for pd.read_csv
        Returns:
            Dictionary containing X, y, feature_names, etc.
        """
        print(f"Loading dataset from: {filepath}")
        # Load CSV
        df = pd.read_csv(filepath, **kwargs)
        # Extract target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. "
                           f"Available columns: {list(df.columns)}")
        y = df[target_column].values
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        if drop_columns:
            feature_columns = [col for col in feature_columns if col not in drop_columns]
        # Extract features
        X = df[feature_columns].values
        # Get metadata
        feature_names = feature_columns
        target_names = [f'class_{i}' for i in np.unique(y)]
        # Calculate label stats
        unique_labels, counts = np.unique(y, return_counts=True)
        imbalance = counts.max() / counts.min() if counts.min() > 0 else float('inf')
        label_stats = {
            'n_samples': len(y),
            'n_classes': len(unique_labels),
            'unique_labels': unique_labels.tolist(),
            'counts': counts.tolist(),
            'imbalance_ratio': imbalance,
            'is_balanced': imbalance < 2.0
        }
        dataset_info = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'target_names': target_names,
            'dataframe': df,
            'n_samples': len(df),
            'n_features': len(feature_columns),
            'n_classes': len(np.unique(y)),
            'label_stats': label_stats,
            'filepath': filepath,
            'description': f'Loaded from {os.path.basename(filepath)}'
        }
        # Store in loaded datasets
        dataset_name = os.path.splitext(os.path.basename(filepath))[0]
        self.loaded_datasets[dataset_name] = dataset_info
        print(f"  ✓ Loaded {dataset_info['n_samples']} samples, "
              f"{dataset_info['n_features']} features, "
              f"{dataset_info['n_classes']} classes")
        return dataset_info
    def load_from_sklearn(self, dataset_name: str = 'digits', **kwargs) -> Dict[str, Any]:
        """
        Load dataset from scikit-learn.
        Args:
            dataset_name: Name of sklearn dataset ('digits', 'iris', 'wine')
            **kwargs: Additional arguments for sklearn dataset loader
        Returns:
            Dictionary containing dataset info
        """
        from sklearn.datasets import load_digits, load_iris, load_wine, load_breast_cancer
        dataset_loaders = {
            'digits': load_digits,
            'iris': load_iris,
            'wine': load_wine,
            'breast_cancer': load_breast_cancer
        }
        if dataset_name not in dataset_loaders:
            raise ValueError(f"Dataset '{dataset_name}' not available. "
                           f"Choose from: {list(dataset_loaders.keys())}")
        print(f"Loading sklearn dataset: {dataset_name}")
        data = dataset_loaders[dataset_name](**kwargs)
        # Calculate label stats for sklearn datasets
        unique_labels, counts = np.unique(data.target, return_counts=True)
        imbalance = counts.max() / counts.min() if counts.min() > 0 else float('inf')
        label_stats = {
            'n_samples': data.data.shape[0],
            'n_classes': len(unique_labels),
            'unique_labels': unique_labels.tolist(),
            'counts': counts.tolist(),
            'imbalance_ratio': imbalance,
            'is_balanced': imbalance < 2.0
        }
        dataset_info = {
            'X': data.data,
            'y': data.target,
            'feature_names': data.feature_names if hasattr(data, 'feature_names') 
                            else [f'feature_{i}' for i in range(data.data.shape[1])],
            'target_names': data.target_names if hasattr(data, 'target_names') 
                           else [f'class_{i}' for i in np.unique(data.target)],
            'n_samples': data.data.shape[0],
            'n_features': data.data.shape[1],
            'n_classes': len(np.unique(data.target)),
            'label_stats': label_stats,
            'description': f'sklearn {dataset_name} dataset'
        }
        self.loaded_datasets[dataset_name] = dataset_info
        print(f"  ✓ Loaded {dataset_info['n_samples']} samples, "
              f"{dataset_info['n_features']} features, "
              f"{dataset_info['n_classes']} classes")
        return dataset_info
    def load_from_directory(self, directory: str, pattern: str = "*.csv") -> Dict[str, Dict]:
        """
        Load all datasets from a directory.
        Args:
            directory: Directory containing dataset files
            pattern: File pattern to match (e.g., "*.csv", "*dataset*.csv")
        Returns:
            Dictionary of loaded datasets
        """
        import glob
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        datasets = {}
        filepaths = glob.glob(os.path.join(directory, pattern))
        print(f"Loading datasets from directory: {directory}")
        print(f"Found {len(filepaths)} files matching pattern '{pattern}'")
        for filepath in filepaths:
            try:
                dataset_name = os.path.splitext(os.path.basename(filepath))[0]
                dataset_info = self.load_from_csv(filepath)
                datasets[dataset_name] = dataset_info
            except Exception as e:
                print(f"  ⚠ Warning: Could not load {filepath}: {e}")
        return datasets
    def get_dataset(self, name: str) -> Dict[str, Any]:
        """Get a loaded dataset by name."""
        if name not in self.loaded_datasets:
            raise KeyError(f"Dataset '{name}' not found. "
                         f"Available datasets: {list(self.loaded_datasets.keys())}")
        return self.loaded_datasets[name]
    def list_datasets(self) -> List[str]:
        """List all loaded datasets."""
        return list(self.loaded_datasets.keys())
    def get_dataset_summary(self) -> pd.DataFrame:
        """Get summary statistics of all loaded datasets."""
        summary_data = []
        for name, data in self.loaded_datasets.items():
            summary_data.append({
                'dataset': name,
                'n_samples': data['n_samples'],
                'n_features': data['n_features'],
                'n_classes': data['n_classes'],
                'description': data['description']
            })
        return pd.DataFrame(summary_data)
def main():
    """Test the DataLoader."""
    print("=" * 60)
    print("Phase 1: Data Loading Module")
    print("=" * 60)
    # Initialize loader
    loader = DataLoader(data_dir='../data')
    # Load from sklearn
    print("\n1. Loading sklearn datasets...")
    digits_data = loader.load_from_sklearn('digits')
    iris_data = loader.load_from_sklearn('iris')
    # Try to load from CSV (if Phase 0 data exists)
    print("\n2. Trying to load CSV datasets from Phase 0...")
    csv_dir = '../data/synthetic'
    if os.path.exists(csv_dir):
        csv_datasets = loader.load_from_directory(csv_dir, pattern="*dataset*.csv")
        print(f"   ✓ Loaded {len(csv_datasets)} CSV datasets")
    else:
        print("   ⚠ Phase 0 data not found. Run Phase 0 first.")
    # Display summary
    print("\n3. Dataset Summary:")
    print("-" * 40)
    summary = loader.get_dataset_summary()
    print(summary.to_string(index=False))
    print("\n" + "=" * 60)
    print("Data Loading Complete!")
    print("=" * 60)
    return loader
if __name__ == "__main__":
    loader = main()
