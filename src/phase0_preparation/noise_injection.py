"""
Noise Injection Module for Phase 0.
Create noisy versions of clean datasets for testing noise detection algorithms.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import copy
class NoiseInjector:
    """Inject synthetic noise into datasets."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    def inject_random_noise(self, y: np.ndarray, noise_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject random label noise.
        Args:
            y: Clean labels
            noise_rate: Proportion of labels to corrupt (0-1)
        Returns:
            Tuple of (noisy_labels, noise_mask) where noise_mask indicates corrupted samples
        """
        n_samples = len(y)
        n_noisy = int(n_samples * noise_rate)
        # Randomly select samples to corrupt
        noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)
        # Create noisy labels
        y_noisy = y.copy()
        noise_mask = np.zeros(n_samples, dtype=bool)
        for idx in noisy_indices:
            noise_mask[idx] = True
            original_class = y[idx]
            # Choose a different class randomly
            possible_classes = [c for c in np.unique(y) if c != original_class]
            if possible_classes:
                y_noisy[idx] = np.random.choice(possible_classes)
        return y_noisy, noise_mask
    def inject_class_dependent_noise(self, y: np.ndarray, 
                                     noise_matrix: Optional[np.ndarray] = None,
                                     class_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject class-dependent label noise.
        Args:
            y: Clean labels
            noise_matrix: n_classes x n_classes matrix where noise_matrix[i,j] = P(observed=j|true=i)
            class_names: Optional list of class names
        Returns:
            Tuple of (noisy_labels, noise_mask)
        """
        n_classes = len(np.unique(y))
        y_noisy = y.copy()
        noise_mask = np.zeros(len(y), dtype=bool)
        # Create default noise matrix if not provided
        if noise_matrix is None:
            noise_matrix = np.eye(n_classes) * 0.7  # 70% correct
            for i in range(n_classes):
                # Distribute remaining 30% among other classes
                remaining = 0.3
                other_classes = [j for j in range(n_classes) if j != i]
                for j in other_classes[:-1]:
                    noise_matrix[i, j] = remaining / (len(other_classes) + 1)
                    remaining -= noise_matrix[i, j]
                noise_matrix[i, other_classes[-1]] = remaining
        # Normalize noise matrix rows to sum to 1
        noise_matrix = noise_matrix / noise_matrix.sum(axis=1, keepdims=True)
        # Inject noise based on noise matrix
        for i in range(n_classes):
            class_indices = np.where(y == i)[0]
            if len(class_indices) > 0:
                # For each true class i, sample observed labels from noise_matrix[i, :]
                probs = noise_matrix[i, :]
                noisy_labels_for_class = np.random.choice(
                    n_classes, 
                    size=len(class_indices), 
                    p=probs
                )
                y_noisy[class_indices] = noisy_labels_for_class
                noise_mask[class_indices] = (noisy_labels_for_class != i)
        return y_noisy, noise_mask
    def inject_hard_example_noise(self, X: np.ndarray, y: np.ndarray, 
                                   noise_rate: float = 0.1,
                                   model_type: str = 'logistic') -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject noise on 'hard' examples (samples near decision boundaries).
        Args:
            X: Features
            y: Clean labels
            noise_rate: Proportion of labels to corrupt
            model_type: Type of model to identify hard examples ('logistic', 'svm', 'tree')
        Returns:
            Tuple of (noisy_labels, noise_mask)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_predict
        # Choose model
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=self.random_state)
        elif model_type == 'tree':
            model = DecisionTreeClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        # Get predicted probabilities using cross-validation
        y_pred_proba = cross_val_predict(
            model, X, y, 
            method='predict_proba',
            cv=3,
            n_jobs=-1
        )
        # Calculate uncertainty (1 - max probability)
        uncertainty = 1 - np.max(y_pred_proba, axis=1)
        # Select most uncertain samples to corrupt
        n_noisy = int(len(y) * noise_rate)
        noisy_indices = np.argsort(uncertainty)[-n_noisy:]
        # Inject noise
        y_noisy = y.copy()
        noise_mask = np.zeros(len(y), dtype=bool)
        for idx in noisy_indices:
            noise_mask[idx] = True
            original_class = y[idx]
            # Choose a different class randomly
            possible_classes = [c for c in np.unique(y) if c != original_class]
            if possible_classes:
                y_noisy[idx] = np.random.choice(possible_classes)
        return y_noisy, noise_mask
    def create_noisy_dataset(self, X: np.ndarray, y: np.ndarray,
                             noise_types: List[str] = ['random', 'class_dependent', 'hard'],
                             noise_rates: List[float] = [0.1, 0.1, 0.1],
                             dataset_name: str = 'dataset') -> Dict[str, Dict]:
        """
        Create multiple noisy versions of a dataset.
        Args:
            X: Features
            y: Clean labels
            noise_types: List of noise types to inject
            noise_rates: Corresponding noise rates
            dataset_name: Name of the dataset
        Returns:
            Dictionary of noisy datasets
        """
        noisy_datasets = {
            'clean': {
                'X': X.copy(),
                'y': y.copy(),
                'noise_type': 'clean',
                'noise_rate': 0.0,
                'noise_mask': np.zeros(len(y), dtype=bool),
                'description': f'{dataset_name} (clean)'
            }
        }
        for noise_type, noise_rate in zip(noise_types, noise_rates):
            if noise_type == 'random':
                y_noisy, noise_mask = self.inject_random_noise(y, noise_rate)
                desc = f'{dataset_name} (random noise: {noise_rate*100:.1f}%)'
            elif noise_type == 'class_dependent':
                y_noisy, noise_mask = self.inject_class_dependent_noise(y)
                actual_noise_rate = noise_mask.mean()
                desc = f'{dataset_name} (class-dependent noise: {actual_noise_rate*100:.1f}%)'
            elif noise_type == 'hard':
                y_noisy, noise_mask = self.inject_hard_example_noise(X, y, noise_rate)
                desc = f'{dataset_name} (hard example noise: {noise_rate*100:.1f}%)'
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            noisy_datasets[noise_type] = {
                'X': X.copy(),
                'y': y_noisy,
                'noise_type': noise_type,
                'noise_rate': noise_mask.mean(),
                'noise_mask': noise_mask,
                'description': desc
            }
        return noisy_datasets
    def save_noisy_datasets(self, noisy_datasets: Dict, base_path: str = 'data/synthetic'):
        """Save noisy datasets to CSV files."""
        import os
        os.makedirs(base_path, exist_ok=True)
        for name, data in noisy_datasets.items():
            # Create DataFrame
            X_df = pd.DataFrame(data['X'])
            df = X_df.copy()
            df['target_clean'] = data.get('y_clean', data['y'])  # Store clean labels if available
            df['target_noisy'] = data['y']
            df['is_noisy'] = data.get('noise_mask', False)
            # Add metadata
            df.attrs['description'] = data['description']
            df.attrs['noise_type'] = data['noise_type']
            df.attrs['noise_rate'] = data['noise_rate']
            # Save to CSV
            filename = f'{base_path}/{data["description"].replace(" ", "_").replace(":", "").replace("%", "")}.csv'
            df.to_csv(filename, index=False)
            print(f"✓ Saved {name} dataset to {filename} (noise rate: {data['noise_rate']:.3f})")
        return True
def main():
    """Test the noise injection module."""
    print("=" * 60)
    print("Noise Injection Module")
    print("=" * 60)
    # Create a simple test dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        random_state=42
    )
    # Initialize noise injector
    injector = NoiseInjector(random_state=42)
    # Create noisy datasets
    print("\nCreating noisy versions of test dataset...")
    noisy_datasets = injector.create_noisy_dataset(
        X, y,
        noise_types=['random', 'class_dependent', 'hard'],
        noise_rates=[0.15, 0.2, 0.1],
        dataset_name='test'
    )
    # Display results
    for name, data in noisy_datasets.items():
        print(f"\n{name}:")
        print(f"  Description: {data['description']}")
        print(f"  Noise type: {data['noise_type']}")
        print(f"  Actual noise rate: {data['noise_rate']:.3f}")
        print(f"  Noisy samples: {data['noise_mask'].sum() if 'noise_mask' in data else 0}")
    # Save to CSV
    print("\nSaving datasets...")
    injector.save_noisy_datasets(noisy_datasets)
    print("\n" + "=" * 60)
    print("Noise injection complete!")
    print("=" * 60)
    return noisy_datasets
if __name__ == "__main__":
    datasets = main()
