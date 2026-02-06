"""
Joint distribution and noise matrix estimation for Phase 3.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class JointDistributionEstimator:
    """Estimate joint distribution of observed vs true labels."""
    def __init__(self, random_state: int = 42):
        """Initialize estimator."""
        self.random_state = random_state
        np.random.seed(random_state)
    def estimate_confident_joint(self,
                                labels: np.ndarray,
                                pred_probs: np.ndarray,
                                thresholds: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate confident joint distribution P(y, s).
        Args:
            labels: Observed labels s
            pred_probs: Predicted probabilities P(y|X)
            thresholds: Per-class thresholds (if None, auto-compute)
        Returns:
            Confident joint matrix of shape (n_true_classes, n_observed_classes)
        """
        n_samples = len(labels)
        n_classes = pred_probs.shape[1]
        # Initialize confident joint matrix
        confident_joint = np.zeros((n_classes, n_classes), dtype=np.float64)
        # Compute thresholds if not provided
        if thresholds is None:
            thresholds = self._compute_thresholds(labels, pred_probs)
        print(f"Estimating confident joint with {n_classes} classes...")
        # Build confident joint
        for i in range(n_samples):
            observed_label = labels[i]
            pred_prob = pred_probs[i]
            # Find classes where predicted probability exceeds threshold
            confident_classes = np.where(pred_prob >= thresholds)[0]
            for true_label in confident_classes:
                confident_joint[true_label, observed_label] += 1
        # Normalize to get joint probability distribution
        joint_distribution = confident_joint / confident_joint.sum()
        print(f"  • Confident joint shape: {joint_distribution.shape}")
        print(f"  • Total confident counts: {confident_joint.sum():.0f}")
        return joint_distribution
    def _compute_thresholds(self,
                           labels: np.ndarray,
                           pred_probs: np.ndarray,
                           percentile: float = 90.0) -> np.ndarray:
        """Compute per-class thresholds for confident learning."""
        n_classes = pred_probs.shape[1]
        thresholds = np.zeros(n_classes)
        for class_idx in range(n_classes):
            # Get predicted probabilities for this class
            class_probs = pred_probs[:, class_idx]
            # For threshold computation, we could use various methods
            # Here we use a percentile of predictions where this class was predicted
            if len(class_probs) > 0:
                # Simple method: use percentile of all predictions
                thresholds[class_idx] = np.percentile(class_probs, percentile)
            else:
                thresholds[class_idx] = 0.5  # Default
        print(f"Computed thresholds at {percentile}th percentile:")
        for i, thresh in enumerate(thresholds):
            print(f"  Class {i}: {thresh:.3f}")
        return thresholds
    def estimate_noise_matrices(self,
                               joint_distribution: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Estimate noise matrices from joint distribution.
        Args:
            joint_distribution: P(y, s) - joint of true vs observed
        Returns:
            Dictionary with:
                - noise_matrix: P(s|y) - probability of observing s given true y
                - inverse_noise_matrix: P(y|s) - probability of true y given observed s
        """
        n_classes = joint_distribution.shape[0]
        # Estimate marginal distributions
        py = joint_distribution.sum(axis=1)  # P(y) - true label distribution
        ps = joint_distribution.sum(axis=0)  # P(s) - observed label distribution
        # Avoid division by zero
        py_safe = np.where(py == 0, 1e-10, py)
        ps_safe = np.where(ps == 0, 1e-10, ps)
        # Noise matrix: P(s|y) = P(y,s) / P(y)
        noise_matrix = joint_distribution / py_safe[:, None]
        # Inverse noise matrix: P(y|s) = P(y,s) / P(s)
        inverse_noise_matrix = joint_distribution / ps_safe[None, :]
        # Normalize to ensure proper probability distributions
        noise_matrix = noise_matrix / noise_matrix.sum(axis=1, keepdims=True)
        inverse_noise_matrix = inverse_noise_matrix / inverse_noise_matrix.sum(axis=0, keepdims=True)
        print(f"\nNoise Matrix P(s|y) - Rows sum to 1:")
        self._print_matrix_summary(noise_matrix)
        print(f"\nInverse Noise Matrix P(y|s) - Columns sum to 1:")
        self._print_matrix_summary(inverse_noise_matrix.T)  # Transpose for column view
        return {
            'noise_matrix': noise_matrix,
            'inverse_noise_matrix': inverse_noise_matrix,
            'py': py,
            'ps': ps
        }
    def _print_matrix_summary(self, matrix: np.ndarray):
        """Print summary of a matrix."""
        n_classes = matrix.shape[0]
        print(f"  Shape: {matrix.shape}")
        print(f"  Diagonal (correct rates): {np.diag(matrix).round(3)}")
        print(f"  Off-diagonal mean: {matrix[~np.eye(n_classes, dtype=bool)].mean():.4f}")
        print(f"  Min value: {matrix.min():.4f}, Max value: {matrix.max():.4f}")
        # Show most common errors
        if n_classes <= 10:  # Only show for small number of classes
            print(f"  Matrix:")
            for i in range(n_classes):
                row_str = "  ".join([f"{val:.3f}" for val in matrix[i]])
                print(f"    Class {i}: [{row_str}]")
    def estimate_latent_distribution(self,
                                    labels: np.ndarray,
                                    confident_joint: np.ndarray,
                                    method: str = 'cnt') -> np.ndarray:
        """
        Estimate latent (true) label distribution P(y).
        Args:
            labels: Observed labels
            confident_joint: Confident joint matrix
            method: Estimation method ('cnt', 'inv', 'mle')
        Returns:
            Estimated latent distribution P(y)
        """
        n_classes = confident_joint.shape[0]
        if method == 'cnt':
            # Count method: use confident counts
            py = confident_joint.sum(axis=1)
            py = py / py.sum()
        elif method == 'inv':
            # Inverse method: P(y) = sum_s P(y|s) * P(s)
            ps = np.bincount(labels, minlength=n_classes) / len(labels)
            # Estimate P(y|s) from confident joint
            ps_safe = np.where(ps == 0, 1e-10, ps)
            p_y_given_s = confident_joint / ps_safe[None, :]
            p_y_given_s = p_y_given_s / p_y_given_s.sum(axis=0, keepdims=True)
            py = p_y_given_s @ ps
        elif method == 'mle':
            # Simple maximum likelihood from confident joint
            py = confident_joint.sum(axis=1)
            py = py / py.sum()
        else:
            raise ValueError(f"Unknown method: {method}")
        print(f"\nLatent Distribution P(y) estimated with '{method}' method:")
        for i, prob in enumerate(py):
            print(f"  Class {i}: {prob:.4f}")
        return py
    def analyze_noise_patterns(self,
                              noise_matrix: np.ndarray,
                              class_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze patterns in the noise matrix.
        Args:
            noise_matrix: P(s|y) matrix
            class_names: Optional names for classes
        Returns:
            DataFrame with noise pattern analysis
        """
        n_classes = noise_matrix.shape[0]
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        analysis_data = []
        for true_class in range(n_classes):
            row = noise_matrix[true_class]
            # Self-noise (probability of correct label)
            self_noise = row[true_class]
            # Find most common error
            other_classes = [c for c in range(n_classes) if c != true_class]
            if other_classes:
                error_probs = row[other_classes]
                most_common_error = other_classes[np.argmax(error_probs)]
                most_common_error_prob = np.max(error_probs)
                # Error concentration
                total_error = 1 - self_noise
                error_concentration = most_common_error_prob / total_error if total_error > 0 else 0
            else:
                most_common_error = -1
                most_common_error_prob = 0
                error_concentration = 0
            analysis_data.append({
                'true_class': true_class,
                'true_class_name': class_names[true_class],
                'self_noise_rate': 1 - self_noise,  # Probability of error
                'correct_rate': self_noise,
                'most_common_error_class': most_common_error,
                'most_common_error_class_name': class_names[most_common_error] if most_common_error >= 0 else 'None',
                'most_common_error_prob': most_common_error_prob,
                'error_concentration': error_concentration,
                'total_error_prob': 1 - self_noise
            })
        df_analysis = pd.DataFrame(analysis_data)
        # Sort by self-noise rate (most noisy first)
        df_analysis = df_analysis.sort_values('self_noise_rate', ascending=False)
        return df_analysis
def main():
    """Test the JointDistributionEstimator."""
    print("=" * 60)
    print("Joint Distribution Estimation Test")
    print("=" * 60)
    # Create synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    print("\n1. Creating synthetic dataset...")
    X, y_true = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=4,
        n_informative=6,
        random_state=42
    )
    # Add noise to create observed labels
    np.random.seed(42)
    noise_mask = np.random.rand(len(y_true)) < 0.2
    y_observed = y_true.copy()
    for i in np.where(noise_mask)[0]:
        other_classes = [c for c in np.unique(y_true) if c != y_true[i]]
        y_observed[i] = np.random.choice(other_classes)
    print(f"  • Samples: {X.shape[0]}")
    print(f"  • True classes: {len(np.unique(y_true))}")
    print(f"  • Noise rate: {noise_mask.mean():.1%}")
    # Get predicted probabilities
    print("\n2. Getting predicted probabilities...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    pred_probs = cross_val_predict(
        model, X_scaled, y_observed,
        cv=5, method='predict_proba', n_jobs=-1
    )
    print(f"  • Predicted probabilities shape: {pred_probs.shape}")
    # Initialize estimator
    print("\n3. Estimating joint distribution...")
    estimator = JointDistributionEstimator(random_state=42)
    # Estimate confident joint
    joint_dist = estimator.estimate_confident_joint(y_observed, pred_probs)
    # Estimate noise matrices
    print("\n4. Estimating noise matrices...")
    noise_results = estimator.estimate_noise_matrices(joint_dist)
    # Estimate latent distribution
    print("\n5. Estimating latent distribution...")
    py = estimator.estimate_latent_distribution(y_observed, joint_dist, method='cnt')
    # Analyze noise patterns
    print("\n6. Analyzing noise patterns...")
    noise_analysis = estimator.analyze_noise_patterns(noise_results['noise_matrix'])
    print("\nNoise Pattern Analysis:")
    print(noise_analysis[['true_class_name', 'correct_rate', 'self_noise_rate', 
                         'most_common_error_class_name', 'most_common_error_prob']].to_string(index=False))
    print("\n" + "=" * 60)
    print("JOINT DISTRIBUTION ESTIMATION COMPLETE!")
    print("=" * 60)
    return estimator, joint_dist, noise_results, noise_analysis
if __name__ == "__main__":
    estimator, joint_dist, noise_results, noise_analysis = main()

