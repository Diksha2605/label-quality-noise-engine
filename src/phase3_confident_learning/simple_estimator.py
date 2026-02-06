"""
Phase 3: Simplified Confident Learning Module
Goal: Implement noise estimation without CleanLab dependency
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class SimpleNoiseEstimator:
    """Simple noise estimation without CleanLab dependency."""
    def __init__(self, 
                 cv_n_folds: int = 5,
                 confidence_threshold: float = 0.5,
                 random_state: int = 42):
        """
        Initialize simple noise estimator.
        Args:
            cv_n_folds: Number of cross-validation folds
            confidence_threshold: Threshold for identifying label errors
            random_state: Random seed for reproducibility
        """
        self.cv_n_folds = cv_n_folds
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        # Results storage
        self.label_errors = None
        self.label_quality_scores = None
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            clf=None) -> Dict[str, Any]:
        """
        Fit noise estimation model.
        Args:
            X: Feature matrix
            y: Noisy labels
            clf: Optional classifier (if None, will use default)
        Returns:
            Dictionary with noise estimation results
        """
        print("=" * 60)
        print("SIMPLE NOISE ESTIMATION")
        print("=" * 60)
        print(f"\n1. Estimating label noise...")
        print(f"   • Samples: {len(y):,}")
        print(f"   • Classes: {len(np.unique(y))}")
        print(f"   • CV folds: {self.cv_n_folds}")
        print(f"   • Confidence threshold: {self.confidence_threshold}")
        # If no classifier provided, use default
        if clf is None:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        # Step 1: Get predicted probabilities
        print("\n2. Computing predicted probabilities...")
        pred_probs = self._get_predicted_probabilities(X, y, clf)
        # Step 2: Find label issues based on confidence
        print("\n3. Finding label issues...")
        label_issues_mask = self._find_label_issues(y, pred_probs)
        n_label_issues = label_issues_mask.sum()
        label_issues_rate = n_label_issues / len(y)
        print(f"   • Found {n_label_issues} potential label errors ({label_issues_rate:.1%})")
        # Step 3: Compute label quality scores (self-confidence)
        print("\n4. Computing label quality scores...")
        label_quality_scores = pred_probs[np.arange(len(y)), y]
        # Step 4: Rank label errors by confidence
        print("\n5. Ranking label errors...")
        ranked_label_errors = self._rank_label_errors(
            labels=y,
            pred_probs=pred_probs,
            label_issues_mask=label_issues_mask
        )
        # Step 5: Class-level analysis
        print("\n6. Performing class-level analysis...")
        class_analysis = self._analyze_by_class(
            labels=y,
            label_issues_mask=label_issues_mask
        )
        # Step 6: Estimate noise matrix
        print("\n7. Estimating noise matrix...")
        noise_matrix = self._estimate_noise_matrix(y, pred_probs)
        # Store results
        self.label_errors = label_issues_mask
        self.label_quality_scores = label_quality_scores
        # Compile comprehensive results
        results = {
            'noise_matrix': noise_matrix,
            'label_errors_mask': label_issues_mask,
            'label_quality_scores': label_quality_scores,
            'ranked_label_errors': ranked_label_errors,
            'class_analysis': class_analysis,
            'n_label_errors': n_label_issues,
            'label_error_rate': float(label_issues_rate),
            'n_samples': len(y),
            'n_classes': len(np.unique(y)),
            'avg_label_quality': float(label_quality_scores.mean()),
            'median_label_quality': float(np.median(label_quality_scores))
        }
        print(f"\n" + "=" * 60)
        print("NOISE ESTIMATION COMPLETE")
        print("=" * 60)
        print(f"\nResults:")
        print(f"  • Label errors found: {n_label_issues:,} ({label_issues_rate:.1%})")
        print(f"  • Average label quality: {results['avg_label_quality']:.3f}")
        print(f"  • Median label quality: {results['median_label_quality']:.3f}")
        return results
    def _get_predicted_probabilities(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray,
                                    clf) -> np.ndarray:
        """Get cross-validated predicted probabilities."""
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import StandardScaler
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Get cross-validated predicted probabilities
        pred_probs = cross_val_predict(
            clf, X_scaled, y,
            cv=self.cv_n_folds,
            method='predict_proba',
            n_jobs=-1
        )
        return pred_probs
    def _find_label_issues(self,
                          labels: np.ndarray,
                          pred_probs: np.ndarray) -> np.ndarray:
        """Find label issues based on confidence threshold."""
        n_samples = len(labels)
        # Method 1: Low self-confidence
        self_confidence = pred_probs[np.arange(n_samples), labels]
        low_confidence_errors = self_confidence < self.confidence_threshold
        # Method 2: Predicted label differs from observed label
        predicted_labels = np.argmax(pred_probs, axis=1)
        prediction_errors = predicted_labels != labels
        # Combine both methods
        label_issues = low_confidence_errors | prediction_errors
        return label_issues
    def _rank_label_errors(self,
                          labels: np.ndarray,
                          pred_probs: np.ndarray,
                          label_issues_mask: np.ndarray) -> pd.DataFrame:
        """Rank label errors by confidence margin."""
        # Get indices of label errors
        error_indices = np.where(label_issues_mask)[0]
        if len(error_indices) == 0:
            return pd.DataFrame()
        ranked_data = []
        for idx in error_indices:
            current_label = labels[idx]
            pred_probs_sample = pred_probs[idx]
            # Get self-confidence
            self_confidence = pred_probs_sample[current_label]
            # Get suggested label (highest probability other than current)
            top_classes = np.argsort(pred_probs_sample)[::-1]
            suggested_label = top_classes[0] if top_classes[0] != current_label else top_classes[1]
            suggested_confidence = pred_probs_sample[suggested_label]
            # Calculate confidence margin
            margin = suggested_confidence - self_confidence
            ranked_data.append({
                'sample_index': idx,
                'current_label': current_label,
                'suggested_label': suggested_label,
                'suggested_confidence': suggested_confidence,
                'self_confidence': self_confidence,
                'margin': margin,
                'label_quality': self_confidence
            })
        # Create DataFrame and sort by margin (largest margin = most confident error)
        df_ranked = pd.DataFrame(ranked_data)
        df_ranked = df_ranked.sort_values('margin', ascending=False)
        df_ranked['rank'] = range(1, len(df_ranked) + 1)
        return df_ranked
    def _analyze_by_class(self,
                         labels: np.ndarray,
                         label_issues_mask: np.ndarray) -> pd.DataFrame:
        """Analyze label errors by class."""
        unique_classes = np.unique(labels)
        class_stats = []
        for class_label in unique_classes:
            class_mask = labels == class_label
            n_class_samples = class_mask.sum()
            if n_class_samples > 0:
                n_class_errors = label_issues_mask[class_mask].sum()
                error_rate = n_class_errors / n_class_samples
                class_stats.append({
                    'class': class_label,
                    'n_samples': n_class_samples,
                    'n_label_errors': n_class_errors,
                    'label_error_rate': error_rate
                })
        df_class_stats = pd.DataFrame(class_stats)
        # Sort by error rate (descending)
        df_class_stats = df_class_stats.sort_values('label_error_rate', ascending=False)
        return df_class_stats
    def _estimate_noise_matrix(self,
                              labels: np.ndarray,
                              pred_probs: np.ndarray) -> np.ndarray:
        """Estimate noise matrix from predictions."""
        n_classes = pred_probs.shape[1]
        confusion_matrix = np.zeros((n_classes, n_classes))
        for i, true_label in enumerate(labels):
            pred_label = np.argmax(pred_probs[i])
            confusion_matrix[pred_label, true_label] += 1
        # Normalize to get noise matrix P(s|y)
        if confusion_matrix.sum() > 0:
            # Normalize each column (true label) to sum to 1
            column_sums = confusion_matrix.sum(axis=0)
            column_sums_safe = np.where(column_sums == 0, 1, column_sums)
            noise_matrix = confusion_matrix / column_sums_safe
        else:
            noise_matrix = np.zeros((n_classes, n_classes))
        return noise_matrix
    def get_cleaned_labels(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          clf=None,
                          threshold: float = 0.6) -> np.ndarray:
        """
        Get cleaned labels by correcting identified errors.
        Args:
            X: Feature matrix
            y: Original labels
            clf: Classifier (if None, uses default)
            threshold: Confidence threshold for correction
        Returns:
            Cleaned labels
        """
        if self.label_errors is None:
            self.fit(X, y, clf)
        # Get predicted probabilities
        if clf is None:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        pred_probs = self._get_predicted_probabilities(X, y, clf)
        # Create cleaned labels
        cleaned_labels = y.copy()
        # Get error indices
        error_indices = np.where(self.label_errors)[0]
        for idx in error_indices:
            current_label = y[idx]
            pred_probs_sample = pred_probs[idx]
            # Get suggested label (highest probability other than current)
            top_classes = np.argsort(pred_probs_sample)[::-1]
            suggested_label = top_classes[0] if top_classes[0] != current_label else top_classes[1]
            suggested_confidence = pred_probs_sample[suggested_label]
            # Only correct if confidence is above threshold
            if suggested_confidence >= threshold:
                cleaned_labels[idx] = suggested_label
        n_corrected = (cleaned_labels != y).sum()
        print(f"\nCorrected {n_corrected} labels ({n_corrected/len(y):.1%})")
        return cleaned_labels
def main():
    """Test the SimpleNoiseEstimator."""
    print("=" * 60)
    print("Phase 3: Simple Noise Estimation (No CleanLab Required)")
    print("=" * 60)
    # Create a test dataset with known noise
    from sklearn.datasets import make_classification
    print("\n1. Creating test dataset with synthetic noise...")
    X, y_clean = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42
    )
    # Inject noise (15% of samples)
    np.random.seed(42)
    noise_mask = np.random.rand(len(y_clean)) < 0.15
    y_noisy = y_clean.copy()
    for i in np.where(noise_mask)[0]:
        other_classes = [c for c in np.unique(y_clean) if c != y_clean[i]]
        y_noisy[i] = np.random.choice(other_classes)
    print(f"  • Samples: {X.shape[0]}")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Classes: {len(np.unique(y_clean))}")
    print(f"  • Injected noise: {noise_mask.sum()} samples ({noise_mask.mean():.1%})")
    # Initialize estimator
    print("\n2. Initializing Simple Noise Estimator...")
    estimator = SimpleNoiseEstimator(
        cv_n_folds=5,
        confidence_threshold=0.5,
        random_state=42
    )
    # Fit on noisy data
    print("\n3. Fitting estimator...")
    results = estimator.fit(X, y_noisy)
    # Show ranked label errors
    print("\n4. Top 10 most confident label errors:")
    if not results['ranked_label_errors'].empty:
        top_errors = results['ranked_label_errors'].head(10)
        print(top_errors[['sample_index', 'current_label', 'suggested_label', 
                         'suggested_confidence', 'self_confidence', 'margin']].to_string(index=False))
    # Show class analysis
    print("\n5. Class-level analysis:")
    if not results['class_analysis'].empty:
        print(results['class_analysis'][['class', 'n_samples', 'n_label_errors', 
                                        'label_error_rate']].to_string(index=False))
    # Show noise matrix
    print("\n6. Estimated Noise Matrix:")
    noise_matrix = results['noise_matrix']
    if noise_matrix is not None:
        n_classes = noise_matrix.shape[0]
        for i in range(n_classes):
            row_str = "  ".join([f"{val:.3f}" for val in noise_matrix[i]])
            print(f"  Class {i} -> [{row_str}]")
    print("\n" + "=" * 60)
    print("SIMPLE NOISE ESTIMATION COMPLETE!")
    print("=" * 60)
    return estimator, results
if __name__ == "__main__":
    estimator, results = main()

