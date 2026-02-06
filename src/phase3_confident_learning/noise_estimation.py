"""
Phase 3: Confident Learning Module (Core) - Updated for CleanLab API
Goal: Implement main noise estimation algorithm using CleanLab
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class ConfidentLearningNoiseEstimator:
    """Main noise estimation using CleanLab's Confident Learning."""
    def __init__(self, 
                 cv_n_folds: int = 5,
                 prune_method: str = 'prune_by_noise_rate',
                 random_state: int = 42):
        """
        Initialize confident learning estimator.
        Args:
            cv_n_folds: Number of cross-validation folds
            prune_method: Method for pruning label errors ('prune_by_noise_rate', 
                         'prune_by_class', 'both')
            random_state: Random seed for reproducibility
        """
        self.cv_n_folds = cv_n_folds
        self.prune_method = prune_method
        self.random_state = random_state
        # Results storage
        self.joint_distribution = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.label_errors = None
        self.label_quality_scores = None
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            clf=None) -> Dict[str, Any]:
        """
        Fit confident learning model to estimate label noise.
        Args:
            X: Feature matrix
            y: Noisy labels
            clf: Optional classifier (if None, will use default)
        Returns:
            Dictionary with noise estimation results
        """
        try:
            import cleanlab
            from cleanlab.filter import find_label_issues
            from cleanlab.rank import get_label_quality_scores
        except ImportError as e:
            print(f"Error: CleanLab not installed. Install with: pip install cleanlab")
            raise e
        print("=" * 60)
        print("CONFIDENT LEARNING NOISE ESTIMATION")
        print("=" * 60)
        print(f"\n1. Estimating label noise with CleanLab...")
        print(f"   • Samples: {len(y):,}")
        print(f"   • Classes: {len(np.unique(y))}")
        print(f"   • CV folds: {self.cv_n_folds}")
        print(f"   • Prune method: {self.prune_method}")
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
        # Step 2: Find label issues using CleanLab
        print("\n3. Finding label issues...")
        try:
            # Try the new API first
            label_issues_mask = find_label_issues(
                labels=y,
                pred_probs=pred_probs,
                return_indices_ranked_by='self_confidence',
                filter_by=self.prune_method,
                n_jobs=-1
            )
        except Exception as e:
            print(f"  ⚠ CleanLab API error: {e}")
            print("  Falling back to basic method...")
            # Fallback: use simple confidence threshold
            label_issues_mask = self._find_label_issues_fallback(y, pred_probs)
        n_label_issues = label_issues_mask.sum()
        label_issues_rate = n_label_issues / len(y)
        print(f"   • Found {n_label_issues} potential label errors ({label_issues_rate:.1%})")
        # Step 3: Compute confident joint and noise matrices
        print("\n4. Estimating noise statistics...")
        try:
            # Try to compute confident joint
            from cleanlab.count import compute_confident_joint
            confident_joint = compute_confident_joint(
                labels=y,
                pred_probs=pred_probs,
            )
            # Estimate noise matrix
            noise_matrix = self._estimate_noise_matrix_from_joint(confident_joint)
        except Exception as e:
            print(f"  ⚠ Could not compute confident joint: {e}")
            print("  Using simplified estimation...")
            confident_joint = self._estimate_simple_joint(y, pred_probs)
            noise_matrix = self._estimate_noise_matrix_from_joint(confident_joint)
        # Estimate inverse noise matrix
        if noise_matrix is not None:
            inverse_noise_matrix = self._estimate_inverse_noise_matrix(noise_matrix)
        else:
            inverse_noise_matrix = None
        # Step 4: Compute label quality scores
        print("\n5. Computing label quality scores...")
        try:
            label_quality_scores = get_label_quality_scores(
                labels=y,
                pred_probs=pred_probs
            )
        except:
            print("  ⚠ Using fallback quality scores...")
            label_quality_scores = self._compute_label_quality_scores_fallback(y, pred_probs)
        # Step 5: Rank label errors by confidence
        print("\n6. Ranking label errors...")
        ranked_label_errors = self._rank_label_errors(
            labels=y,
            pred_probs=pred_probs,
            label_issues_mask=label_issues_mask,
            label_quality_scores=label_quality_scores
        )
        # Step 6: Class-level analysis
        print("\n7. Performing class-level analysis...")
        class_analysis = self._analyze_by_class(
            labels=y,
            label_issues_mask=label_issues_mask,
            noise_matrix=noise_matrix
        )
        # Store results
        self.joint_distribution = confident_joint
        self.noise_matrix = noise_matrix
        self.inverse_noise_matrix = inverse_noise_matrix
        self.label_errors = label_issues_mask
        self.label_quality_scores = label_quality_scores
        # Compile comprehensive results
        results = {
            'confident_joint': confident_joint,
            'noise_matrix': noise_matrix,
            'inverse_noise_matrix': inverse_noise_matrix,
            'label_errors_mask': label_issues_mask,
            'label_quality_scores': label_quality_scores,
            'ranked_label_errors': ranked_label_errors,
            'class_analysis': class_analysis,
            'n_label_errors': n_label_issues,
            'label_error_rate': label_issues_rate,
            'n_samples': len(y),
            'n_classes': len(np.unique(y)),
            'avg_label_quality': label_quality_scores.mean() if label_quality_scores is not None else 0,
            'median_label_quality': np.median(label_quality_scores) if label_quality_scores is not None else 0
        }
        print(f"\n" + "=" * 60)
        print("CONFIDENT LEARNING COMPLETE")
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
    def _find_label_issues_fallback(self,
                                   labels: np.ndarray,
                                   pred_probs: np.ndarray,
                                   threshold: float = 0.5) -> np.ndarray:
        """Fallback method for finding label issues."""
        # Simple method: labels where predicted probability < threshold
        n_samples = len(labels)
        label_confidence = pred_probs[np.arange(n_samples), labels]
        label_issues = label_confidence < threshold
        return label_issues
    def _estimate_simple_joint(self,
                              labels: np.ndarray,
                              pred_probs: np.ndarray) -> np.ndarray:
        """Simple estimation of joint distribution."""
        n_classes = pred_probs.shape[1]
        joint = np.zeros((n_classes, n_classes))
        for i, label in enumerate(labels):
            pred_label = np.argmax(pred_probs[i])
            joint[pred_label, label] += 1
        # Normalize
        if joint.sum() > 0:
            joint = joint / joint.sum()
        return joint
    def _estimate_noise_matrix_from_joint(self, joint: np.ndarray) -> np.ndarray:
        """Estimate noise matrix from joint distribution."""
        if joint is None or joint.sum() == 0:
            return None
        # P(s|y) = P(y,s) / P(y)
        py = joint.sum(axis=1)  # P(y)
        py_safe = np.where(py == 0, 1e-10, py)
        noise_matrix = joint / py_safe[:, None]
        # Normalize rows
        noise_matrix = noise_matrix / noise_matrix.sum(axis=1, keepdims=True)
        return noise_matrix
    def _estimate_inverse_noise_matrix(self, noise_matrix: np.ndarray) -> np.ndarray:
        """Estimate inverse noise matrix from noise matrix."""
        if noise_matrix is None:
            return None
        # P(y|s) = P(s|y) * P(y) / P(s)
        n_classes = noise_matrix.shape[0]
        # Assume uniform prior for simplicity
        py = np.ones(n_classes) / n_classes
        ps = noise_matrix.T @ py
        ps_safe = np.where(ps == 0, 1e-10, ps)
        inverse_noise_matrix = (noise_matrix * py[:, None]) / ps_safe[None, :]
        inverse_noise_matrix = inverse_noise_matrix.T
        # Normalize columns
        inverse_noise_matrix = inverse_noise_matrix / inverse_noise_matrix.sum(axis=0, keepdims=True)
        return inverse_noise_matrix
    def _compute_label_quality_scores_fallback(self,
                                              labels: np.ndarray,
                                              pred_probs: np.ndarray) -> np.ndarray:
        """Fallback method for computing label quality scores."""
        # Use self-confidence as quality score
        n_samples = len(labels)
        quality_scores = pred_probs[np.arange(n_samples), labels]
        return quality_scores
    def _rank_label_errors(self,
                          labels: np.ndarray,
                          pred_probs: np.ndarray,
                          label_issues_mask: np.ndarray,
                          label_quality_scores: np.ndarray) -> pd.DataFrame:
        """Rank label errors by confidence."""
        # Get indices of label errors
        error_indices = np.where(label_issues_mask)[0]
        if len(error_indices) == 0:
            return pd.DataFrame()
        # Get predicted class (highest probability other than current label)
        predicted_labels = np.argmax(pred_probs, axis=1)
        # For each error, get alternative class with highest probability
        ranked_data = []
        for idx in error_indices:
            current_label = labels[idx]
            pred_probs_sample = pred_probs[idx]
            # Get top predicted classes (excluding current label)
            top_classes = np.argsort(pred_probs_sample)[::-1]
            suggested_label = top_classes[0] if top_classes[0] != current_label else top_classes[1]
            suggested_confidence = pred_probs_sample[suggested_label]
            ranked_data.append({
                'sample_index': idx,
                'current_label': current_label,
                'suggested_label': suggested_label,
                'suggested_confidence': suggested_confidence,
                'self_confidence': pred_probs_sample[current_label],
                'label_quality_score': label_quality_scores[idx] if label_quality_scores is not None else pred_probs_sample[current_label],
                'margin': suggested_confidence - pred_probs_sample[current_label]
            })
        # Create DataFrame and sort by margin (largest margin = most confident error)
        df_ranked = pd.DataFrame(ranked_data)
        df_ranked = df_ranked.sort_values('margin', ascending=False)
        df_ranked['rank'] = range(1, len(df_ranked) + 1)
        return df_ranked
    def _analyze_by_class(self,
                         labels: np.ndarray,
                         label_issues_mask: np.ndarray,
                         noise_matrix: np.ndarray) -> pd.DataFrame:
        """Analyze label errors by class."""
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        class_stats = []
        for i, class_label in enumerate(unique_classes):
            class_mask = labels == class_label
            n_class_samples = class_mask.sum()
            if n_class_samples > 0:
                n_class_errors = label_issues_mask[class_mask].sum()
                error_rate = n_class_errors / n_class_samples if n_class_samples > 0 else 0
                # Get confusion patterns from noise matrix
                if noise_matrix is not None and i < noise_matrix.shape[0]:
                    # Most common error for this class
                    row = noise_matrix[i]
                    if row.sum() > 0:
                        most_common_error = np.argsort(row)[-2] if n_classes > 1 else 0  # Second highest (first is self)
                        error_fraction = row[most_common_error] / row.sum() if row.sum() > 0 else 0
                    else:
                        most_common_error = -1
                        error_fraction = 0
                else:
                    most_common_error = -1
                    error_fraction = 0
                class_stats.append({
                    'class': class_label,
                    'n_samples': n_class_samples,
                    'n_label_errors': n_class_errors,
                    'label_error_rate': error_rate,
                    'most_common_error_class': most_common_error,
                    'error_fraction_to_common': error_fraction
                })
        return pd.DataFrame(class_stats)
    def get_cleaned_labels(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          clf=None,
                          threshold: float = 0.5) -> np.ndarray:
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
        if self.label_errors is None or self.label_quality_scores is None:
            self.fit(X, y, clf)
        # Get predicted probabilities for suggested labels
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
    def evaluate_on_synthetic_noise(self,
                                   X: np.ndarray,
                                   y_clean: np.ndarray,
                                   y_noisy: np.ndarray,
                                   clf=None) -> Dict[str, float]:
        """
        Evaluate confident learning on synthetic noise.
        Args:
            X: Feature matrix
            y_clean: Clean labels (ground truth)
            y_noisy: Noisy labels
            clf: Classifier
        Returns:
            Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION ON SYNTHETIC NOISE")
        print("=" * 60)
        # Fit on noisy labels
        results = self.fit(X, y_noisy, clf)
        # True noisy samples
        true_noisy_mask = y_noisy != y_clean
        n_true_noisy = true_noisy_mask.sum()
        true_noise_rate = n_true_noisy / len(y_noisy)
        # Detected label errors
        detected_mask = results['label_errors_mask']
        n_detected = detected_mask.sum()
        # Calculate metrics
        tp = np.sum(true_noisy_mask & detected_mask)  # True Positives
        fp = np.sum(~true_noisy_mask & detected_mask)  # False Positives
        fn = np.sum(true_noisy_mask & ~detected_mask)  # False Negatives
        tn = np.sum(~true_noisy_mask & ~detected_mask)  # True Negatives
        # Metrics
        accuracy = (tp + tn) / len(y_noisy) if len(y_noisy) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics = {
            'true_noise_rate': true_noise_rate,
            'detected_noise_rate': n_detected / len(y_noisy),
            'detection_accuracy': accuracy,
            'detection_precision': precision,
            'detection_recall': recall,
            'detection_f1': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'n_true_noisy': int(n_true_noisy),
            'n_detected': int(n_detected)
        }
        print(f"\nEvaluation Results:")
        print(f"  • True noise rate: {true_noise_rate:.3f} ({n_true_noisy:,} samples)")
        print(f"  • Detected noise rate: {metrics['detected_noise_rate']:.3f} ({n_detected:,} samples)")
        print(f"  • Detection accuracy: {accuracy:.3f}")
        print(f"  • Detection precision: {precision:.3f}")
        print(f"  • Detection recall: {recall:.3f}")
        print(f"  • Detection F1-score: {f1:.3f}")
        return metrics
def main():
    """Test the ConfidentLearningNoiseEstimator."""
    print("=" * 60)
    print("Phase 3: Confident Learning Module (Test)")
    print("=" * 60)
    # Create a test dataset with known noise
    from sklearn.datasets import make_classification
    print("\n1. Creating test dataset with synthetic noise...")
    X, y_clean = make_classification(
        n_samples=500,  # Smaller for testing
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
    print("\n2. Initializing Confident Learning estimator...")
    estimator = ConfidentLearningNoiseEstimator(
        cv_n_folds=5,
        prune_method='prune_by_noise_rate',
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
                                        'label_error_rate', 'most_common_error_class']].to_string(index=False))
    # Evaluate on synthetic noise
    print("\n6. Evaluating detection performance...")
    evaluation = estimator.evaluate_on_synthetic_noise(X, y_clean, y_noisy)
    print("\n" + "=" * 60)
    print("CONFIDENT LEARNING TEST COMPLETE!")
    print("=" * 60)
    return estimator, results, evaluation
if __name__ == "__main__":
    estimator, results, evaluation = main()
