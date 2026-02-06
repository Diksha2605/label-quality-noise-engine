"""
Phase 2: Noise Detection - Baseline
Goal: Detect suspicious labels using simple disagreement analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class BaselineNoiseDetector:
    """Baseline noise detection using classifier disagreement."""
    def __init__(self, 
                 model_type: str = 'logistic',
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize baseline detector.
        Args:
            model_type: Type of classifier ('logistic', 'random_forest', 'svm')
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {}
    def _get_model(self):
        """Get the appropriate classifier model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                probability=True,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    def compute_disagreement_scores(self, 
                                   X: np.ndarray, 
                                   y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute disagreement scores using K-fold cross-validation.
        Args:
            X: Feature matrix
            y: Labels
        Returns:
            Dictionary with disagreement metrics for each sample
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        print(f"Computing disagreement scores using {self.n_folds}-fold CV...")
        n_samples = len(y)
        # Initialize arrays to store results
        disagreement_scores = np.zeros(n_samples)
        predicted_labels = np.zeros(n_samples, dtype=int)
        prediction_confidences = np.zeros(n_samples)
        fold_indices = np.zeros(n_samples, dtype=int)
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, 
                             shuffle=True, 
                             random_state=self.random_state)
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        fold_num = 0
        for train_idx, val_idx in skf.split(X_scaled, y):
            fold_num += 1
            print(f"  Processing fold {fold_num}/{self.n_folds}...")
            # Split data
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train = y[train_idx]
            # Train model
            model = self._get_model()
            model.fit(X_train, y_train)
            # Predict on validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            # Store predictions
            predicted_labels[val_idx] = y_pred
            fold_indices[val_idx] = fold_num
            # Calculate confidence (max probability)
            prediction_confidences[val_idx] = np.max(y_pred_proba, axis=1)
            # Calculate disagreement (1 if prediction != label, 0 otherwise)
            disagreement_scores[val_idx] = (y_pred != y[val_idx]).astype(float)
        # Calculate additional metrics
        confidence_scores = prediction_confidences
        # Identify suspicious samples (disagreement = 1)
        suspicious_mask = disagreement_scores == 1
        # Create comprehensive results dictionary
        results = {
            'disagreement_scores': disagreement_scores,
            'predicted_labels': predicted_labels,
            'confidence_scores': confidence_scores,
            'suspicious_mask': suspicious_mask,
            'fold_indices': fold_indices,
            'n_suspicious': suspicious_mask.sum(),
            'disagreement_rate': suspicious_mask.mean(),
            'avg_confidence': confidence_scores.mean(),
            'avg_confidence_suspicious': confidence_scores[suspicious_mask].mean() if suspicious_mask.any() else 0,
            'avg_confidence_correct': confidence_scores[~suspicious_mask].mean() if (~suspicious_mask).any() else 0
        }
        print(f"\nDisagreement analysis complete:")
        print(f"  • Total samples: {n_samples}")
        print(f"  • Suspicious labels: {results['n_suspicious']} ({results['disagreement_rate']:.1%})")
        print(f"  • Average confidence: {results['avg_confidence']:.3f}")
        print(f"  • Confidence (suspicious): {results['avg_confidence_suspicious']:.3f}")
        print(f"  • Confidence (correct): {results['avg_confidence_correct']:.3f}")
        return results
    def rank_suspicious_samples(self, 
                               results: Dict[str, np.ndarray],
                               top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Rank suspicious samples by confidence score.
        Args:
            results: Results dictionary from compute_disagreement_scores
            top_k: Number of top suspicious samples to return (None for all)
        Returns:
            DataFrame with ranked suspicious samples
        """
        # Get indices of suspicious samples
        suspicious_indices = np.where(results['suspicious_mask'])[0]
        if len(suspicious_indices) == 0:
            print("No suspicious samples found.")
            return pd.DataFrame()
        # Prepare data for ranking
        ranking_data = []
        for idx in suspicious_indices:
            ranking_data.append({
                'sample_index': idx,
                'true_label': -1,  # Will be filled by caller
                'predicted_label': results['predicted_labels'][idx],
                'confidence': results['confidence_scores'][idx],
                'disagreement_score': results['disagreement_scores'][idx],
                'fold': results['fold_indices'][idx]
            })
        # Create DataFrame
        df_ranking = pd.DataFrame(ranking_data)
        # Sort by confidence (ascending - lower confidence = more suspicious)
        df_ranking = df_ranking.sort_values('confidence', ascending=True)
        # Add rank
        df_ranking['rank'] = range(1, len(df_ranking) + 1)
        # Select top-k if specified
        if top_k is not None:
            df_ranking = df_ranking.head(top_k)
        return df_ranking
    def analyze_by_class(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Analyze disagreement by class.
        Args:
            X: Feature matrix (not used but kept for API consistency)
            y: True labels
            results: Results dictionary
        Returns:
            DataFrame with class-level statistics
        """
        unique_classes = np.unique(y)
        class_stats = []
        for class_label in unique_classes:
            class_mask = y == class_label
            n_class_samples = class_mask.sum()
            if n_class_samples > 0:
                class_disagreement = results['disagreement_scores'][class_mask]
                class_confidence = results['confidence_scores'][class_mask]
                class_stats.append({
                    'class': class_label,
                    'n_samples': n_class_samples,
                    'n_suspicious': class_disagreement.sum(),
                    'disagreement_rate': class_disagreement.mean(),
                    'avg_confidence': class_confidence.mean(),
                    'avg_confidence_suspicious': class_confidence[class_disagreement == 1].mean() 
                                                if (class_disagreement == 1).any() else 0,
                    'avg_confidence_correct': class_confidence[class_disagreement == 0].mean() 
                                             if (class_disagreement == 0).any() else 0
                })
        df_class_stats = pd.DataFrame(class_stats)
        # Sort by disagreement rate (descending)
        df_class_stats = df_class_stats.sort_values('disagreement_rate', ascending=False)
        return df_class_stats
    def evaluate_detection_accuracy(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   y_true_clean: np.ndarray,
                                   results: Dict[str, np.ndarray],
                                   threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate detection accuracy if ground truth is available.
        Args:
            X: Feature matrix
            y: Noisy labels (to be evaluated)
            y_true_clean: Ground truth clean labels
            results: Results from compute_disagreement_scores
            threshold: Threshold for considering a sample suspicious
        Returns:
            Dictionary with evaluation metrics
        """
        # True noisy samples (where y != y_true_clean)
        true_noisy_mask = y != y_true_clean
        # Predicted suspicious samples
        pred_suspicious_mask = results['suspicious_mask']
        # Calculate metrics
        n_true_noisy = true_noisy_mask.sum()
        n_pred_suspicious = pred_suspicious_mask.sum()
        # True Positives: Correctly identified as noisy
        tp = np.sum(true_noisy_mask & pred_suspicious_mask)
        # False Positives: Incorrectly identified as noisy
        fp = np.sum(~true_noisy_mask & pred_suspicious_mask)
        # False Negatives: Missed noisy samples
        fn = np.sum(true_noisy_mask & ~pred_suspicious_mask)
        # True Negatives: Correctly identified as clean
        tn = np.sum(~true_noisy_mask & ~pred_suspicious_mask)
        # Calculate metrics
        accuracy = (tp + tn) / len(y) if len(y) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'true_noisy_rate': n_true_noisy / len(y),
            'detected_noisy_rate': n_pred_suspicious / len(y),
            'detection_difference': abs(n_true_noisy - n_pred_suspicious) / len(y)
        }
        print(f"\nDetection Evaluation:")
        print(f"  • True noisy samples: {n_true_noisy} ({metrics['true_noisy_rate']:.1%})")
        print(f"  • Detected suspicious: {n_pred_suspicious} ({metrics['detected_noisy_rate']:.1%})")
        print(f"  • Accuracy: {accuracy:.3f}")
        print(f"  • Precision: {precision:.3f}")
        print(f"  • Recall: {recall:.3f}")
        print(f"  • F1-Score: {f1:.3f}")
        return metrics
def main():
    """Test the BaselineNoiseDetector."""
    print("=" * 60)
    print("Phase 2: Baseline Noise Detection")
    print("=" * 60)
    # Create a test dataset with known noise
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    print("\n1. Creating test dataset with synthetic noise...")
    X, y_clean = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42
    )
    # Inject some noise
    np.random.seed(42)
    noise_indices = np.random.choice(len(y_clean), size=100, replace=False)
    y_noisy = y_clean.copy()
    for idx in noise_indices:
        # Change to a different class
        other_classes = [c for c in np.unique(y_clean) if c != y_clean[idx]]
        y_noisy[idx] = np.random.choice(other_classes)
    print(f"  • Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  • Injected noise: {len(noise_indices)} samples ({len(noise_indices)/len(y_clean):.1%})")
    # Initialize detector
    print("\n2. Initializing noise detector...")
    detector = BaselineNoiseDetector(
        model_type='logistic',
        n_folds=5,
        random_state=42
    )
    # Compute disagreement scores
    print("\n3. Computing disagreement scores...")
    results = detector.compute_disagreement_scores(X, y_noisy)
    # Rank suspicious samples
    print("\n4. Ranking suspicious samples...")
    suspicious_ranking = detector.rank_suspicious_samples(results, top_k=10)
    if not suspicious_ranking.empty:
        print("\nTop 10 most suspicious samples:")
        print(suspicious_ranking[['sample_index', 'predicted_label', 'confidence', 'fold']].to_string(index=False))
    # Analyze by class
    print("\n5. Analyzing by class...")
    class_analysis = detector.analyze_by_class(X, y_noisy, results)
    print("\nClass-level disagreement analysis:")
    print(class_analysis[['class', 'n_samples', 'n_suspicious', 'disagreement_rate', 'avg_confidence']].to_string(index=False))
    # Evaluate detection accuracy
    print("\n6. Evaluating detection accuracy...")
    evaluation = detector.evaluate_detection_accuracy(
        X, y_noisy, y_clean, results
    )
    print("\n" + "=" * 60)
    print("Baseline Noise Detection Complete!")
    print("=" * 60)
    return detector, results, suspicious_ranking, class_analysis, evaluation
if __name__ == "__main__":
    detector, results, suspicious_ranking, class_analysis, evaluation = main()
