"""
Confidence scoring and label quality assessment for Phase 3.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class ConfidenceScorer:
    """Compute confidence scores and assess label quality."""
    def __init__(self, random_state: int = 42):
        """Initialize scorer."""
        self.random_state = random_state
        np.random.seed(random_state)
    def compute_label_quality_scores(self,
                                    labels: np.ndarray,
                                    pred_probs: np.ndarray,
                                    method: str = 'self_confidence') -> np.ndarray:
        """
        Compute label quality scores.
        Args:
            labels: Observed labels
            pred_probs: Predicted probabilities
            method: Scoring method ('self_confidence', 'normalized_margin', 'confidence_weighted')
        Returns:
            Label quality scores (higher = better label quality)
        """
        n_samples = len(labels)
        if method == 'self_confidence':
            # Self-confidence: probability assigned to the observed label
            scores = pred_probs[np.arange(n_samples), labels]
        elif method == 'normalized_margin':
            # Normalized margin: difference between top probability and second top
            sorted_probs = np.sort(pred_probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            # Check if observed label is the top prediction
            top_predictions = np.argmax(pred_probs, axis=1)
            is_top = (top_predictions == labels).astype(float)
            # Combine: margin if correct, negative margin if incorrect
            scores = np.where(is_top == 1, margins, -margins)
            # Normalize to [0, 1]
            scores = (scores + 1) / 2
        elif method == 'confidence_weighted':
            # Weighted combination of self-confidence and consistency
            self_confidence = pred_probs[np.arange(n_samples), labels]
            # Consistency: agreement with nearest neighbors (simplified)
            consistency = self._compute_consistency_scores(labels, pred_probs)
            # Weighted average
            scores = 0.7 * self_confidence + 0.3 * consistency
        else:
            raise ValueError(f"Unknown method: {method}")
        print(f"Computed label quality scores using '{method}' method:")
        print(f"  • Min score: {scores.min():.4f}")
        print(f"  • Max score: {scores.max():.4f}")
        print(f"  • Mean score: {scores.mean():.4f}")
        print(f"  • Median score: {np.median(scores):.4f}")
        return scores
    def _compute_consistency_scores(self,
                                   labels: np.ndarray,
                                   pred_probs: np.ndarray,
                                   k: int = 5) -> np.ndarray:
        """Compute consistency scores based on neighbor agreement."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        n_samples = len(labels)
        # Use predicted probabilities as features for nearest neighbors
        scaler = StandardScaler()
        features = scaler.fit_transform(pred_probs)
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features)
        distances, indices = nbrs.kneighbors(features)
        # Compute agreement with neighbors (excluding self)
        consistency_scores = np.zeros(n_samples)
        for i in range(n_samples):
            neighbor_indices = indices[i, 1:]  # Exclude self
            neighbor_labels = labels[neighbor_indices]
            # Agreement rate with neighbors
            agreement = np.mean(neighbor_labels == labels[i])
            consistency_scores[i] = agreement
        return consistency_scores
    def identify_label_errors(self,
                             labels: np.ndarray,
                             label_quality_scores: np.ndarray,
                             threshold_method: str = 'percentile',
                             threshold_value: float = 0.1) -> np.ndarray:
        """
        Identify label errors based on quality scores.
        Args:
            labels: Observed labels
            label_quality_scores: Quality scores (higher = better)
            threshold_method: 'percentile', 'absolute', or 'adaptive'
            threshold_value: Threshold value or percentile
        Returns:
            Boolean mask of label errors
        """
        n_samples = len(labels)
        if threshold_method == 'percentile':
            # Use percentile threshold
            threshold = np.percentile(label_quality_scores, threshold_value * 100)
            label_errors = label_quality_scores <= threshold
        elif threshold_method == 'absolute':
            # Use absolute threshold
            label_errors = label_quality_scores <= threshold_value
        elif threshold_method == 'adaptive':
            # Adaptive threshold based on class distribution
            unique_classes = np.unique(labels)
            label_errors = np.zeros(n_samples, dtype=bool)
            for class_label in unique_classes:
                class_mask = labels == class_label
                class_scores = label_quality_scores[class_mask]
                if len(class_scores) > 0:
                    # Use class-specific percentile
                    class_threshold = np.percentile(class_scores, threshold_value * 100)
                    class_errors = class_scores <= class_threshold
                    label_errors[class_mask] = class_errors
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
        n_errors = label_errors.sum()
        error_rate = n_errors / n_samples
        print(f"\nLabel Error Identification:")
        print(f"  • Method: {threshold_method}")
        print(f"  • Threshold value: {threshold_value}")
        print(f"  • Label errors identified: {n_errors} ({error_rate:.1%})")
        return label_errors
    def rank_label_errors(self,
                         labels: np.ndarray,
                         pred_probs: np.ndarray,
                         label_errors_mask: np.ndarray,
                         ranking_method: str = 'confidence_margin') -> pd.DataFrame:
        """
        Rank label errors by confidence.
        Args:
            labels: Observed labels
            pred_probs: Predicted probabilities
            label_errors_mask: Boolean mask of label errors
            ranking_method: 'confidence_margin', 'self_confidence', 'normalized_margin'
        Returns:
            DataFrame with ranked label errors
        """
        error_indices = np.where(label_errors_mask)[0]
        if len(error_indices) == 0:
            return pd.DataFrame()
        ranked_data = []
        for idx in error_indices:
            current_label = labels[idx]
            pred_prob = pred_probs[idx]
            # Get self-confidence (probability of observed label)
            self_confidence = pred_prob[current_label]
            # Get suggested label (highest probability other than current)
            sorted_indices = np.argsort(pred_prob)[::-1]
            suggested_label = sorted_indices[0] if sorted_indices[0] != current_label else sorted_indices[1]
            suggested_confidence = pred_prob[suggested_label]
            # Compute ranking score based on method
            if ranking_method == 'confidence_margin':
                # Margin between suggested and current confidence
                ranking_score = suggested_confidence - self_confidence
            elif ranking_method == 'self_confidence':
                # Lower self-confidence = more suspicious
                ranking_score = -self_confidence  # Negative so lower scores rank higher
            elif ranking_method == 'normalized_margin':
                # Normalized margin considering all classes
                sorted_probs = np.sort(pred_prob)[::-1]
                if len(sorted_probs) > 1:
                    margin = sorted_probs[0] - sorted_probs[1]
                    # If suggested is not top, use negative margin
                    if suggested_label != sorted_indices[0]:
                        margin = -margin
                    ranking_score = margin
                else:
                    ranking_score = suggested_confidence - self_confidence
            else:
                raise ValueError(f"Unknown ranking method: {ranking_method}")
            ranked_data.append({
                'sample_index': idx,
                'current_label': current_label,
                'suggested_label': suggested_label,
                'self_confidence': self_confidence,
                'suggested_confidence': suggested_confidence,
                'confidence_margin': suggested_confidence - self_confidence,
                'ranking_score': ranking_score
            })
        # Create DataFrame and sort
        df_ranked = pd.DataFrame(ranked_data)
        if ranking_method == 'self_confidence':
            df_ranked = df_ranked.sort_values('self_confidence', ascending=True)
        else:
            df_ranked = df_ranked.sort_values('ranking_score', ascending=False)
        df_ranked['rank'] = range(1, len(df_ranked) + 1)
        return df_ranked
    def assess_dataset_quality(self,
                              labels: np.ndarray,
                              label_quality_scores: np.ndarray,
                              label_errors_mask: np.ndarray) -> Dict[str, Any]:
        """
        Assess overall dataset quality.
        Args:
            labels: Observed labels
            label_quality_scores: Quality scores
            label_errors_mask: Label errors mask
        Returns:
            Dictionary with quality assessment metrics
        """
        n_samples = len(labels)
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        # Overall metrics
        avg_quality = label_quality_scores.mean()
        median_quality = np.median(label_quality_scores)
        n_errors = label_errors_mask.sum()
        error_rate = n_errors / n_samples
        # Class-level metrics
        class_metrics = []
        for class_label in unique_classes:
            class_mask = labels == class_label
            n_class_samples = class_mask.sum()
            if n_class_samples > 0:
                class_scores = label_quality_scores[class_mask]
                class_errors = label_errors_mask[class_mask]
                class_metrics.append({
                    'class': class_label,
                    'n_samples': n_class_samples,
                    'avg_quality': class_scores.mean(),
                    'median_quality': np.median(class_scores),
                    'n_errors': class_errors.sum(),
                    'error_rate': class_errors.mean(),
                    'quality_10th_percentile': np.percentile(class_scores, 10),
                    'quality_90th_percentile': np.percentile(class_scores, 90)
                })
        # Compile assessment
        assessment = {
            'overall': {
                'n_samples': n_samples,
                'n_classes': n_classes,
                'avg_label_quality': avg_quality,
                'median_label_quality': median_quality,
                'label_quality_std': label_quality_scores.std(),
                'n_label_errors': n_errors,
                'label_error_rate': error_rate,
                'quality_10th_percentile': np.percentile(label_quality_scores, 10),
                'quality_90th_percentile': np.percentile(label_quality_scores, 90)
            },
            'class_metrics': pd.DataFrame(class_metrics),
            'quality_distribution': {
                'histogram': np.histogram(label_quality_scores, bins=20),
                'percentiles': {p: np.percentile(label_quality_scores, p) 
                              for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
            }
        }
        return assessment
    def generate_quality_report(self,
                               labels: np.ndarray,
                               pred_probs: np.ndarray,
                               dataset_name: str = 'Dataset') -> Dict[str, Any]:
        """
        Generate comprehensive label quality report.
        Args:
            labels: Observed labels
            pred_probs: Predicted probabilities
            dataset_name: Dataset name
        Returns:
            Comprehensive quality report
        """
        print(f"\nGenerating label quality report for {dataset_name}...")
        # Compute label quality scores
        quality_scores = self.compute_label_quality_scores(
            labels, pred_probs, method='self_confidence'
        )
        # Identify label errors
        label_errors = self.identify_label_errors(
            labels, quality_scores, 
            threshold_method='percentile', 
            threshold_value=0.1
        )
        # Rank label errors
        ranked_errors = self.rank_label_errors(
            labels, pred_probs, label_errors,
            ranking_method='confidence_margin'
        )
        # Assess dataset quality
        quality_assessment = self.assess_dataset_quality(
            labels, quality_scores, label_errors
        )
        # Compile report
        report = {
            'dataset_name': dataset_name,
            'n_samples': len(labels),
            'n_classes': len(np.unique(labels)),
            'label_quality_scores': quality_scores,
            'label_errors_mask': label_errors,
            'ranked_label_errors': ranked_errors,
            'quality_assessment': quality_assessment,
            'summary': {
                'avg_label_quality': quality_assessment['overall']['avg_label_quality'],
                'label_error_rate': quality_assessment['overall']['label_error_rate'],
                'n_label_errors': quality_assessment['overall']['n_label_errors']
            }
        }
        print(f"\nLabel Quality Report Summary for {dataset_name}:")
        print(f"  • Average label quality: {report['summary']['avg_label_quality']:.4f}")
        print(f"  • Label errors: {report['summary']['n_label_errors']} "
              f"({report['summary']['label_error_rate']:.1%})")
        return report
def main():
    """Test the ConfidenceScorer."""
    print("=" * 60)
    print("Confidence Scoring Test")
    print("=" * 60)
    # Create synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=300,
        n_features=15,
        n_classes=3,
        n_informative=8,
        random_state=42
    )
    # Add some noise
    np.random.seed(42)
    noise_mask = np.random.rand(len(y)) < 0.15
    y_noisy = y.copy()
    for i in np.where(noise_mask)[0]:
        other_classes = [c for c in np.unique(y) if c != y[i]]
        y_noisy[i] = np.random.choice(other_classes)
    print(f"  • Samples: {X.shape[0]}")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Classes: {len(np.unique(y))}")
    print(f"  • Noise rate: {noise_mask.mean():.1%}")
    # Get predicted probabilities
    print("\n2. Getting predicted probabilities...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    pred_probs = cross_val_predict(
        model, X_scaled, y_noisy,
        cv=5, method='predict_proba', n_jobs=-1
    )
    # Initialize scorer
    print("\n3. Initializing ConfidenceScorer...")
    scorer = ConfidenceScorer(random_state=42)
    # Generate quality report
    print("\n4. Generating quality report...")
    report = scorer.generate_quality_report(
        y_noisy, pred_probs, dataset_name='Synthetic Test'
    )
    # Show top ranked errors
    print("\n5. Top 10 most confident label errors:")
    if not report['ranked_label_errors'].empty:
        top_errors = report['ranked_label_errors'].head(10)
        print(top_errors[['sample_index', 'current_label', 'suggested_label', 
                         'self_confidence', 'suggested_confidence', 'confidence_margin']].to_string(index=False))
    # Show class metrics
    print("\n6. Class-level quality metrics:")
    if not report['quality_assessment']['class_metrics'].empty:
        class_df = report['quality_assessment']['class_metrics']
        print(class_df[['class', 'n_samples', 'avg_quality', 'n_errors', 'error_rate']].to_string(index=False))
    print("\n" + "=" * 60)
    print("CONFIDENCE SCORING TEST COMPLETE!")
    print("=" * 60)
    return scorer, report
if __name__ == "__main__":
    scorer, report = main()
