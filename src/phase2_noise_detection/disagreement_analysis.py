"""
Disagreement analysis for Phase 2.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class DisagreementAnalyzer:
    """Analyze label disagreements from multiple perspectives."""
    def __init__(self, random_state: int = 42):
        """Initialize analyzer."""
        self.random_state = random_state
        np.random.seed(random_state)
    def compute_disagreement_metrics(self,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    y_pred: np.ndarray,
                                    confidence_scores: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive disagreement metrics.
        Args:
            X: Feature matrix
            y: True labels
            y_pred: Predicted labels
            confidence_scores: Prediction confidence scores
        Returns:
            Dictionary of disagreement metrics
        """
        n_samples = len(y)
        # Basic disagreement
        disagreement = (y != y_pred).astype(int)
        n_disagreements = disagreement.sum()
        disagreement_rate = n_disagreements / n_samples
        # Confidence statistics
        confidence_correct = confidence_scores[disagreement == 0]
        confidence_incorrect = confidence_scores[disagreement == 1]
        # Per-class statistics
        unique_classes = np.unique(y)
        class_stats = []
        for class_label in unique_classes:
            class_mask = y == class_label
            if class_mask.any():
                class_disagreement = disagreement[class_mask]
                class_confidence = confidence_scores[class_mask]
                class_stats.append({
                    'class': class_label,
                    'n_samples': class_mask.sum(),
                    'n_disagreements': class_disagreement.sum(),
                    'disagreement_rate': class_disagreement.mean(),
                    'avg_confidence': class_confidence.mean(),
                    'avg_confidence_correct': class_confidence[class_disagreement == 0].mean() 
                                             if (class_disagreement == 0).any() else 0,
                    'avg_confidence_incorrect': class_confidence[class_disagreement == 1].mean() 
                                               if (class_disagreement == 1).any() else 0
                })
        # Feature importance for disagreements (simplified)
        feature_importance = self._estimate_feature_importance(X, disagreement)
        # Compile metrics
        metrics = {
            'n_samples': n_samples,
            'n_disagreements': n_disagreements,
            'disagreement_rate': disagreement_rate,
            'overall_confidence': confidence_scores.mean(),
            'confidence_correct': confidence_correct.mean() if len(confidence_correct) > 0 else 0,
            'confidence_incorrect': confidence_incorrect.mean() if len(confidence_incorrect) > 0 else 0,
            'confidence_gap': (confidence_correct.mean() - confidence_incorrect.mean()) 
                            if len(confidence_correct) > 0 and len(confidence_incorrect) > 0 else 0,
            'class_stats': class_stats,
            'feature_importance': feature_importance,
            'disagreement_indices': np.where(disagreement == 1)[0],
            'agreement_indices': np.where(disagreement == 0)[0]
        }
        return metrics
    def _estimate_feature_importance(self, 
                                    X: np.ndarray, 
                                    disagreement: np.ndarray) -> np.ndarray:
        """
        Estimate feature importance for disagreement detection.
        Args:
            X: Feature matrix
            disagreement: Disagreement labels (0=agree, 1=disagree)
        Returns:
            Feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier
        if disagreement.sum() == 0 or (disagreement == 0).sum() == 0:
            # No disagreements or no agreements
            return np.zeros(X.shape[1])
        # Balance the classes by sampling
        disagree_indices = np.where(disagreement == 1)[0]
        agree_indices = np.where(disagreement == 0)[0]
        # Sample to balance classes
        n_samples = min(len(disagree_indices), len(agree_indices), 100)
        if n_samples < 10:
            return np.zeros(X.shape[1])
        sampled_disagree = np.random.choice(disagree_indices, n_samples, replace=False)
        sampled_agree = np.random.choice(agree_indices, n_samples, replace=False)
        # Create balanced dataset
        X_balanced = np.vstack([X[sampled_disagree], X[sampled_agree]])
        y_balanced = np.hstack([np.ones(n_samples), np.zeros(n_samples)])  # 1=disagree, 0=agree
        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        clf.fit(X_balanced, y_balanced)
        return clf.feature_importances_
    def analyze_disagreement_patterns(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     y_pred: np.ndarray,
                                     feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze patterns in disagreements.
        Args:
            X: Feature matrix
            y: True labels
            y_pred: Predicted labels
            feature_names: Optional feature names
        Returns:
            DataFrame with disagreement patterns
        """
        if feature_names is None:
            n_features = X.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
        # Identify disagreement samples
        disagree_indices = np.where(y != y_pred)[0]
        if len(disagree_indices) == 0:
            return pd.DataFrame()
        # Analyze feature differences
        patterns = []
        for idx in disagree_indices:
            true_label = y[idx]
            pred_label = y_pred[idx]
            # Calculate distance to class centroids
            true_class_mask = y == true_label
            pred_class_mask = y == pred_label
            if true_class_mask.any() and pred_class_mask.any():
                true_centroid = X[true_class_mask].mean(axis=0)
                pred_centroid = X[pred_class_mask].mean(axis=0)
                # Distance to true class centroid
                dist_to_true = np.linalg.norm(X[idx] - true_centroid)
                # Distance to predicted class centroid
                dist_to_pred = np.linalg.norm(X[idx] - pred_centroid)
                # Which features contribute most to the distance
                feature_diffs = np.abs(X[idx] - true_centroid)
                top_features_idx = np.argsort(feature_diffs)[-3:]  # Top 3 features
                top_features = [feature_names[i] for i in top_features_idx]
                patterns.append({
                    'sample_index': idx,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'dist_to_true_centroid': dist_to_true,
                    'dist_to_pred_centroid': dist_to_pred,
                    'closer_to_pred': dist_to_pred < dist_to_true,
                    'distance_ratio': dist_to_pred / dist_to_true if dist_to_true > 0 else float('inf'),
                    'top_features': ', '.join(top_features)
                })
        df_patterns = pd.DataFrame(patterns)
        if not df_patterns.empty:
            # Add additional metrics
            df_patterns['ambiguity_score'] = df_patterns['distance_ratio'].apply(
                lambda x: min(x, 1/x) if x > 0 else 0
            )
            # Sort by ambiguity (most ambiguous first)
            df_patterns = df_patterns.sort_values('ambiguity_score', ascending=False)
            df_patterns['rank'] = range(1, len(df_patterns) + 1)
        return df_patterns
    def generate_disagreement_report(self,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    cv_results: Dict[str, Any],
                                    dataset_name: str = 'Dataset') -> Dict[str, Any]:
        """
        Generate comprehensive disagreement report.
        Args:
            X: Feature matrix
            y: True labels
            cv_results: Cross-validation results
            dataset_name: Dataset name
        Returns:
            Comprehensive report
        """
        # Extract predictions and confidence
        y_pred = cv_results['y_pred']
        confidence_scores = cv_results['confidence_scores']
        # Compute metrics
        metrics = self.compute_disagreement_metrics(X, y, y_pred, confidence_scores)
        # Analyze patterns
        patterns = self.analyze_disagreement_patterns(X, y, y_pred)
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, patterns)
        # Compile report
        report = {
            'dataset': dataset_name,
            'metrics': metrics,
            'patterns': patterns,
            'recommendations': recommendations,
            'summary': {
                'n_samples': metrics['n_samples'],
                'n_disagreements': metrics['n_disagreements'],
                'disagreement_rate': metrics['disagreement_rate'],
                'confidence_gap': metrics['confidence_gap'],
                'most_problematic_class': max(metrics['class_stats'], 
                                            key=lambda x: x['disagreement_rate'])['class']
                if metrics['class_stats'] else None
            }
        }
        return report
    def _generate_recommendations(self,
                                 metrics: Dict[str, Any],
                                 patterns: pd.DataFrame) -> List[str]:
        """Generate recommendations based on disagreement analysis."""
        recommendations = []
        # Basic recommendations
        if metrics['disagreement_rate'] > 0.3:
            recommendations.append("High disagreement rate (>30%). Consider reviewing labels or improving features.")
        elif metrics['disagreement_rate'] > 0.1:
            recommendations.append("Moderate disagreement rate (10-30%). Some label review recommended.")
        else:
            recommendations.append("Low disagreement rate (<10%). Labels appear consistent.")
        # Confidence-based recommendations
        if metrics['confidence_gap'] > 0.2:
            recommendations.append("Large confidence gap between correct and incorrect predictions. Model is uncertain about disagreements.")
        elif metrics['confidence_gap'] < 0.05:
            recommendations.append("Small confidence gap. Model is equally confident about correct and incorrect predictions.")
        # Class imbalance recommendations
        class_rates = [stat['disagreement_rate'] for stat in metrics['class_stats']]
        if max(class_rates) > 2 * min(class_rates):
            recommendations.append("Significant variation in disagreement rates across classes. Some classes may have poorer quality labels.")
        # Pattern-based recommendations
        if not patterns.empty:
            ambiguous_count = patterns['ambiguity_score'].mean()
            if ambiguous_count > 0.7:
                recommendations.append("Many ambiguous samples found. These are good candidates for relabeling.")
            if patterns['closer_to_pred'].mean() > 0.6:
                recommendations.append("Most misclassified samples are closer to predicted class centroid. Labels may be incorrect.")
        return recommendations
    def visualize_disagreement_analysis(self,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       cv_results: Dict[str, Any],
                                       dataset_name: str = 'Dataset') -> None:
        """
        Visualize disagreement analysis.
        Args:
            X: Feature matrix
            y: True labels
            cv_results: Cross-validation results
            dataset_name: Dataset name
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        y_pred = cv_results['y_pred']
        confidence_scores = cv_results['confidence_scores']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # 1. Confidence vs Disagreement
        ax1 = axes[0, 0]
        disagree_mask = (y != y_pred)
        ax1.scatter(confidence_scores[~disagree_mask], 
                   np.zeros((~disagree_mask).sum()),
                   alpha=0.5, label='Agree', color='green')
        ax1.scatter(confidence_scores[disagree_mask], 
                   np.ones(disagree_mask.sum()),
                   alpha=0.5, label='Disagree', color='red')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Disagreement (0=Agree, 1=Disagree)')
        ax1.set_title('Confidence vs Disagreement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 2. Class-wise disagreement rates
        ax2 = axes[0, 1]
        metrics = self.compute_disagreement_metrics(X, y, y_pred, confidence_scores)
        class_stats = metrics['class_stats']
        if class_stats:
            class_df = pd.DataFrame(class_stats)
            bars = ax2.bar(class_df['class'], class_df['disagreement_rate'],
                          color='coral', edgecolor='black')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Disagreement Rate')
            ax2.set_title('Class-wise Disagreement Rates')
            ax2.grid(True, alpha=0.3)
            for bar, rate in zip(bars, class_df['disagreement_rate']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
        # 3. Feature importance for disagreement
        ax3 = axes[1, 0]
        feature_importance = metrics['feature_importance']
        if feature_importance.sum() > 0:
            n_top_features = min(10, len(feature_importance))
            top_indices = np.argsort(feature_importance)[-n_top_features:]
            top_importance = feature_importance[top_indices]
            ax3.barh(range(n_top_features), top_importance,
                    color='skyblue', edgecolor='black')
            ax3.set_yticks(range(n_top_features))
            ax3.set_yticklabels([f'Feature {i}' for i in top_indices])
            ax3.set_xlabel('Importance')
            ax3.set_title('Top Features for Disagreement Detection')
            ax3.grid(True, alpha=0.3)
        # 4. Distance ratio histogram
        ax4 = axes[1, 1]
        patterns = self.analyze_disagreement_patterns(X, y, y_pred)
        if not patterns.empty:
            ax4.hist(patterns['distance_ratio'].clip(0, 5), bins=30,
                    alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Distance Ratio (pred/true)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distance Ratio Distribution\n(clipped at 5)')
            ax4.axvline(x=1, color='red', linestyle='--', label='Equal distance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        plt.suptitle(f'Disagreement Analysis - {dataset_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
def main():
    """Test the DisagreementAnalyzer."""
    print("=" * 60)
    print("Phase 2: Disagreement Analysis")
    print("=" * 60)
    # Create test dataset
    from sklearn.datasets import make_classification
    print("\nCreating test dataset...")
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_classes=3,
        n_informative=6,
        random_state=42
    )
    # Simulate CV results
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    y_pred = cross_val_predict(model, X_scaled, y, cv=5, n_jobs=-1)
    y_pred_proba = cross_val_predict(model, X_scaled, y, cv=5, method='predict_proba', n_jobs=-1)
    confidence_scores = np.max(y_pred_proba, axis=1)
    cv_results = {
        'y_pred': y_pred,
        'confidence_scores': confidence_scores
    }
    print(f"  • Samples: {X.shape[0]}")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Classes: {len(np.unique(y))}")
    print(f"  • Disagreements: {(y != y_pred).sum()} ({(y != y_pred).mean():.1%})")
    # Initialize analyzer
    analyzer = DisagreementAnalyzer(random_state=42)
    # Generate report
    print("\nGenerating disagreement report...")
    report = analyzer.generate_disagreement_report(X, y, cv_results, dataset_name='Test Dataset')
    print("\nSummary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    # Visualize
    print("\nCreating visualizations...")
    analyzer.visualize_disagreement_analysis(X, y, cv_results, dataset_name='Test Dataset')
    print("\n" + "=" * 60)
    print("Disagreement Analysis Complete!")
    print("=" * 60)
    return analyzer, report
if __name__ == "__main__":
    analyzer, report = main()
