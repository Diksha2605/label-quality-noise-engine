"""
Cross-validation utilities for Phase 2.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')
class CrossValidationAnalyzer:
    """Advanced cross-validation analysis for noise detection."""
    def __init__(self, 
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize CV analyzer.
        Args:
            n_folds: Number of folds
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.random_state = random_state
    def run_cv_analysis(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       model_type: str = 'logistic') -> Dict[str, Any]:
        """
        Run comprehensive cross-validation analysis.
        Args:
            X: Feature matrix
            y: Labels
            model_type: Type of classifier
        Returns:
            Comprehensive CV results
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        print(f"Running {self.n_folds}-fold cross-validation analysis...")
        # Select model
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Get cross-validated predictions
        y_pred = cross_val_predict(
            model, X_scaled, y,
            cv=StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state),
            n_jobs=-1,
            method='predict'
        )
        # Get predicted probabilities
        y_pred_proba = cross_val_predict(
            model, X_scaled, y,
            cv=StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state),
            n_jobs=-1,
            method='predict_proba'
        )
        # Calculate confidence scores (max probability)
        confidence_scores = np.max(y_pred_proba, axis=1)
        # Calculate disagreement
        disagreement = (y_pred != y).astype(int)
        # Calculate per-fold statistics
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_stats = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
            fold_accuracy = accuracy_score(y[val_idx], y_pred[val_idx])
            fold_disagreement = disagreement[val_idx].mean()
            fold_confidence = confidence_scores[val_idx].mean()
            fold_stats.append({
                'fold': fold,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'accuracy': fold_accuracy,
                'disagreement_rate': fold_disagreement,
                'avg_confidence': fold_confidence
            })
        # Overall metrics
        overall_accuracy = accuracy_score(y, y_pred)
        overall_disagreement = disagreement.mean()
        overall_confidence = confidence_scores.mean()
        # Create confusion matrix between true and predicted labels
        conf_matrix = confusion_matrix(y, y_pred)
        # Calculate class-wise metrics
        unique_classes = np.unique(y)
        class_metrics = []
        for class_label in unique_classes:
            class_mask = y == class_label
            if class_mask.any():
                class_accuracy = accuracy_score(y[class_mask], y_pred[class_mask])
                class_disagreement = disagreement[class_mask].mean()
                class_confidence = confidence_scores[class_mask].mean()
                class_metrics.append({
                    'class': class_label,
                    'n_samples': class_mask.sum(),
                    'accuracy': class_accuracy,
                    'disagreement_rate': class_disagreement,
                    'avg_confidence': class_confidence
                })
        # Compile results
        results = {
            'y_true': y,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confidence_scores': confidence_scores,
            'disagreement': disagreement,
            'fold_stats': fold_stats,
            'overall_accuracy': overall_accuracy,
            'overall_disagreement': overall_disagreement,
            'overall_confidence': overall_confidence,
            'confusion_matrix': conf_matrix,
            'class_metrics': class_metrics,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': len(unique_classes)
        }
        print(f"\nCV Analysis Complete:")
        print(f"  • Overall accuracy: {overall_accuracy:.3f}")
        print(f"  • Overall disagreement: {overall_disagreement:.3f} ({overall_disagreement*100:.1f}%)")
        print(f"  • Average confidence: {overall_confidence:.3f}")
        print(f"  • Suspicious samples: {disagreement.sum()} ({overall_disagreement*100:.1f}%)")
        return results
    def identify_consistent_misclassifications(self,
                                              results: Dict[str, Any],
                                              min_folds: int = 3) -> pd.DataFrame:
        """
        Identify samples that are consistently misclassified across folds.
        Args:
            results: Results from run_cv_analysis
            min_folds: Minimum number of folds where sample must be misclassified
        Returns:
            DataFrame with consistently misclassified samples
        """
        # Note: This is a simplified version
        # In practice, you'd need to track predictions per fold
        y_true = results['y_true']
        y_pred = results['y_pred']
        confidence_scores = results['confidence_scores']
        disagreement = results['disagreement']
        # Identify misclassified samples
        misclassified_indices = np.where(disagreement == 1)[0]
        if len(misclassified_indices) == 0:
            return pd.DataFrame()
        # Create DataFrame
        misclassified_data = []
        for idx in misclassified_indices:
            misclassified_data.append({
                'sample_index': idx,
                'true_label': y_true[idx],
                'predicted_label': y_pred[idx],
                'confidence': confidence_scores[idx],
                'is_suspicious': True
            })
        df_misclassified = pd.DataFrame(misclassified_data)
        # Sort by confidence (ascending)
        df_misclassified = df_misclassified.sort_values('confidence', ascending=True)
        df_misclassified['rank'] = range(1, len(df_misclassified) + 1)
        return df_misclassified
    def generate_cv_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive CV report.
        Args:
            results: CV results
        Returns:
            Report dictionary
        """
        report = {
            'summary': {
                'n_samples': results['n_samples'],
                'n_features': results['n_features'],
                'n_classes': results['n_classes'],
                'overall_accuracy': results['overall_accuracy'],
                'overall_disagreement': results['overall_disagreement'],
                'overall_confidence': results['overall_confidence'],
                'n_suspicious': results['disagreement'].sum(),
                'suspicious_rate': results['disagreement'].mean()
            },
            'fold_statistics': pd.DataFrame(results['fold_stats']),
            'class_statistics': pd.DataFrame(results['class_metrics']),
            'confusion_matrix': results['confusion_matrix']
        }
        return report
    def visualize_cv_results(self, results: Dict[str, Any], 
                            dataset_name: str = 'Dataset') -> None:
        """
        Visualize CV results.
        Args:
            results: CV results
            dataset_name: Dataset name for titles
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # 1. Fold-wise accuracy
        ax1 = axes[0, 0]
        fold_stats_df = pd.DataFrame(results['fold_stats'])
        ax1.bar(fold_stats_df['fold'], fold_stats_df['accuracy'], 
               color='skyblue', edgecolor='black')
        ax1.axhline(y=results['overall_accuracy'], color='red', 
                   linestyle='--', label=f'Overall: {results["overall_accuracy"]:.3f}')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Fold-wise Cross-Validation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 2. Confidence distribution
        ax2 = axes[0, 1]
        confidence_scores = results['confidence_scores']
        disagreement = results['disagreement']
        # Separate confidence by correct vs incorrect predictions
        correct_conf = confidence_scores[disagreement == 0]
        incorrect_conf = confidence_scores[disagreement == 1]
        ax2.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
        if len(incorrect_conf) > 0:
            ax2.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution by Prediction Correctness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # 3. Class-wise disagreement
        ax3 = axes[1, 0]
        class_stats_df = pd.DataFrame(results['class_metrics'])
        if not class_stats_df.empty:
            bars = ax3.bar(class_stats_df['class'], class_stats_df['disagreement_rate'],
                          color='coral', edgecolor='black')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Disagreement Rate')
            ax3.set_title('Class-wise Disagreement Rate')
            ax3.grid(True, alpha=0.3)
            # Add value labels
            for bar, rate in zip(bars, class_stats_df['disagreement_rate']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
        # 4. Confusion matrix heatmap
        ax4 = axes[1, 1]
        conf_matrix = results['confusion_matrix']
        if conf_matrix.size > 0:
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       ax=ax4, cbar_kws={'label': 'Count'})
            ax4.set_xlabel('Predicted Label')
            ax4.set_ylabel('True Label')
            ax4.set_title('Confusion Matrix')
        plt.suptitle(f'Cross-Validation Analysis - {dataset_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
def main():
    """Test the CrossValidationAnalyzer."""
    print("=" * 60)
    print("Phase 2: Cross-Validation Analysis")
    print("=" * 60)
    # Create test dataset
    from sklearn.datasets import make_classification
    print("\nCreating test dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_classes=3,
        n_informative=8,
        random_state=42
    )
    # Add some noise
    np.random.seed(42)
    noise_mask = np.random.rand(len(y)) < 0.1
    y_noisy = y.copy()
    for i in np.where(noise_mask)[0]:
        other_classes = [c for c in np.unique(y) if c != y[i]]
        y_noisy[i] = np.random.choice(other_classes)
    print(f"  • Samples: {X.shape[0]}")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Classes: {len(np.unique(y))}")
    print(f"  • Injected noise: {noise_mask.sum()} samples ({noise_mask.mean():.1%})")
    # Initialize analyzer
    analyzer = CrossValidationAnalyzer(n_folds=5, random_state=42)
    # Run CV analysis
    print("\nRunning cross-validation analysis...")
    results = analyzer.run_cv_analysis(X, y_noisy, model_type='logistic')
    # Generate report
    print("\nGenerating report...")
    report = analyzer.generate_cv_report(results)
    print("\nSummary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    # Identify misclassifications
    print("\nIdentifying misclassifications...")
    misclassified = analyzer.identify_consistent_misclassifications(results)
    if not misclassified.empty:
        print(f"\nFound {len(misclassified)} misclassified samples")
        print("\nTop 10 most suspicious (lowest confidence):")
        print(misclassified.head(10)[['sample_index', 'true_label', 'predicted_label', 'confidence']].to_string(index=False))
    # Visualize
    print("\nCreating visualizations...")
    analyzer.visualize_cv_results(results, dataset_name='Test Dataset')
    print("\n" + "=" * 60)
    print("Cross-Validation Analysis Complete!")
    print("=" * 60)
    return analyzer, results, misclassified
if __name__ == "__main__":
    analyzer, results, misclassified = main()

