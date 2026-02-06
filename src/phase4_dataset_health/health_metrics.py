"""
Phase 4: Dataset Health Profiling
Goal: Develop comprehensive metrics to assess dataset quality and label reliability
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class DatasetHealthMetrics:
    """Comprehensive dataset health assessment metrics."""
    def __init__(self):
        """Initialize health metrics calculator."""
        pass
    def calculate_class_balance_metrics(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Calculate class balance metrics.
        Args:
            y: Labels
        Returns:
            Dictionary of class balance metrics
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        n_samples = len(y)
        # Basic statistics
        min_count = class_counts.min()
        max_count = class_counts.max()
        avg_count = class_counts.mean()
        std_count = class_counts.std()
        # Balance metrics
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        entropy = -np.sum((class_counts / n_samples) * np.log2(class_counts / n_samples + 1e-10))
        max_entropy = np.log2(n_classes)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        # Skewness metrics
        from scipy.stats import skew
        count_skewness = skew(class_counts) if len(class_counts) > 2 else 0
        # Minority class metrics
        minority_classes = class_counts < avg_count * 0.5
        n_minority_classes = minority_classes.sum()
        minority_ratio = n_minority_classes / n_classes if n_classes > 0 else 0
        metrics = {
            'n_classes': n_classes,
            'n_samples': n_samples,
            'min_class_size': int(min_count),
            'max_class_size': int(max_count),
            'avg_class_size': float(avg_count),
            'std_class_size': float(std_count),
            'imbalance_ratio': float(imbalance_ratio),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'count_skewness': float(count_skewness),
            'n_minority_classes': int(n_minority_classes),
            'minority_ratio': float(minority_ratio),
            'is_balanced': imbalance_ratio < 2.0,  # Threshold for balanced dataset
            'class_distribution': {
                int(cls): int(count) for cls, count in zip(unique_classes, class_counts)
            }
        }
        return metrics
    def calculate_feature_quality_metrics(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Calculate feature quality metrics.
        Args:
            X: Feature matrix
        Returns:
            Dictionary of feature quality metrics
        """
        n_samples, n_features = X.shape
        # Missing values
        missing_values = np.isnan(X).sum()
        missing_ratio = missing_values / (n_samples * n_features)
        # Constant features
        feature_stds = np.std(X, axis=0)
        n_constant_features = np.sum(feature_stds < 1e-10)
        constant_features_ratio = n_constant_features / n_features if n_features > 0 else 0
        # Correlation analysis
        if n_features > 1:
            corr_matrix = np.corrcoef(X.T)
            np.fill_diagonal(corr_matrix, 0)
            avg_correlation = np.mean(np.abs(corr_matrix))
            max_correlation = np.max(np.abs(corr_matrix))
            high_correlation_features = np.sum(np.abs(corr_matrix) > 0.9) / 2  # Divide by 2 because symmetric
        else:
            avg_correlation = 0
            max_correlation = 0
            high_correlation_features = 0
        # Feature statistics
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        metrics = {
            'n_features': n_features,
            'n_samples': n_samples,
            'missing_values': int(missing_values),
            'missing_ratio': float(missing_ratio),
            'n_constant_features': int(n_constant_features),
            'constant_features_ratio': float(constant_features_ratio),
            'avg_feature_correlation': float(avg_correlation),
            'max_feature_correlation': float(max_correlation),
            'high_correlation_features': int(high_correlation_features),
            'avg_feature_mean': float(np.mean(feature_means)),
            'std_feature_mean': float(np.std(feature_means)),
            'avg_feature_std': float(np.mean(feature_stds)),
            'std_feature_std': float(np.std(feature_stds)),
            'feature_quality_score': self._calculate_feature_quality_score(
                missing_ratio, constant_features_ratio, avg_correlation
            )
        }
        return metrics
    def calculate_label_consistency_metrics(self, 
                                           X: np.ndarray, 
                                           y: np.ndarray,
                                           n_neighbors: int = 5) -> Dict[str, Any]:
        """
        Calculate label consistency metrics using nearest neighbors.
        Args:
            X: Feature matrix
            y: Labels
            n_neighbors: Number of neighbors to consider
        Returns:
            Dictionary of label consistency metrics
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        n_samples = len(y)
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n_samples)).fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        # Calculate label agreement with neighbors
        label_agreements = []
        for i in range(n_samples):
            neighbor_labels = y[indices[i, 1:]]  # Exclude self
            agreement = np.mean(neighbor_labels == y[i])
            label_agreements.append(agreement)
        label_agreements = np.array(label_agreements)
        # Calculate metrics
        avg_label_agreement = np.mean(label_agreements)
        std_label_agreement = np.std(label_agreements)
        # Identify inconsistent labels
        inconsistency_threshold = 0.5  # Less than 50% agreement with neighbors
        inconsistent_mask = label_agreements < inconsistency_threshold
        n_inconsistent = inconsistent_mask.sum()
        inconsistency_ratio = n_inconsistent / n_samples
        # Class-level consistency
        unique_classes = np.unique(y)
        class_consistencies = []
        for cls in unique_classes:
            class_mask = y == cls
            if class_mask.sum() > 0:
                class_agreement = label_agreements[class_mask].mean()
                class_consistencies.append(class_agreement)
        avg_class_consistency = np.mean(class_consistencies) if class_consistencies else 0
        min_class_consistency = np.min(class_consistencies) if class_consistencies else 0
        metrics = {
            'n_samples': n_samples,
            'avg_label_agreement': float(avg_label_agreement),
            'std_label_agreement': float(std_label_agreement),
            'n_inconsistent_labels': int(n_inconsistent),
            'inconsistency_ratio': float(inconsistency_ratio),
            'avg_class_consistency': float(avg_class_consistency),
            'min_class_consistency': float(min_class_consistency),
            'label_consistency_score': self._calculate_consistency_score(
                avg_label_agreement, inconsistency_ratio
            ),
            'label_agreements': label_agreements.tolist(),
            'inconsistent_indices': np.where(inconsistent_mask)[0].tolist()
        }
        return metrics
    def calculate_separability_metrics(self, 
                                      X: np.ndarray, 
                                      y: np.ndarray) -> Dict[str, Any]:
        """
        Calculate class separability metrics.
        Args:
            X: Feature matrix
            y: Labels
        Returns:
            Dictionary of separability metrics
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Linear Discriminant Analysis
        try:
            lda = LinearDiscriminantAnalysis()
            X_lda = lda.fit_transform(X_scaled, y)
            # Calculate LDA-based separability
            if X_lda.shape[1] > 0:
                # Calculate between-class scatter
                class_means = [X_lda[y == cls].mean(axis=0) for cls in unique_classes]
                overall_mean = X_lda.mean(axis=0)
                between_scatter = sum([
                    len(X_lda[y == cls]) * np.outer(mean - overall_mean, mean - overall_mean)
                    for cls, mean in zip(unique_classes, class_means)
                ])
                # Calculate within-class scatter
                within_scatter = sum([
                    np.cov(X_lda[y == cls].T) * (len(X_lda[y == cls]) - 1)
                    for cls in unique_classes
                ])
                # Fisher discriminant ratio
                if np.linalg.matrix_rank(within_scatter) == within_scatter.shape[0]:
                    fisher_ratio = np.trace(np.linalg.inv(within_scatter) @ between_scatter)
                else:
                    fisher_ratio = 0
            else:
                fisher_ratio = 0
        except:
            fisher_ratio = 0
        # PCA-based separability
        pca = PCA(n_components=min(2, n_features))
        X_pca = pca.fit_transform(X_scaled)
        # Calculate inter-class distances
        from scipy.spatial.distance import cdist
        class_distances = []
        for i, cls_i in enumerate(unique_classes):
            for j, cls_j in enumerate(unique_classes):
                if i < j:
                    samples_i = X_scaled[y == cls_i]
                    samples_j = X_scaled[y == cls_j]
                    if len(samples_i) > 0 and len(samples_j) > 0:
                        distances = cdist(samples_i, samples_j)
                        avg_distance = distances.mean()
                        class_distances.append(avg_distance)
        avg_inter_class_distance = np.mean(class_distances) if class_distances else 0
        # Calculate intra-class compactness
        intra_class_compactness = []
        for cls in unique_classes:
            samples = X_scaled[y == cls]
            if len(samples) > 1:
                center = samples.mean(axis=0)
                distances = np.linalg.norm(samples - center, axis=1)
                compactness = distances.mean()
                intra_class_compactness.append(compactness)
        avg_intra_class_compactness = np.mean(intra_class_compactness) if intra_class_compactness else 0
        # Separability score
        if avg_intra_class_compactness > 0:
            separability_score = avg_inter_class_distance / avg_intra_class_compactness
        else:
            separability_score = 0
        metrics = {
            'n_classes': n_classes,
            'fisher_discriminant_ratio': float(fisher_ratio),
            'avg_inter_class_distance': float(avg_inter_class_distance),
            'avg_intra_class_compactness': float(avg_intra_class_compactness),
            'separability_score': float(separability_score),
            'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist() if hasattr(pca, 'explained_variance_ratio_') else [],
            'separability_quality': self._assess_separability_quality(separability_score)
        }
        return metrics
    def calculate_overall_health_score(self, 
                                      X: np.ndarray, 
                                      y: np.ndarray,
                                      weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate overall dataset health score.
        Args:
            X: Feature matrix
            y: Labels
            weights: Optional weights for different health components
        Returns:
            Comprehensive health assessment
        """
        if weights is None:
            weights = {
                'class_balance': 0.25,
                'feature_quality': 0.25,
                'label_consistency': 0.25,
                'separability': 0.25
            }
        # Calculate all component metrics
        class_balance = self.calculate_class_balance_metrics(y)
        feature_quality = self.calculate_feature_quality_metrics(X)
        label_consistency = self.calculate_label_consistency_metrics(X, y)
        separability = self.calculate_separability_metrics(X, y)
        # Calculate component scores
        balance_score = self._normalize_score(class_balance['normalized_entropy'])
        feature_score = feature_quality['feature_quality_score']
        consistency_score = label_consistency['label_consistency_score']
        separability_score = self._normalize_score(separability['separability_score'])
        # Calculate weighted overall score
        overall_score = (
            weights['class_balance'] * balance_score +
            weights['feature_quality'] * feature_score +
            weights['label_consistency'] * consistency_score +
            weights['separability'] * separability_score
        )
        # Determine health status
        if overall_score >= 0.8:
            health_status = "EXCELLENT"
        elif overall_score >= 0.6:
            health_status = "GOOD"
        elif overall_score >= 0.4:
            health_status = "FAIR"
        else:
            health_status = "POOR"
        # Identify issues
        issues = []
        if balance_score < 0.5:
            issues.append("Class imbalance")
        if feature_score < 0.5:
            issues.append("Feature quality issues")
        if consistency_score < 0.5:
            issues.append("Label inconsistency")
        if separability_score < 0.5:
            issues.append("Poor class separability")
        health_assessment = {
            'overall_score': float(overall_score),
            'health_status': health_status,
            'component_scores': {
                'class_balance': float(balance_score),
                'feature_quality': float(feature_score),
                'label_consistency': float(consistency_score),
                'separability': float(separability_score)
            },
            'weights': weights,
            'issues': issues,
            'n_issues': len(issues),
            'recommendations': self._generate_recommendations(
                balance_score, feature_score, consistency_score, separability_score
            ),
            'component_metrics': {
                'class_balance': class_balance,
                'feature_quality': feature_quality,
                'label_consistency': label_consistency,
                'separability': separability
            }
        }
        return health_assessment
    def _calculate_feature_quality_score(self, 
                                        missing_ratio: float, 
                                        constant_ratio: float,
                                        avg_correlation: float) -> float:
        """Calculate feature quality score from 0 to 1."""
        # Penalize missing values
        missing_penalty = 1.0 - min(missing_ratio * 10, 1.0)
        # Penalize constant features
        constant_penalty = 1.0 - min(constant_ratio * 5, 1.0)
        # Penalize high correlation (some correlation is good, too much is bad)
        if avg_correlation < 0.3:
            correlation_score = 1.0
        elif avg_correlation < 0.7:
            correlation_score = 0.7
        else:
            correlation_score = 0.3
        # Combine scores
        score = (missing_penalty + constant_penalty + correlation_score) / 3
        return max(0, min(1, score))
    def _calculate_consistency_score(self, 
                                    avg_agreement: float, 
                                    inconsistency_ratio: float) -> float:
        """Calculate label consistency score from 0 to 1."""
        # Reward high agreement
        agreement_score = avg_agreement
        # Penalize inconsistency
        inconsistency_penalty = 1.0 - min(inconsistency_ratio * 2, 1.0)
        # Combine
        score = (agreement_score + inconsistency_penalty) / 2
        return max(0, min(1, score))
    def _normalize_score(self, value: float, min_val: float = 0, max_val: float = 1) -> float:
        """Normalize score to 0-1 range."""
        return max(0, min(1, (value - min_val) / (max_val - min_val) if max_val > min_val else 0))
    def _assess_separability_quality(self, separability_score: float) -> str:
        """Assess separability quality based on score."""
        if separability_score > 2.0:
            return "EXCELLENT"
        elif separability_score > 1.0:
            return "GOOD"
        elif separability_score > 0.5:
            return "FAIR"
        else:
            return "POOR"
    def _generate_recommendations(self, 
                                 balance_score: float,
                                 feature_score: float,
                                 consistency_score: float,
                                 separability_score: float) -> List[str]:
        """Generate recommendations based on component scores."""
        recommendations = []
        if balance_score < 0.6:
            recommendations.extend([
                "Consider oversampling minority classes",
                "Use class weighting in models",
                "Apply SMOTE or other balancing techniques"
            ])
        if feature_score < 0.6:
            recommendations.extend([
                "Check for missing values",
                "Remove constant features",
                "Apply feature selection",
                "Normalize/standardize features"
            ])
        if consistency_score < 0.6:
            recommendations.extend([
                "Review labels for inconsistent samples",
                "Consider relabeling suspicious samples",
                "Use ensemble methods for robust predictions"
            ])
        if separability_score < 0.6:
            recommendations.extend([
                "Feature engineering may help",
                "Consider dimensionality reduction",
                "Try different distance metrics"
            ])
        # Add general recommendations
        if len(recommendations) == 0:
            recommendations.append("Dataset looks healthy! Continue with modeling.")
        return recommendations
    def generate_health_report(self, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              dataset_name: str = "Dataset") -> Dict[str, Any]:
        """
        Generate comprehensive health report.
        Args:
            X: Feature matrix
            y: Labels
            dataset_name: Name of the dataset
        Returns:
            Comprehensive health report
        """
        print(f"\n{'='*60}")
        print(f"DATASET HEALTH REPORT: {dataset_name}")
        print(f"{'='*60}")
        # Calculate overall health
        health_assessment = self.calculate_overall_health_score(X, y)
        # Print summary
        print(f"\n📊 OVERALL HEALTH SCORE: {health_assessment['overall_score']:.3f}")
        print(f"   Status: {health_assessment['health_status']}")
        print(f"\n📈 COMPONENT SCORES:")
        for component, score in health_assessment['component_scores'].items():
            print(f"   • {component.replace('_', ' ').title()}: {score:.3f}")
        print(f"\n⚠️  IDENTIFIED ISSUES ({health_assessment['n_issues']}):")
        if health_assessment['issues']:
            for issue in health_assessment['issues']:
                print(f"   • {issue}")
        else:
            print("   • No major issues detected")
        print(f"\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(health_assessment['recommendations'], 1):
            print(f"   {i}. {recommendation}")
        # Detailed metrics
        print(f"\n📋 DETAILED METRICS:")
        metrics = health_assessment['component_metrics']
        # Class balance
        balance = metrics['class_balance']
        print(f"\n   Class Balance:")
        print(f"     • Classes: {balance['n_classes']}")
        print(f"     • Imbalance ratio: {balance['imbalance_ratio']:.2f}")
        print(f"     • Minority classes: {balance['n_minority_classes']}")
        # Feature quality
        features = metrics['feature_quality']
        print(f"\n   Feature Quality:")
        print(f"     • Features: {features['n_features']}")
        print(f"     • Missing values: {features['missing_values']} ({features['missing_ratio']:.1%})")
        print(f"     • Constant features: {features['n_constant_features']}")
        # Label consistency
        consistency = metrics['label_consistency']
        print(f"\n   Label Consistency:")
        print(f"     • Avg neighbor agreement: {consistency['avg_label_agreement']:.3f}")
        print(f"     • Inconsistent labels: {consistency['n_inconsistent_labels']} ({consistency['inconsistency_ratio']:.1%})")
        # Separability
        separability = metrics['separability']
        print(f"\n   Class Separability:")
        print(f"     • Separability score: {separability['separability_score']:.3f}")
        print(f"     • Quality: {separability['separability_quality']}")
        print(f"\n{'='*60}")
        print("HEALTH REPORT COMPLETE")
        print(f"{'='*60}")
        # Add dataset info to report
        health_assessment['dataset_info'] = {
            'name': dataset_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }
        return health_assessment
def main():
    """Test the DatasetHealthMetrics."""
    print("=" * 60)
    print("Phase 4: Dataset Health Metrics")
    print("=" * 60)
    # Create test datasets
    from sklearn.datasets import make_classification, load_digits
    print("\n1. Creating test datasets...")
    # Create a healthy dataset
    X_healthy, y_healthy = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.4, 0.3, 0.3],  # Slightly imbalanced
        random_state=42
    )
    # Create an unhealthy dataset (imbalanced, noisy)
    X_unhealthy, y_unhealthy = make_classification(
        n_samples=500,
        n_features=30,
        n_classes=4,
        n_informative=10,
        n_redundant=10,
        weights=[0.6, 0.2, 0.15, 0.05],  # Highly imbalanced
        flip_y=0.2,  # 20% label noise
        random_state=42
    )
    print(f"   • Healthy dataset: {X_healthy.shape[0]} samples, {X_healthy.shape[1]} features")
    print(f"   • Unhealthy dataset: {X_unhealthy.shape[0]} samples, {X_unhealthy.shape[1]} features")
    # Initialize health metrics
    print("\n2. Initializing health metrics calculator...")
    health_metrics = DatasetHealthMetrics()
    # Analyze healthy dataset
    print("\n3. Analyzing healthy dataset...")
    healthy_report = health_metrics.generate_health_report(
        X_healthy, y_healthy, "Healthy Test Dataset"
    )
    # Analyze unhealthy dataset
    print("\n4. Analyzing unhealthy dataset...")
    unhealthy_report = health_metrics.generate_health_report(
        X_unhealthy, y_unhealthy, "Unhealthy Test Dataset"
    )
    # Compare datasets
    print("\n5. Dataset Comparison:")
    print("-" * 40)
    print(f"{'Metric':<30} {'Healthy':<10} {'Unhealthy':<10}")
    print("-" * 60)
    metrics_to_compare = [
        ('Overall Score', 'overall_score'),
        ('Class Balance', 'component_scores.class_balance'),
        ('Feature Quality', 'component_scores.feature_quality'),
        ('Label Consistency', 'component_scores.label_consistency'),
        ('Separability', 'component_scores.separability')
    ]
    for metric_name, metric_path in metrics_to_compare:
        # Get value from nested dictionary
        healthy_val = healthy_report
        unhealthy_val = unhealthy_report
        for key in metric_path.split('.'):
            healthy_val = healthy_val[key]
            unhealthy_val = unhealthy_val[key]
        print(f"{metric_name:<30} {healthy_val:<10.3f} {unhealthy_val:<10.3f}")
    print("\n" + "=" * 60)
    print("Health Metrics Complete!")
    print("=" * 60)
    return health_metrics, healthy_report, unhealthy_report
if __name__ == "__main__":
    health_metrics, healthy_report, unhealthy_report = main()
