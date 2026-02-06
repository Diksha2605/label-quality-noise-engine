"""
Quality scoring system for Phase 4.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')
class QualityScorer:
    """Unified quality scoring system for datasets and samples."""
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize quality scorer.
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.scores = {}
        self.metrics_history = []
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'weights': {
                'dataset_health': 0.4,
                'class_quality': 0.3,
                'sample_quality': 0.2,
                'label_confidence': 0.1
            },
            'thresholds': {
                'excellent': 0.8,
                'good': 0.6,
                'fair': 0.4,
                'poor': 0.0
            },
            'scoring_functions': {
                'dataset_health': self._score_dataset_health,
                'class_quality': self._score_class_quality,
                'sample_quality': self._score_sample_quality,
                'label_confidence': self._score_label_confidence
            }
        }
    def compute_dataset_quality_score(self, 
                                     X: np.ndarray, 
                                     y: np.ndarray,
                                     dataset_name: str = "Dataset") -> Dict[str, Any]:
        """
        Compute comprehensive quality score for entire dataset.
        Args:
            X: Feature matrix
            y: Labels
            dataset_name: Name of the dataset
        Returns:
            Comprehensive quality assessment
        """
        print(f"\n{'='*60}")
        print(f"QUALITY SCORING: {dataset_name}")
        print(f"{'='*60}")
        # Calculate component scores
        component_scores = {}
        component_details = {}
        for component, scoring_fn in self.config['scoring_functions'].items():
            print(f"\nCalculating {component.replace('_', ' ')} score...")
            score, details = scoring_fn(X, y)
            component_scores[component] = score
            component_details[component] = details
        # Calculate weighted overall score
        overall_score = 0
        for component, weight in self.config['weights'].items():
            overall_score += component_scores[component] * weight
        # Determine quality grade
        quality_grade = self._determine_quality_grade(overall_score)
        # Generate quality report
        quality_report = self._generate_quality_report(
            overall_score, quality_grade, component_scores, component_details
        )
        # Add dataset info
        quality_report['dataset_info'] = {
            'name': dataset_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }
        # Store results
        self.scores[dataset_name] = quality_report
        # Print summary
        self._print_quality_summary(quality_report)
        return quality_report
    def compute_sample_quality_scores(self, 
                                     X: np.ndarray, 
                                     y: np.ndarray) -> np.ndarray:
        """
        Compute quality score for each individual sample.
        Args:
            X: Feature matrix
            y: Labels
        Returns:
            Array of sample quality scores
        """
        n_samples = len(y)
        sample_scores = np.zeros(n_samples)
        # Multiple factors for sample quality
        factors = {
            'neighbor_agreement': self._compute_neighbor_agreement(X, y),
            'outlier_score': self._compute_outlier_score(X),
            'margin_score': self._compute_margin_score(X, y),
            'consistency_score': self._compute_consistency_score(X, y)
        }
        # Combine factors
        for i in range(n_samples):
            scores = []
            for factor_name, factor_scores in factors.items():
                if factor_scores is not None and len(factor_scores) > i:
                    scores.append(factor_scores[i])
            if scores:
                sample_scores[i] = np.mean(scores)
            else:
                sample_scores[i] = 0.5  # Default neutral score
        return sample_scores
    def identify_low_quality_samples(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray,
                                    threshold: float = 0.3,
                                    top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Identify low-quality samples.
        Args:
            X: Feature matrix
            y: Labels
            threshold: Quality score threshold
            top_k: Return only top K lowest quality samples
        Returns:
            DataFrame with low-quality samples
        """
        sample_scores = self.compute_sample_quality_scores(X, y)
        # Identify low-quality samples
        low_quality_mask = sample_scores < threshold
        low_quality_indices = np.where(low_quality_mask)[0]
        # Create results
        results = []
        for idx in low_quality_indices:
            results.append({
                'sample_index': idx,
                'label': y[idx],
                'quality_score': sample_scores[idx],
                'quality_percentile': np.sum(sample_scores < sample_scores[idx]) / len(sample_scores)
            })
        # Create DataFrame
        if results:
            df_results = pd.DataFrame(results)
            # Sort by quality score (ascending = worst first)
            df_results = df_results.sort_values('quality_score', ascending=True)
            # Limit to top_k if specified
            if top_k is not None:
                df_results = df_results.head(top_k)
        else:
            # Return empty DataFrame with correct columns
            df_results = pd.DataFrame(columns=['sample_index', 'label', 'quality_score', 'quality_percentile'])
        return df_results
    def compute_class_quality_scores(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray) -> Dict[int, Dict]:
        """
        Compute quality scores for each class.
        Args:
            X: Feature matrix
            y: Labels
        Returns:
            Dictionary of class quality assessments
        """
        try:
            from phase4_dataset_health.class_profiling import ClassProfiler
            # Use class profiler
            profiler = ClassProfiler()
            class_profiles = profiler.profile_all_classes(X, y)
            class_quality = {}
            for class_label, profile in class_profiles.items():
                # Safely extract values with defaults
                difficulty = profile.get('difficulty_score', 0)
                outlier_analysis = profile.get('outlier_analysis', {})
                confusion_analysis = profile.get('confusion_analysis', {})
                issues = profile.get('issues', [])
                # Calculate class quality score from profile
                quality_score = self._calculate_class_quality_from_profile(profile)
                class_quality[class_label] = {
                    'quality_score': quality_score,
                    'n_samples': profile.get('n_samples', 0),
                    'difficulty_score': difficulty,
                    'outlier_ratio': outlier_analysis.get('outlier_ratio', 0),
                    'confusion_rate': confusion_analysis.get('confusion_rate', 0),
                    'issues': issues
                }
            return class_quality
        except Exception as e:
            print(f"Warning: Could not compute class quality scores: {e}")
            # Return empty dict as fallback
            return {}
    def _score_dataset_health(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """Score dataset health component."""
        from phase4_dataset_health.health_metrics import DatasetHealthMetrics
        health_metrics = DatasetHealthMetrics()
        health_assessment = health_metrics.calculate_overall_health_score(X, y)
        score = health_assessment['overall_score']
        details = {
            'component_scores': health_assessment['component_scores'],
            'health_status': health_assessment['health_status'],
            'issues': health_assessment['issues']
        }
        return score, details
    def _score_class_quality(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """Score class quality component."""
        class_quality = self.compute_class_quality_scores(X, y)
        if not class_quality:
            return 0.5, {'error': 'No classes found'}
        # Average class quality score
        scores = [info['quality_score'] for info in class_quality.values()]
        avg_score = np.mean(scores)
        # Identify problematic classes
        problematic_classes = []
        for class_label, info in class_quality.items():
            if info['quality_score'] < 0.5:
                problematic_classes.append({
                    'class': class_label,
                    'score': info['quality_score'],
                    'issues': info.get('issues', [])
                })
        details = {
            'avg_class_score': avg_score,
            'min_class_score': min(scores) if scores else 0,
            'max_class_score': max(scores) if scores else 0,
            'problematic_classes': problematic_classes,
            'n_problematic': len(problematic_classes)
        }
        return avg_score, details
    def _score_sample_quality(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """Score sample quality component."""
        sample_scores = self.compute_sample_quality_scores(X, y)
        if len(sample_scores) == 0:
            return 0.5, {'error': 'No samples'}
        avg_score = np.mean(sample_scores)
        # Count low-quality samples
        low_quality_threshold = 0.3
        n_low_quality = np.sum(sample_scores < low_quality_threshold)
        low_quality_ratio = n_low_quality / len(sample_scores)
        details = {
            'avg_sample_score': avg_score,
            'std_sample_score': np.std(sample_scores),
            'n_low_quality_samples': int(n_low_quality),
            'low_quality_ratio': float(low_quality_ratio),
            'sample_score_distribution': {
                'min': float(np.min(sample_scores)),
                'q1': float(np.percentile(sample_scores, 25)),
                'median': float(np.median(sample_scores)),
                'q3': float(np.percentile(sample_scores, 75)),
                'max': float(np.max(sample_scores))
            }
        }
        return avg_score, details
    def _score_label_confidence(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """Score label confidence component."""
        from sklearn.model_selection import cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        n_samples = len(y)
        if n_samples < 10:
            return 0.5, {'error': 'Insufficient samples for confidence estimation'}
        try:
            # Use Random Forest for confidence estimation
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Get predicted probabilities
            y_proba = cross_val_predict(
                model, X_scaled, y,
                method='predict_proba',
                cv=min(5, n_samples // 10)
            )
            # Calculate confidence scores (max probability)
            confidence_scores = np.max(y_proba, axis=1)
            avg_confidence = np.mean(confidence_scores)
            # Count low-confidence predictions
            low_confidence_threshold = 0.5
            n_low_confidence = np.sum(confidence_scores < low_confidence_threshold)
            low_confidence_ratio = n_low_confidence / n_samples
            details = {
                'avg_confidence': float(avg_confidence),
                'std_confidence': float(np.std(confidence_scores)),
                'n_low_confidence': int(n_low_confidence),
                'low_confidence_ratio': float(low_confidence_ratio),
                'confidence_distribution': {
                    'min': float(np.min(confidence_scores)),
                    'q1': float(np.percentile(confidence_scores, 25)),
                    'median': float(np.median(confidence_scores)),
                    'q3': float(np.percentile(confidence_scores, 75)),
                    'max': float(np.max(confidence_scores))
                }
            }
            return avg_confidence, details
        except Exception as e:
            # Fallback to simpler estimation
            print(f"  Warning: Could not compute label confidence: {e}")
            # Estimate based on class balance
            unique_classes, counts = np.unique(y, return_counts=True)
            if len(unique_classes) > 1:
                # More balanced = higher confidence
                balance_score = 1.0 - (np.std(counts) / np.mean(counts)) / 2
            else:
                balance_score = 0.5
            return balance_score, {'method': 'fallback', 'balance_score': balance_score}
    def _compute_neighbor_agreement(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute agreement with nearest neighbors."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        n_samples = len(y)
        if n_samples < 5:
            return np.ones(n_samples) * 0.5
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_neighbors = min(5, n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_scaled)
        _, indices = nbrs.kneighbors(X_scaled)
        agreements = np.zeros(n_samples)
        for i in range(n_samples):
            neighbor_labels = y[indices[i, 1:]]  # Exclude self
            agreements[i] = np.mean(neighbor_labels == y[i])
        return agreements
    def _compute_outlier_score(self, X: np.ndarray) -> np.ndarray:
        """Compute outlier scores for samples."""
        from sklearn.ensemble import IsolationForest
        n_samples = len(X)
        if n_samples < 10:
            return np.ones(n_samples) * 0.5
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_scores = iso_forest.fit_predict(X)
            # Convert to 0-1 range (1 = inlier, 0 = outlier)
            normalized_scores = (outlier_scores + 1) / 2
            return normalized_scores
        except:
            return np.ones(n_samples) * 0.5
    def _compute_margin_score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute margin (distance to decision boundary) scores."""
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        n_samples = len(y)
        if n_samples < 20 or len(np.unique(y)) < 2:
            return np.ones(n_samples) * 0.5
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Use SVM for margin estimation
            svm = SVC(kernel='linear', random_state=42)
            svm.fit(X_scaled, y)
            # Compute distance to hyperplane
            if hasattr(svm, 'decision_function'):
                distances = svm.decision_function(X_scaled)
                if len(distances.shape) > 1:  # Multi-class
                    # Use margin from true class
                    margins = np.abs(distances[np.arange(n_samples), y])
                else:  # Binary
                    margins = np.abs(distances)
                # Normalize margins
                if margins.max() > margins.min():
                    normalized = (margins - margins.min()) / (margins.max() - margins.min())
                else:
                    normalized = np.ones_like(margins) * 0.5
                return normalized
            else:
                return np.ones(n_samples) * 0.5
        except Exception as e:
            print(f"  Warning: Could not compute margin scores: {e}")
            return np.ones(n_samples) * 0.5
    def _compute_consistency_score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute consistency scores using multiple models."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import StandardScaler
        n_samples = len(y)
        if n_samples < 20:
            return np.ones(n_samples) * 0.5
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            models = [
                RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                LogisticRegression(max_iter=1000, random_state=42),
                GaussianNB()
            ]
            all_predictions = []
            for model in models:
                # Simple train-test split for efficiency
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, _ = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_scaled)
                all_predictions.append(predictions)
            # Compute consistency (agreement between models)
            consistency_scores = np.zeros(n_samples)
            for i in range(n_samples):
                predictions_i = [pred[i] for pred in all_predictions]
                # Majority vote agreement
                from collections import Counter
                most_common = Counter(predictions_i).most_common(1)[0][0]
                agreement = np.mean([pred == most_common for pred in predictions_i])
                consistency_scores[i] = agreement
            return consistency_scores
        except Exception as e:
            print(f"  Warning: Could not compute consistency scores: {e}")
            return np.ones(n_samples) * 0.5
    def _calculate_class_quality_from_profile(self, profile: Dict) -> float:
        """Calculate class quality score from profiling results."""
        try:
            # Safely extract values with defaults
            n_samples = profile.get('n_samples', 0)
            difficulty = profile.get('difficulty_score', 0)
            outlier_analysis = profile.get('outlier_analysis', {})
            confusion_analysis = profile.get('confusion_analysis', {})
            outlier_ratio = outlier_analysis.get('outlier_ratio', 0)
            confusion_rate = confusion_analysis.get('confusion_rate', 0)
            # Calculate base score from sample size
            if n_samples >= 100:
                size_score = 1.0
            elif n_samples >= 50:
                size_score = 0.8
            elif n_samples >= 20:
                size_score = 0.6
            elif n_samples >= 10:
                size_score = 0.4
            else:
                size_score = 0.2
            # Penalize for difficulty, outliers, and confusion
            penalty = (difficulty + outlier_ratio + confusion_rate) / 3
            # Combine
            quality_score = size_score * (1 - penalty * 0.5)  # Maximum 50% penalty
            return max(0, min(1, quality_score))
        except Exception as e:
            print(f"Warning: Error calculating class quality: {e}")
            return 0.5  # Default neutral score
    def _determine_quality_grade(self, score: float) -> str:
        """Determine quality grade based on score."""
        thresholds = self.config['thresholds']
        if score >= thresholds['excellent']:
            return "EXCELLENT"
        elif score >= thresholds['good']:
            return "GOOD"
        elif score >= thresholds['fair']:
            return "FAIR"
        else:
            return "POOR"
    def _generate_quality_report(self,
                               overall_score: float,
                               quality_grade: str,
                               component_scores: Dict[str, float],
                               component_details: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        # Identify weakest component
        weakest_component = min(component_scores.items(), key=lambda x: x[1])
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(
            component_scores, component_details
        )
        report = {
            'overall_score': float(overall_score),
            'quality_grade': quality_grade,
            'component_scores': component_scores,
            'component_details': component_details,
            'weakest_component': {
                'name': weakest_component[0],
                'score': weakest_component[1]
            },
            'recommendations': recommendations,
            'n_recommendations': len(recommendations)
        }
        return report
    def _generate_quality_recommendations(self,
                                        component_scores: Dict[str, float],
                                        component_details: Dict[str, Dict]) -> List[str]:
        """Generate specific recommendations based on component scores."""
        recommendations = []
        # Check each component
        for component, score in component_scores.items():
            if score < 0.6:  # Below good threshold
                details = component_details[component]
                if component == 'dataset_health':
                    issues = details.get('issues', [])
                    if issues:
                        recommendations.append(f"Address dataset health issues: {', '.join(issues[:3])}")
                elif component == 'class_quality':
                    problematic = details.get('problematic_classes', [])
                    if problematic:
                        top_issue = problematic[0]
                        recommendations.append(
                            f"Improve quality of Class {top_issue['class']} "
                            f"(score: {top_issue['score']:.2f})"
                        )
                elif component == 'sample_quality':
                    low_quality_ratio = details.get('low_quality_ratio', 0)
                    if low_quality_ratio > 0.1:
                        recommendations.append(
                            f"Review {details.get('n_low_quality_samples', 0)} "
                            f"low-quality samples ({low_quality_ratio:.1%})"
                        )
                elif component == 'label_confidence':
                    low_conf_ratio = details.get('low_confidence_ratio', 0)
                    if low_conf_ratio > 0.2:
                        recommendations.append(
                            f"Check labels for {details.get('n_low_confidence', 0)} "
                            f"low-confidence samples ({low_conf_ratio:.1%})"
                        )
        # Add general recommendation if none specific
        if not recommendations:
            recommendations.append("Dataset quality looks good! Focus on model optimization.")
        return recommendations
    def _print_quality_summary(self, quality_report: Dict[str, Any]):
        """Print quality summary."""
        print(f"\n📊 QUALITY ASSESSMENT SUMMARY:")
        print(f"   • Overall Score: {quality_report['overall_score']:.3f}")
        print(f"   • Quality Grade: {quality_report['quality_grade']}")
        print(f"\n📈 COMPONENT SCORES:")
        for component, score in quality_report['component_scores'].items():
            print(f"   • {component.replace('_', ' ').title()}: {score:.3f}")
        print(f"\n⚠️  WEAKEST COMPONENT:")
        weak = quality_report['weakest_component']
        print(f"   • {weak['name'].replace('_', ' ').title()}: {weak['score']:.3f}")
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(quality_report['recommendations'], 1):
            print(f"   {i}. {rec}")
        print(f"\n{'='*60}")
        print("QUALITY SCORING COMPLETE")
        print(f"{'='*60}")
def main():
    """Test the QualityScorer."""
    print("=" * 60)
    print("Phase 4: Quality Scoring System")
    print("=" * 60)
    # Create test datasets
    from sklearn.datasets import make_classification
    print("\n1. Creating test datasets...")
    # High-quality dataset
    X_good, y_good = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=2,
        weights=[0.4, 0.35, 0.25],
        random_state=42
    )
    # Low-quality dataset
    X_poor, y_poor = make_classification(
        n_samples=300,
        n_features=30,
        n_classes=5,
        n_informative=10,
        n_redundant=15,
        weights=[0.5, 0.2, 0.15, 0.1, 0.05],
        flip_y=0.15,
        random_state=42
    )
    print(f"   • Good dataset: {X_good.shape[0]} samples, {X_good.shape[1]} features")
    print(f"   • Poor dataset: {X_poor.shape[0]} samples, {X_poor.shape[1]} features")
    # Initialize quality scorer
    print("\n2. Initializing quality scorer...")
    scorer = QualityScorer()
    # Score good dataset
    print("\n3. Scoring good dataset...")
    good_quality = scorer.compute_dataset_quality_score(
        X_good, y_good, "High-Quality Test Dataset"
    )
    # Score poor dataset
    print("\n4. Scoring poor dataset...")
    poor_quality = scorer.compute_dataset_quality_score(
        X_poor, y_poor, "Low-Quality Test Dataset"
    )
    # Compare datasets
    print("\n5. Quality Comparison:")
    print("-" * 60)
    print(f"{'Metric':<30} {'Good':<10} {'Poor':<10} {'Difference':<10}")
    print("-" * 60)
    metrics = [
        ('Overall Quality', 'overall_score'),
        ('Dataset Health', 'component_scores.dataset_health'),
        ('Class Quality', 'component_scores.class_quality'),
        ('Sample Quality', 'component_scores.sample_quality'),
        ('Label Confidence', 'component_scores.label_confidence')
    ]
    for metric_name, metric_path in metrics:
        good_val = good_quality
        poor_val = poor_quality
        for key in metric_path.split('.'):
            good_val = good_val[key]
            poor_val = poor_val[key]
        diff = good_val - poor_val
        print(f"{metric_name:<30} {good_val:<10.3f} {poor_val:<10.3f} {diff:<10.3f}")
    # Test sample quality scoring
    print("\n6. Testing sample-level quality scoring...")
    sample_scores = scorer.compute_sample_quality_scores(X_good, y_good)
    print(f"   • Sample scores computed: {len(sample_scores)} scores")
    print(f"   • Average sample score: {np.mean(sample_scores):.3f}")
    print(f"   • Standard deviation: {np.std(sample_scores):.3f}")
    # Identify low-quality samples
    low_quality_samples = scorer.identify_low_quality_samples(
        X_good, y_good, threshold=0.4, top_k=5
    )
    if not low_quality_samples.empty:
        print(f"\n   Top 5 lowest quality samples:")
        for idx, row in low_quality_samples.iterrows():
            print(f"      • Sample {row['sample_index']}: "
                  f"score={row['quality_score']:.3f}, "
                  f"label={row['label']}")
    print("\n" + "=" * 60)
    print("Quality Scoring Complete!")
    print("=" * 60)
    return scorer, good_quality, poor_quality
if __name__ == "__main__":
    scorer, good_quality, poor_quality = main()
