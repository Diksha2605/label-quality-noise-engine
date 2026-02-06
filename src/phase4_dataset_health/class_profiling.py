"""
Class profiling for Phase 4.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
class ClassProfiler:
    """Deep dive analysis for individual classes."""
    def __init__(self):
        """Initialize class profiler."""
        self.profiles = {}
    def profile_class(self, 
                     X: np.ndarray, 
                     y: np.ndarray, 
                     class_label: int) -> Dict[str, Any]:
        """
        Create detailed profile for a specific class.
        Args:
            X: Feature matrix
            y: Labels
            class_label: Class to profile
        Returns:
            Detailed class profile
        """
        # Get class samples
        class_mask = y == class_label
        X_class = X[class_mask]
        n_class_samples = len(X_class)
        if n_class_samples == 0:
            return {
                'class_label': int(class_label),
                'n_samples': 0,
                'percentage': 0.0,
                'feature_stats': {},
                'outlier_analysis': {'n_outliers': 0, 'outlier_ratio': 0.0, 'outlier_score': 0},
                'boundary_analysis': {'boundary_score': 0, 'avg_distance_to_other': 0, 'min_distance_to_other': 0},
                'confusion_analysis': {'confusion_rate': 0, 'confused_with': [], 'confusion_matrix': {}, 'n_confusions': 0},
                'difficulty_score': 0.0
            }
        # Basic statistics
        profile = {
            'class_label': int(class_label),
            'n_samples': n_class_samples,
            'percentage': n_class_samples / len(y),
            'feature_stats': self._calculate_class_feature_stats(X_class),
            'outlier_analysis': self._analyze_class_outliers(X_class),
            'boundary_analysis': self._analyze_class_boundaries(X, y, class_label),
            'confusion_analysis': self._analyze_class_confusions(X, y, class_label)
        }
        return profile
    def profile_all_classes(self, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           target_names: Optional[List[str]] = None) -> Dict[int, Dict]:
        """
        Profile all classes in the dataset.
        Args:
            X: Feature matrix
            y: Labels
            target_names: Optional class names
        Returns:
            Dictionary of class profiles
        """
        unique_classes = np.unique(y)
        if target_names is None:
            target_names = [f'Class {i}' for i in unique_classes]
        print(f"Profiling {len(unique_classes)} classes...")
        all_profiles = {}
        for i, class_label in enumerate(unique_classes):
            print(f"  Processing class {class_label} ({target_names[i]})...")
            profile = self.profile_class(X, y, class_label)
            profile['class_name'] = target_names[i]
            # Calculate class difficulty score
            profile['difficulty_score'] = self._calculate_class_difficulty(
                profile['boundary_analysis'],
                profile['confusion_analysis']
            )
            all_profiles[class_label] = profile
        # Calculate relative metrics
        self._calculate_relative_metrics(all_profiles)
        return all_profiles
    def identify_problematic_classes(self, 
                                    class_profiles: Dict[int, Dict],
                                    threshold: float = 0.7) -> List[Dict]:
        """
        Identify classes with potential problems.
        Args:
            class_profiles: Dictionary of class profiles
            threshold: Difficulty score threshold
        Returns:
            List of problematic class profiles
        """
        problematic_classes = []
        for class_label, profile in class_profiles.items():
            difficulty = profile.get('difficulty_score', 0)
            # Check for various issues
            issues = []
            # Small class size
            if profile['n_samples'] < 50:
                issues.append(f"Small size ({profile['n_samples']} samples)")
            # High difficulty
            if difficulty > threshold:
                issues.append(f"High difficulty (score: {difficulty:.2f})")
            # Many outliers
            outlier_ratio = profile.get('outlier_analysis', {}).get('outlier_ratio', 0)
            if outlier_ratio > 0.1:
                issues.append(f"Many outliers ({outlier_ratio:.1%})")
            # High confusion
            confusion_rate = profile.get('confusion_analysis', {}).get('confusion_rate', 0)
            if confusion_rate > 0.3:
                issues.append(f"High confusion ({confusion_rate:.1%})")
            # Poor separation
            boundary_score = profile.get('boundary_analysis', {}).get('boundary_score', 0)
            if boundary_score < 0.5:
                issues.append(f"Poor separation (score: {boundary_score:.2f})")
            if issues:
                problematic_profile = profile.copy()
                problematic_profile['issues'] = issues
                problematic_profile['n_issues'] = len(issues)
                problematic_profile['priority'] = self._calculate_priority(
                    len(issues), difficulty, outlier_ratio
                )
                problematic_classes.append(problematic_profile)
        # Sort by priority
        problematic_classes.sort(key=lambda x: x['priority'], reverse=True)
        return problematic_classes
    def generate_class_report(self, 
                             X: np.ndarray, 
                             y: np.ndarray,
                             dataset_name: str = "Dataset",
                             target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive class-level report.
        Args:
            X: Feature matrix
            y: Labels
            dataset_name: Name of dataset
            target_names: Optional class names
        Returns:
            Comprehensive class report
        """
        print(f"\n{'='*60}")
        print(f"CLASS PROFILING REPORT: {dataset_name}")
        print(f"{'='*60}")
        # Profile all classes
        class_profiles = self.profile_all_classes(X, y, target_names)
        # Identify problematic classes
        problematic_classes = self.identify_problematic_classes(class_profiles)
        # Generate summary statistics
        summary = self._generate_summary_statistics(class_profiles, problematic_classes)
        # Print report
        print(f"\n📊 CLASS SUMMARY:")
        print(f"   • Total classes: {summary['n_classes']}")
        print(f"   • Total samples: {summary['total_samples']}")
        print(f"   • Problematic classes: {summary['n_problematic_classes']} ({summary['problematic_ratio']:.1%})")
        if problematic_classes:
            print(f"\n⚠️  PROBLEMATIC CLASSES (sorted by priority):")
            for i, profile in enumerate(problematic_classes[:5], 1):  # Top 5
                print(f"\n   {i}. Class {profile['class_label']} ({profile.get('class_name', 'N/A')}):")
                print(f"      • Samples: {profile['n_samples']} ({profile['percentage']:.1%})")
                print(f"      • Difficulty score: {profile['difficulty_score']:.2f}")
                print(f"      • Issues: {', '.join(profile['issues'])}")
                print(f"      • Priority: {profile['priority']:.2f}")
        else:
            print(f"\n✅ No problematic classes detected!")
        # Print class distribution
        print(f"\n📈 CLASS DISTRIBUTION:")
        for class_label, profile in class_profiles.items():
            class_name = profile.get('class_name', f'Class {class_label}')
            print(f"   • {class_name}: {profile['n_samples']} samples ({profile['percentage']:.1%})")
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = self._generate_recommendations(problematic_classes, summary)
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        print(f"\n{'='*60}")
        print("CLASS PROFILING COMPLETE")
        print(f"{'='*60}")
        # Compile full report
        full_report = {
            'dataset_name': dataset_name,
            'summary': summary,
            'class_profiles': class_profiles,
            'problematic_classes': problematic_classes,
            'recommendations': recommendations,
            'n_classes': len(class_profiles),
            'n_problematic': len(problematic_classes)
        }
        return full_report
    def _calculate_class_feature_stats(self, X_class: np.ndarray) -> Dict[str, Any]:
        """Calculate feature statistics for a class."""
        n_samples, n_features = X_class.shape
        if n_samples == 0:
            return {}
        feature_means = np.mean(X_class, axis=0)
        feature_stds = np.std(X_class, axis=0)
        feature_skews = stats.skew(X_class, axis=0) if n_samples > 2 else np.zeros(n_features)
        feature_kurtosis = stats.kurtosis(X_class, axis=0) if n_samples > 3 else np.zeros(n_features)
        # Identify most distinctive features
        if n_features > 1:
            # Calculate coefficient of variation
            cv = feature_stds / (np.abs(feature_means) + 1e-10)
            # Find features with highest and lowest variation
            distinctive_idx = np.argsort(cv)[-5:]  # Top 5 most variable
            stable_idx = np.argsort(cv)[:5]        # Top 5 most stable
        else:
            distinctive_idx = stable_idx = np.array([0])
        return {
            'mean': feature_means.tolist(),
            'std': feature_stds.tolist(),
            'skew': feature_skews.tolist(),
            'kurtosis': feature_kurtosis.tolist(),
            'distinctive_features': distinctive_idx.tolist(),
            'stable_features': stable_idx.tolist(),
            'avg_std': float(np.mean(feature_stds)),
            'max_std': float(np.max(feature_stds))
        }
    def _analyze_class_outliers(self, X_class: np.ndarray) -> Dict[str, Any]:
        """Analyze outliers within a class."""
        from sklearn.neighbors import LocalOutlierFactor
        n_samples = len(X_class)
        if n_samples < 10:
            return {
                'n_outliers': 0,
                'outlier_ratio': 0.0,
                'outlier_indices': [],
                'outlier_score': 0
            }
        try:
            # Use LOF for outlier detection
            lof = LocalOutlierFactor(n_neighbors=min(20, n_samples - 1))
            outlier_labels = lof.fit_predict(X_class)
            n_outliers = np.sum(outlier_labels == -1)
            outlier_ratio = n_outliers / n_samples if n_samples > 0 else 0.0
            outlier_indices = np.where(outlier_labels == -1)[0]
            return {
                'n_outliers': int(n_outliers),
                'outlier_ratio': float(outlier_ratio),
                'outlier_indices': outlier_indices.tolist(),
                'outlier_score': float(outlier_ratio * 100)  # Score from 0-100
            }
        except:
            return {
                'n_outliers': 0,
                'outlier_ratio': 0.0,
                'outlier_indices': [],
                'outlier_score': 0
            }
    def _analyze_class_boundaries(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray, 
                                 class_label: int) -> Dict[str, Any]:
        """Analyze class boundary characteristics."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        class_mask = y == class_label
        other_mask = y != class_label
        if class_mask.sum() == 0 or other_mask.sum() == 0:
            return {
                'boundary_score': 0.0,
                'avg_distance_to_other': 0.0,
                'min_distance_to_other': 0.0,
                'distances': []
            }
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Get class and other samples
        X_class = X_scaled[class_mask]
        X_other = X_scaled[other_mask]
        # Find nearest neighbors from other classes
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_other)
        distances, _ = nbrs.kneighbors(X_class)
        avg_distance = distances.mean() if len(distances) > 0 else 0.0
        min_distance = distances.min() if len(distances) > 0 else 0.0
        # Normalize distances (empirical scaling)
        if avg_distance > 0:
            # Higher distance = better separation
            boundary_score = min(1.0, avg_distance / 5.0)  # Scale to 0-1
        else:
            boundary_score = 0.0
        return {
            'boundary_score': float(boundary_score),
            'avg_distance_to_other': float(avg_distance),
            'min_distance_to_other': float(min_distance),
            'distances': distances.flatten().tolist() if len(distances) > 0 else []
        }
    def _analyze_class_confusions(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray, 
                                 class_label: int) -> Dict[str, Any]:
        """Analyze which classes this class gets confused with."""
        from sklearn.neighbors import KNeighborsClassifier
        class_mask = y == class_label
        if class_mask.sum() < 10:
            return {
                'confusion_rate': 0.0,
                'confused_with': [],
                'confusion_matrix': {},
                'n_confusions': 0
            }
        # Use k-NN to predict class labels
        knn = KNeighborsClassifier(n_neighbors=5)
        # We'll use leave-one-out for this class
        X_class = X[class_mask]
        y_class = y[class_mask]
        confusions = {}
        for i in range(len(X_class)):
            # Leave-one-out cross-validation
            X_train = np.delete(X_class, i, axis=0)
            y_train = np.delete(y_class, i)
            X_test = X_class[i:i+1]
            knn.fit(X_train, y_train)
            pred = knn.predict(X_test)[0]
            if pred != class_label:
                confusions[pred] = confusions.get(pred, 0) + 1
        total_confusions = sum(confusions.values())
        confusion_rate = total_confusions / len(X_class) if len(X_class) > 0 else 0.0
        # Sort by frequency
        confused_with = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        return {
            'confusion_rate': float(confusion_rate),
            'confused_with': [(int(k), int(v)) for k, v in confused_with],
            'confusion_matrix': {int(k): int(v) for k, v in confusions.items()},
            'n_confusions': int(total_confusions)
        }
    def _calculate_class_difficulty(self, 
                                   boundary_analysis: Dict,
                                   confusion_analysis: Dict) -> float:
        """Calculate overall difficulty score for a class (0-1)."""
        boundary_score = boundary_analysis.get('boundary_score', 0.0)
        confusion_rate = confusion_analysis.get('confusion_rate', 0.0)
        # Invert boundary score (lower boundary score = more difficult)
        boundary_difficulty = 1.0 - boundary_score
        # Combine difficulties
        difficulty = (boundary_difficulty + confusion_rate) / 2
        return min(1.0, max(0, difficulty))
    def _calculate_relative_metrics(self, class_profiles: Dict[int, Dict]):
        """Calculate relative metrics between classes."""
        # Calculate average feature std across all classes
        all_avg_stds = []
        for profile in class_profiles.values():
            feature_stats = profile.get('feature_stats', {})
            avg_std = feature_stats.get('avg_std', 0)
            all_avg_stds.append(avg_std)
        global_avg_std = np.mean(all_avg_stds) if all_avg_stds else 0.0
        for class_label, profile in class_profiles.items():
            # Calculate relative feature variability
            feature_stats = profile.get('feature_stats', {})
            class_avg_std = feature_stats.get('avg_std', 0.0)
            if global_avg_std > 0:
                relative_variability = class_avg_std / global_avg_std
            else:
                relative_variability = 1.0
            if 'feature_stats' not in profile:
                profile['feature_stats'] = {}
            profile['feature_stats']['relative_variability'] = float(relative_variability)
            # Flag if class is unusually variable or stable
            if relative_variability > 1.5:
                profile['feature_stats']['variability_status'] = 'HIGHLY_VARIABLE'
            elif relative_variability < 0.5:
                profile['feature_stats']['variability_status'] = 'VERY_STABLE'
            else:
                profile['feature_stats']['variability_status'] = 'NORMAL'
    def _calculate_priority(self, 
                           n_issues: int, 
                           difficulty: float,
                           outlier_ratio: float) -> float:
        """Calculate priority score for addressing class issues."""
        priority = (
            n_issues * 0.3 +
            difficulty * 0.4 +
            outlier_ratio * 0.3
        )
        return min(1.0, priority)
    def _generate_summary_statistics(self, 
                                    class_profiles: Dict[int, Dict],
                                    problematic_classes: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the class report."""
        n_classes = len(class_profiles)
        total_samples = sum(p.get('n_samples', 0) for p in class_profiles.values())
        # Calculate average difficulty
        difficulties = []
        for profile in class_profiles.values():
            difficulty = profile.get('difficulty_score', 0.0)
            difficulties.append(difficulty)
        avg_difficulty = np.mean(difficulties) if difficulties else 0.0
        # Find easiest and hardest classes
        easiest_label = None
        easiest_difficulty = float('inf')
        hardest_label = None
        hardest_difficulty = -float('inf')
        for class_label, profile in class_profiles.items():
            difficulty = profile.get('difficulty_score', 0.0)
            if difficulty < easiest_difficulty:
                easiest_difficulty = difficulty
                easiest_label = class_label
            if difficulty > hardest_difficulty:
                hardest_difficulty = difficulty
                hardest_label = class_label
        return {
            'n_classes': n_classes,
            'total_samples': total_samples,
            'n_problematic_classes': len(problematic_classes),
            'problematic_ratio': len(problematic_classes) / n_classes if n_classes > 0 else 0.0,
            'avg_difficulty': float(avg_difficulty),
            'easiest_class': {
                'label': easiest_label,
                'difficulty': easiest_difficulty if easiest_label is not None else 0.0
            },
            'hardest_class': {
                'label': hardest_label,
                'difficulty': hardest_difficulty if hardest_label is not None else 0.0
            }
        }
    def _generate_recommendations(self, 
                                 problematic_classes: List[Dict],
                                 summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on class analysis."""
        recommendations = []
        if problematic_classes:
            # Get top 3 problematic classes
            top_problematic = problematic_classes[:3]
            for profile in top_problematic:
                class_name = profile.get('class_name', f'Class {profile.get("class_label", "Unknown")}')
                recs = []
                # Generate specific recommendations
                if profile.get('n_samples', 0) < 50:
                    recs.append(f"collect more samples for {class_name}")
                if 'High difficulty' in ' '.join(profile.get('issues', [])):
                    recs.append(f"review labeling for {class_name}")
                if 'Many outliers' in ' '.join(profile.get('issues', [])):
                    recs.append(f"check data quality for {class_name}")
                if 'High confusion' in ' '.join(profile.get('issues', [])):
                    recs.append(f"consider merging {class_name} with similar classes")
                if recs:
                    recommendations.append(f"For {class_name}: {', '.join(recs)}")
        # General recommendations
        if summary.get('avg_difficulty', 0) > 0.7:
            recommendations.append("Consider feature engineering to improve separability")
        if summary.get('problematic_ratio', 0) > 0.5:
            recommendations.append("Dataset has many problematic classes - consider comprehensive review")
        if not recommendations:
            recommendations.append("Classes look generally healthy. Focus on overall model improvement.")
        return recommendations
def main():
    """Test the ClassProfiler."""
    print("=" * 60)
    print("Phase 4: Class Profiling")
    print("=" * 60)
    # Create test dataset
    from sklearn.datasets import make_classification
    print("\n1. Creating test dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_informative=15,
        n_clusters_per_class=2,
        weights=[0.4, 0.25, 0.2, 0.1, 0.05],  # Imbalanced
        random_state=42
    )
    print(f"   • Samples: {X.shape[0]}")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Classes: {len(np.unique(y))}")
    # Initialize profiler
    print("\n2. Initializing class profiler...")
    profiler = ClassProfiler()
    # Generate class report
    print("\n3. Generating class report...")
    class_report = profiler.generate_class_report(
        X, y,
        dataset_name="Test Classification Dataset",
        target_names=[f"Category_{i}" for i in range(5)]
    )
    # Test individual class profiling
    print("\n4. Profiling specific classes...")
    # Profile the largest class
    unique_classes, counts = np.unique(y, return_counts=True)
    largest_class = unique_classes[np.argmax(counts)]
    print(f"\n   Detailed profile for largest class (Class {largest_class}):")
    class_profile = profiler.profile_class(X, y, largest_class)
    print(f"   • Samples: {class_profile.get('n_samples', 0)}")
    print(f"   • Difficulty score: {class_profile.get('difficulty_score', 0):.3f}")
    outlier_analysis = class_profile.get('outlier_analysis', {})
    print(f"   • Outlier ratio: {outlier_analysis.get('outlier_ratio', 0):.3f}")
    print("\n" + "=" * 60)
    print("Class Profiling Complete!")
    print("=" * 60)
    return profiler, class_report
if __name__ == "__main__":
    profiler, class_report = main()
