"""
Exploratory Analysis for Phase 1.
Analyze label distributions and dataset characteristics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class ExploratoryAnalyzer:
    """Perform exploratory data analysis on datasets."""
    def __init__(self, style: str = 'whitegrid'):
        """Initialize with plotting style."""
        self.style = style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
    def analyze_label_distribution(self, y: np.ndarray, 
                                   target_names: Optional[List[str]] = None,
                                   dataset_name: str = 'Dataset') -> Dict[str, Any]:
        """
        Analyze label distribution.
        Args:
            y: Target labels
            target_names: Optional names for each class
            dataset_name: Name of dataset for reporting
        Returns:
            Dictionary with label statistics
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(unique_labels)
        if target_names is None:
            target_names = [f'Class {i}' for i in unique_labels]
        elif len(target_names) != n_classes:
            target_names = [f'Class {i}' for i in unique_labels]
        # Calculate statistics
        proportions = counts / n_samples
        imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
        stats = {
            'dataset_name': dataset_name,
            'n_samples': n_samples,
            'n_classes': n_classes,
            'unique_labels': unique_labels,
            'counts': counts,
            'proportions': proportions,
            'target_names': target_names,
            'imbalance_ratio': imbalance_ratio,
            'is_balanced': imbalance_ratio < 2.0,  # Arbitrary threshold
            'min_samples': counts.min(),
            'max_samples': counts.max(),
            'mean_samples': counts.mean(),
            'std_samples': counts.std()
        }
        return stats
    def plot_label_distribution(self, y: np.ndarray, 
                                target_names: Optional[List[str]] = None,
                                dataset_name: str = 'Dataset',
                                ax: Optional[plt.Axes] = None,
                                show_values: bool = True) -> plt.Figure:
        """
        Plot label distribution.
        Returns:
            Matplotlib figure
        """
        stats = self.analyze_label_distribution(y, target_names, dataset_name)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        # Create bar plot
        bars = ax.bar(range(len(stats['unique_labels'])), stats['counts'], 
                     color=sns.color_palette("husl", len(stats['unique_labels'])))
        # Add labels and title
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Label Distribution - {dataset_name}\n'
                    f'Total: {stats["n_samples"]} samples, {stats["n_classes"]} classes')
        ax.set_xticks(range(len(stats['unique_labels'])))
        ax.set_xticklabels(stats['target_names'], rotation=45, ha='right')
        # Add value labels on bars
        if show_values:
            for bar, count, prop in zip(bars, stats['counts'], stats['proportions']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{count}\n({prop:.1%})', ha='center', va='bottom', fontsize=9)
        # Add imbalance info
        if stats['imbalance_ratio'] > 2.0:
            ax.text(0.02, 0.98, f"Imbalance ratio: {stats['imbalance_ratio']:.2f}",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        plt.tight_layout()
        return fig
    def analyze_feature_statistics(self, X: np.ndarray, 
                                   feature_names: Optional[List[str]] = None,
                                   dataset_name: str = 'Dataset') -> pd.DataFrame:
        """
        Calculate basic statistics for each feature.
        Returns:
            DataFrame with feature statistics
        """
        if feature_names is None:
            n_features = X.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
        stats_data = []
        for i, (feature, name) in enumerate(zip(X.T, feature_names)):
            stats = {
                'feature': name,
                'mean': np.mean(feature),
                'std': np.std(feature),
                'min': np.min(feature),
                'max': np.max(feature),
                'median': np.median(feature),
                'q1': np.percentile(feature, 25),
                'q3': np.percentile(feature, 75),
                'missing': np.isnan(feature).sum() if np.any(np.isnan(feature)) else 0,
                'zeros': np.sum(feature == 0)
            }
            stats_data.append(stats)
        df_stats = pd.DataFrame(stats_data)
        df_stats['missing_pct'] = df_stats['missing'] / len(X) * 100
        return df_stats
    def plot_feature_distributions(self, X: np.ndarray, 
                                   feature_names: Optional[List[str]] = None,
                                   dataset_name: str = 'Dataset',
                                   n_cols: int = 4) -> plt.Figure:
        """
        Plot distribution of features.
        Returns:
            Matplotlib figure
        """
        if feature_names is None:
            n_features = X.shape[1]
            feature_names = [f'Feature {i}' for i in range(n_features)]
        n_features = X.shape[1]
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        for i, (feature, name) in enumerate(zip(X.T, feature_names)):
            ax = axes[i]
            # Plot histogram
            ax.hist(feature, bins=30, alpha=0.7, edgecolor='black')
            # Add statistics
            mean = np.mean(feature)
            std = np.std(feature)
            ax.axvline(mean, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean:.2f}')
            ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1, alpha=0.7)
            ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1, alpha=0.7)
            ax.set_title(f'{name}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        plt.suptitle(f'Feature Distributions - {dataset_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    def generate_dataset_report(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for a dataset.
        Args:
            dataset_info: Dictionary from DataLoader
        Returns:
            Comprehensive report dictionary
        """
        X = dataset_info['X']
        y = dataset_info['y']
        feature_names = dataset_info['feature_names']
        target_names = dataset_info['target_names']
        dataset_name = dataset_info.get('description', 'Dataset')
        print(f"\nGenerating report for: {dataset_name}")
        print("-" * 40)
        # 1. Label analysis
        label_stats = self.analyze_label_distribution(y, target_names, dataset_name)
        print("Label Statistics:")
        print(f"  • Samples: {label_stats['n_samples']:,}")
        print(f"  • Classes: {label_stats['n_classes']}")
        print(f"  • Imbalance ratio: {label_stats['imbalance_ratio']:.2f}")
        print(f"  • Balanced: {'Yes' if label_stats['is_balanced'] else 'No'}")
        print(f"  • Min class size: {label_stats['min_samples']}")
        print(f"  • Max class size: {label_stats['max_samples']}")
        # 2. Feature analysis
        feature_stats = self.analyze_feature_statistics(X, feature_names, dataset_name)
        print("\nFeature Statistics:")
        print(f"  • Number of features: {len(feature_names)}")
        print(f"  • Feature means range: [{feature_stats['mean'].min():.2f}, {feature_stats['mean'].max():.2f}]")
        print(f"  • Feature std range: [{feature_stats['std'].min():.2f}, {feature_stats['std'].max():.2f}]")
        if feature_stats['missing_pct'].sum() > 0:
            print(f"  • Missing values: {feature_stats['missing'].sum():,} total "
                  f"({feature_stats['missing_pct'].sum():.1f}%)")
        # 3. Create comprehensive report
        report = {
            'dataset_info': dataset_info,
            'label_stats': label_stats,
            'feature_stats': feature_stats,
            'summary': {
                'n_samples': label_stats['n_samples'],
                'n_features': len(feature_names),
                'n_classes': label_stats['n_classes'],
                'imbalance_ratio': label_stats['imbalance_ratio'],
                'is_balanced': label_stats['is_balanced'],
                'has_missing_values': feature_stats['missing'].sum() > 0,
                'feature_correlation': self._calculate_feature_correlation(X)
            }
        }
        return report
    def _calculate_feature_correlation(self, X: np.ndarray) -> float:
        """Calculate average absolute correlation between features."""
        if X.shape[1] < 2:
            return 0.0
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
        # Calculate average absolute correlation
        avg_corr = np.mean(np.abs(corr_matrix))
        return avg_corr
    def plot_comprehensive_report(self, dataset_info: Dict[str, Any], 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization report.
        Returns:
            Matplotlib figure
        """
        X = dataset_info['X']
        y = dataset_info['y']
        feature_names = dataset_info['feature_names']
        target_names = dataset_info['target_names']
        dataset_name = dataset_info.get('description', 'Dataset')
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        # 1. Label distribution (top left)
        ax1 = plt.subplot(2, 2, 1)
        self.plot_label_distribution(y, target_names, dataset_name, ax=ax1)
        # 2. Feature distributions (top right - example features)
        ax2 = plt.subplot(2, 2, 2)
        # Show first few features
        n_sample_features = min(3, X.shape[1])
        sample_X = X[:, :n_sample_features]
        sample_names = feature_names[:n_sample_features]
        for i, (feature, name) in enumerate(zip(sample_X.T, sample_names)):
            ax2.hist(feature, bins=30, alpha=0.5, label=name)
        ax2.set_title(f'Sample Feature Distributions (first {n_sample_features})')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # 3. Class separation visualization (bottom left)
        ax3 = plt.subplot(2, 2, 3)
        # Use PCA for dimensionality reduction if many features
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
            ax3.set_title('Class Separation (PCA)')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        else:
            X_reduced = X
            ax3.set_title('Class Separation')
            ax3.set_xlabel(feature_names[0])
            ax3.set_ylabel(feature_names[1] if X.shape[1] > 1 else '')
        # Scatter plot by class
        scatter = ax3.scatter(X_reduced[:, 0], X_reduced[:, 1] if X_reduced.shape[1] > 1 else np.zeros_like(X_reduced[:, 0]),
                           c=y, cmap='tab10', alpha=0.6, edgecolors='w', linewidth=0.5)
        ax3.grid(True, alpha=0.3)
        # 4. Statistics summary (bottom right)
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        # Calculate and display statistics
        label_stats = self.analyze_label_distribution(y, target_names, dataset_name)
        feature_stats = self.analyze_feature_statistics(X, feature_names, dataset_name)
        stats_text = (
            f"Dataset: {dataset_name}\n"
            f"Samples: {label_stats['n_samples']:,}\n"
            f"Features: {len(feature_names)}\n"
            f"Classes: {label_stats['n_classes']}\n"
            f"Imbalance ratio: {label_stats['imbalance_ratio']:.2f}\n"
            f"Balanced: {'Yes' if label_stats['is_balanced'] else 'No'}\n"
            f"Avg feature mean: {feature_stats['mean'].mean():.2f}\n"
            f"Avg feature std: {feature_stats['std'].mean():.2f}\n"
            f"Missing values: {feature_stats['missing'].sum():,}"
        )
        ax4.text(0.1, 0.5, stats_text, fontsize=11, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.suptitle(f'Dataset Exploration Report: {dataset_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Report saved to: {save_path}")
        return fig
def main():
    """Test the ExploratoryAnalyzer."""
    print("=" * 60)
    print("Phase 1: Exploratory Analysis Module")
    print("=" * 60)
    # Create sample data
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    dataset_info = {
        'X': X,
        'y': y,
        'feature_names': digits.feature_names,
        'target_names': digits.target_names,
        'description': 'Digits Dataset'
    }
    # Initialize analyzer
    analyzer = ExploratoryAnalyzer()
    # Generate report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_dataset_report(dataset_info)
    # Create visualization
    print("\nCreating visualization report...")
    fig = analyzer.plot_comprehensive_report(dataset_info)
    print("\n" + "=" * 60)
    print("Exploratory Analysis Complete!")
    print("=" * 60)
    plt.show()
    return analyzer, report
if __name__ == "__main__":
    analyzer, report = main()
