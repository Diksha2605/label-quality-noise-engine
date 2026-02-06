"""
Visualization utilities for Phase 1.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.cm as cm
from matplotlib.colors import Normalize
class DataVisualizer:
    """Advanced visualization tools for dataset exploration."""
    def __init__(self, style: str = 'whitegrid', palette: str = 'husl'):
        """Initialize visualizer with style and palette."""
        self.style = style
        self.palette = palette
        sns.set_style(style)
        sns.set_palette(palette)
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
    def plot_class_comparison(self, datasets: Dict[str, Dict], 
                             metric: str = 'imbalance_ratio') -> plt.Figure:
        """
        Compare multiple datasets by a metric.
        Args:
            datasets: Dictionary of dataset info dictionaries
            metric: Metric to compare ('imbalance_ratio', 'n_samples', 'n_features', 'n_classes')
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # Prepare data for comparison
        comparison_data = []
        for name, info in datasets.items():
            # Calculate label stats if not already calculated
            if 'label_stats' in info:
                # Use pre-calculated stats
                label_stats = info['label_stats']
                imbalance = label_stats.get('imbalance_ratio', 1.0)
            else:
                # Calculate stats
                y = info['y']
                unique_labels, counts = np.unique(y, return_counts=True)
                imbalance = counts.max() / counts.min() if counts.min() > 0 else float('inf')
            comparison_data.append({
                'dataset': name,
                'n_samples': info.get('n_samples', len(info['y'])),
                'n_features': info.get('n_features', info['X'].shape[1]),
                'n_classes': info.get('n_classes', len(np.unique(info['y']))),
                'imbalance_ratio': imbalance,
                'description': info.get('description', name)
            })
        df_comparison = pd.DataFrame(comparison_data)
        # Plot 1: Bar chart of selected metric
        ax1 = axes[0]
        datasets_sorted = df_comparison.sort_values(metric, ascending=False)
        bars = ax1.bar(range(len(datasets_sorted)), datasets_sorted[metric], 
                      color=sns.color_palette(self.palette, len(datasets_sorted)))
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f'Dataset Comparison by {metric.replace("_", " ").title()}')
        ax1.set_xticks(range(len(datasets_sorted)))
        ax1.set_xticklabels(datasets_sorted['dataset'], rotation=45, ha='right')
        # Add value labels
        for bar, value in zip(bars, datasets_sorted[metric]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        # Plot 2: Scatter plot of dataset characteristics
        ax2 = axes[1]
        scatter = ax2.scatter(df_comparison['n_samples'], 
                             df_comparison['n_features'],
                             c=df_comparison['imbalance_ratio'],
                             s=df_comparison['n_classes'] * 100,
                             alpha=0.7,
                             edgecolors='black',
                             linewidth=0.5,
                             cmap='viridis')
        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Dataset Characteristics\n(size = n_classes, color = imbalance)')
        ax2.grid(True, alpha=0.3)
        # Add dataset labels
        for idx, row in df_comparison.iterrows():
            ax2.text(row['n_samples'], row['n_features'], row['dataset'],
                    fontsize=9, ha='center', va='center')
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='Imbalance Ratio')
        plt.tight_layout()
        return fig
    def plot_label_quality_indicators(self, X: np.ndarray, y: np.ndarray,
                                     dataset_name: str = 'Dataset',
                                     n_neighbors: int = 5) -> plt.Figure:
        """
        Plot indicators of potential label issues.
        Args:
            X: Feature matrix
            y: Labels
            dataset_name: Name for title
            n_neighbors: Number of neighbors for k-NN analysis
        Returns:
            Matplotlib figure
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # 1. Label distribution (redundant but useful)
        unique_labels, counts = np.unique(y, return_counts=True)
        axes[0].bar(unique_labels, counts, color=sns.color_palette(self.palette, len(unique_labels)))
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Label Distribution')
        axes[0].grid(True, alpha=0.3)
        # Add percentage labels
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            percentage = count / len(y) * 100
            axes[0].text(label, count, f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontsize=9)
        # 2. Class separation (using first 2 PCA components)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                                 cmap='tab10', alpha=0.6, 
                                 edgecolors='w', linewidth=0.5)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1].set_title('Class Separation (PCA)')
        axes[1].grid(True, alpha=0.3)
        # 3. Nearest neighbor label agreement
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        # Calculate label agreement with neighbors (excluding self)
        label_agreement = []
        for i in range(len(y)):
            neighbor_labels = y[indices[i, 1:]]  # Exclude self
            agreement = np.mean(neighbor_labels == y[i])
            label_agreement.append(agreement)
        label_agreement = np.array(label_agreement)
        # Plot histogram of label agreement
        axes[2].hist(label_agreement, bins=20, alpha=0.7, edgecolor='black')
        axes[2].axvline(np.mean(label_agreement), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(label_agreement):.3f}')
        axes[2].set_xlabel('Label Agreement with Neighbors')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'Label Consistency (k={n_neighbors})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        # 4. Class overlap visualization
        # Calculate average distance to nearest neighbor from different class
        class_distances = []
        for class_label in unique_labels:
            class_mask = y == class_label
            other_mask = y != class_label
            if np.sum(class_mask) > 0 and np.sum(other_mask) > 0:
                # Find nearest neighbor from different class for each sample
                nbrs_class = NearestNeighbors(n_neighbors=1).fit(X_scaled[other_mask])
                distances_to_other = nbrs_class.kneighbors(X_scaled[class_mask])[0]
                avg_distance = np.mean(distances_to_other)
                class_distances.append(avg_distance)
            else:
                class_distances.append(0)
        axes[3].bar(unique_labels, class_distances, 
                   color=sns.color_palette(self.palette, len(unique_labels)))
        axes[3].set_xlabel('Class')
        axes[3].set_ylabel('Avg Distance to Nearest Other Class')
        axes[3].set_title('Class Separability')
        axes[3].grid(True, alpha=0.3)
        plt.suptitle(f'Label Quality Indicators - {dataset_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    def plot_dataset_health_radar(self, dataset_stats: Dict[str, Any]) -> plt.Figure:
        """
        Create radar chart showing dataset health metrics.
        Args:
            dataset_stats: Dictionary with dataset statistics
        Returns:
            Matplotlib figure
        """
        # Metrics to include in radar chart
        metrics = ['class_balance', 'feature_quality', 'sample_sufficiency', 
                  'label_consistency', 'separability']
        # Example values (should be calculated from actual data)
        values = [0.7, 0.8, 0.9, 0.6, 0.75]  # Placeholder
        # Number of variables
        N = len(metrics)
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        # Initialise the spider plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], metrics, size=12)
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
                  color="grey", size=10)
        plt.ylim(0, 1)
        # Plot data
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
               label='Dataset Health', color='blue')
        ax.fill(angles, values, 'b', alpha=0.1)
        # Add title
        plt.title('Dataset Health Radar Chart', size=16, y=1.1)
        return fig
    def save_visualization(self, fig: plt.Figure, filename: str, 
                          directory: str = 'reports/visualizations'):
        """
        Save visualization to file.
        Args:
            fig: Matplotlib figure
            filename: Output filename (without extension)
            directory: Output directory
        """
        import os
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f'{filename}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {filepath}")
        # Also save as PDF for vector graphics
        pdf_path = os.path.join(directory, f'{filename}.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        return filepath
def main():
    """Test the DataVisualizer."""
    print("=" * 60)
    print("Phase 1: Data Visualization Module")
    print("=" * 60)
    # Create sample datasets
    from sklearn.datasets import load_digits, load_iris, load_wine
    datasets = {
        'digits': {
            'X': load_digits().data,
            'y': load_digits().target,
            'description': 'Digits Dataset'
        },
        'iris': {
            'X': load_iris().data,
            'y': load_iris().target,
            'description': 'Iris Dataset'
        },
        'wine': {
            'X': load_wine().data,
            'y': load_wine().target,
            'description': 'Wine Dataset'
        }
    }
    # Initialize visualizer
    visualizer = DataVisualizer()
    # Create class comparison
    print("\nCreating dataset comparison...")
    fig1 = visualizer.plot_class_comparison(datasets)
    # Create label quality indicators for digits
    print("\nCreating label quality indicators for digits dataset...")
    fig2 = visualizer.plot_label_quality_indicators(
        datasets['digits']['X'], 
        datasets['digits']['y'],
        dataset_name='Digits'
    )
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    plt.show()
    return visualizer
if __name__ == "__main__":
    visualizer = main()
