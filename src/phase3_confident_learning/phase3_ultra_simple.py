"""
Ultra-simple Phase 3 runner with guaranteed JSON serialization
"""
import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
print("=" * 80)
print("PHASE 3: ULTRA-SIMPLE NOISE ESTIMATION")
print("=" * 80)
# Create output directory
output_dir = 'reports/phase3_ultra_simple'
os.makedirs(output_dir, exist_ok=True)
# Step 1: Create synthetic data (always works)
print("\n1. Creating synthetic dataset...")
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Create dataset
X, y_clean = make_classification(
    n_samples=200,
    n_features=10,
    n_classes=3,
    n_informative=6,
    random_state=42
)
# Add noise
np.random.seed(42)
noise_mask = np.random.rand(len(y_clean)) < 0.15
y_noisy = y_clean.copy()
for i in np.where(noise_mask)[0]:
    other_classes = [c for c in np.unique(y_clean) if c != y_clean[i]]
    y_noisy[i] = np.random.choice(other_classes)
print(f"  • Samples: {X.shape[0]}")
print(f"  • Features: {X.shape[1]}")
print(f"  • Classes: {len(np.unique(y_clean))}")
print(f"  • Added noise: {noise_mask.sum()} samples ({noise_mask.mean():.1%})")
# Step 2: Simple noise estimation
print("\n2. Running simple noise estimation...")
# Get predicted probabilities
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000, random_state=42)
pred_probs = cross_val_predict(model, X_scaled, y_noisy, cv=5, method='predict_proba')
# Find label issues (simple method)
self_confidence = pred_probs[np.arange(len(y_noisy)), y_noisy]
label_issues = self_confidence < 0.5
n_label_issues = int(label_issues.sum())
label_error_rate = float(label_issues.mean())
print(f"  • Label issues found: {n_label_issues} ({label_error_rate:.1%})")
print(f"  • Average confidence: {float(self_confidence.mean()):.3f}")
# Step 3: Rank label errors
print("\n3. Ranking label errors...")
error_indices = np.where(label_issues)[0]
ranked_errors = []
for idx in error_indices:
    current_label = int(y_noisy[idx])
    pred_prob = pred_probs[idx]
    # Get suggested label
    top_classes = np.argsort(pred_prob)[::-1]
    suggested_label = int(top_classes[0] if top_classes[0] != current_label else top_classes[1])
    suggested_confidence = float(pred_prob[suggested_label])
    self_conf = float(pred_prob[current_label])
    ranked_errors.append({
        'sample_index': int(idx),
        'current_label': current_label,
        'suggested_label': suggested_label,
        'suggested_confidence': suggested_confidence,
        'self_confidence': self_conf,
        'margin': float(suggested_confidence - self_conf)
    })
# Sort by margin
ranked_errors.sort(key=lambda x: x['margin'], reverse=True)
for i, error in enumerate(ranked_errors):
    error['rank'] = i + 1
# Step 4: Save results
print("\n4. Saving results...")
# Save ranked errors
if ranked_errors:
    errors_df = pd.DataFrame(ranked_errors)
    errors_file = os.path.join(output_dir, 'ranked_label_errors.csv')
    errors_df.to_csv(errors_file, index=False)
    print(f"  ✓ Saved {len(ranked_errors)} ranked errors")
# Save quality scores
quality_df = pd.DataFrame({
    'sample_index': range(len(y_noisy)),
    'label': y_noisy,
    'quality_score': [float(score) for score in self_confidence]
})
quality_file = os.path.join(output_dir, 'label_quality_scores.csv')
quality_df.to_csv(quality_file, index=False)
print(f"  ✓ Saved quality scores")
# Save summary
summary = {
    'dataset': 'synthetic_with_noise',
    'n_samples': int(X.shape[0]),
    'n_features': int(X.shape[1]),
    'n_classes': int(len(np.unique(y_clean))),
    'true_noise': int(noise_mask.sum()),
    'true_noise_rate': float(noise_mask.mean()),
    'detected_label_errors': n_label_issues,
    'detected_error_rate': label_error_rate,
    'avg_confidence': float(self_confidence.mean()),
    'analysis_timestamp': datetime.now().isoformat()
}
summary_file = os.path.join(output_dir, 'summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Saved summary")
# Step 5: Final report
print("\n" + "=" * 80)
print("PHASE 3 COMPLETE (Ultra-Simple Version)")
print("=" * 80)
print(f"\n📊 RESULTS:")
print(f"  • Dataset: Synthetic with noise")
print(f"  • Samples: {X.shape[0]:,}")
print(f"  • True noise: {noise_mask.sum():,} ({noise_mask.mean():.1%})")
print(f"  • Detected errors: {n_label_errors:,} ({label_error_rate:.1%})")
print(f"  • Average confidence: {self_confidence.mean():.3f}")
print(f"\n📁 OUTPUT:")
print(f"  • Directory: {output_dir}/")
print(f"  • Ranked errors: {output_dir}/ranked_label_errors.csv")
print(f"  • Quality scores: {output_dir}/label_quality_scores.csv")
print(f"  • Summary: {output_dir}/summary.json")
print(f"\n🎯 NEXT STEPS:")
print("  1. Review ranked_label_errors.csv for potential label issues")
print("  2. Use quality scores to prioritize data cleaning")
print("  3. Proceed to Phase 4: Dataset Health & Class Profiling")
print("=" * 80)
