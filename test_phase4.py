import pandas as pd
from src.phase4_dataset_health.sample_trust import SampleTrustScorer
from src.phase4_dataset_health.class_trust import ClassTrustProfiler
from src.phase4_dataset_health.dataset_health import DatasetHealthScorer

# Dummy dataset
df = pd.DataFrame({
    "label": ["A", "A", "B", "B", "C"],
    "pred_proba_max": [0.9, 0.6, 0.4, 0.8, 0.3],
    "label_error_prob": [0.1, 0.3, 0.6, 0.2, 0.8],
    "cv_disagreement": [0.1, 0.4, 0.7, 0.2, 0.9],
    "class_noise_rate": [0.2, 0.2, 0.5, 0.5, 0.7],
})

# Sample trust
sample_scorer = SampleTrustScorer()
df = sample_scorer.compute_trust_score(
    df,
    "pred_proba_max",
    "label_error_prob",
    "cv_disagreement",
    "class_noise_rate",
)

# Class trust
class_profiler = ClassTrustProfiler()
class_report = class_profiler.compute_class_trust(
    df, label_col="label"
)

# Dataset health
dataset_scorer = DatasetHealthScorer()
dataset_health = dataset_scorer.compute_dataset_health(df)

print("\nSample Trust:")
print(df)

print("\nClass Trust:")
print(class_report)

print("\nDataset Health:")
print(dataset_health)
