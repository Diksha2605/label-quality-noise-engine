import sys
import os
sys.path.append(os.path.abspath("."))

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.phase4_dataset_health.sample_trust import SampleTrustScorer
from src.phase5_synthetic_evaluation.noise_injection import SyntheticNoiseInjector
from src.phase5_synthetic_evaluation.noise_evaluation import NoiseDetectionEvaluator
from src.phase5_synthetic_evaluation.model_evaluation import ModelPerformanceEvaluator


# --------------------------------------------------
# 1. Generate clean synthetic dataset
# --------------------------------------------------
X, y = make_classification(
    n_samples=1200,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=2,
    random_state=42,
)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["label"] = y


# --------------------------------------------------
# 2. Inject synthetic noise
# --------------------------------------------------
injector = SyntheticNoiseInjector()
df = injector.inject_random_noise(
    df,
    label_col="label",
    noise_rate=0.25,
)


# --------------------------------------------------
# 3. Simulate Phase 2–3 outputs
# (in real pipeline these come from models + cleanlab)
# --------------------------------------------------
df["pred_proba_max"] = 0.9 - 0.5 * df["is_noisy_true"]
df["label_error_prob"] = df["is_noisy_true"]
df["cv_disagreement"] = df["is_noisy_true"]
df["class_noise_rate"] = df.groupby("label")["is_noisy_true"].transform("mean")


# --------------------------------------------------
# 4. Phase 4 trust scoring
# --------------------------------------------------
trust_scorer = SampleTrustScorer()
df = trust_scorer.compute_trust_score(
    df,
    confidence_col="pred_proba_max",
    label_error_col="label_error_prob",
    disagreement_col="cv_disagreement",
    class_noise_col="class_noise_rate",
)


# --------------------------------------------------
# 5. Noise detection evaluation
# --------------------------------------------------
detector = NoiseDetectionEvaluator()
noise_metrics = detector.evaluate(df)


# --------------------------------------------------
# 6. Model performance evaluation
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df[[c for c in df.columns if c.startswith("f")]],
    df["label"],
    test_size=0.3,
    random_state=42,
)

trust_train = df.loc[X_train.index, "trust_score"].values

model_eval = ModelPerformanceEvaluator()
model_metrics = model_eval.evaluate(
    X_train.values,
    y_train.values,
    X_test.values,
    y_test.values,
    trust_train,
)


# --------------------------------------------------
# 7. Results
# --------------------------------------------------
print("\n=== Phase 5: Synthetic Evaluation Results ===")

print("\nNoise Detection Metrics:")
for k, v in noise_metrics.items():
    print(f"{k}: {v}")

print("\nModel Performance Impact:")
for k, v in model_metrics.items():
    print(f"{k}: {v}")
