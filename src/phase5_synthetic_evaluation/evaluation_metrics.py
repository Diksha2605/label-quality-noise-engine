import numpy as np


class NoiseDetectionEvaluator:
    """
    Evaluates how well trust scores detect true noise
    """

    def evaluate_detection(self, df, trust_col="trust_score", true_noise_col="is_noisy_true"):
        y_true = df[true_noise_col].values
        suspicion_score = 1 - df[trust_col].values

        # Rank correlation
        rank_corr = np.corrcoef(y_true, suspicion_score)[0, 1]

        # Precision@K (top suspicious samples)
        k = int(0.2 * len(df))
        top_k = np.argsort(suspicion_score)[-k:]
        precision_at_k = y_true[top_k].mean()

        return {
            "rank_correlation": round(rank_corr, 3),
            "precision_at_20pct": round(precision_at_k, 3),
        }
