import numpy as np


class NoiseDetectionEvaluator:
    """
    Evaluates noise detection quality using ranking metrics.
    """

    def evaluate(
        self,
        df,
        trust_col="trust_score",
        true_noise_col="is_noisy_true",
        top_fraction=0.2,
    ):
        y_true = df[true_noise_col].values
        suspicion = 1 - df[trust_col].values

        # Rank correlation (sanity signal)
        rank_corr = np.corrcoef(y_true, suspicion)[0, 1]

        # Precision@TopK
        k = int(top_fraction * len(df))
        top_k_idx = np.argsort(suspicion)[-k:]
        precision_at_k = y_true[top_k_idx].mean()

        return {
            "rank_correlation": round(float(rank_corr), 3),
            "precision_at_top_20pct": round(float(precision_at_k), 3),
        }
