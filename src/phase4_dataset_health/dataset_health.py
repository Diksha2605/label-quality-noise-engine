class DatasetHealthScorer:
    """
    Computes a single dataset health score (0–100)
    """

    def compute_dataset_health(
        self,
        df,
        trust_col="trust_score",
        label_error_col="label_error_prob",
    ):
        avg_trust = df[trust_col].mean()
        noise_ratio = df[label_error_col].mean()
        low_trust_ratio = (df[trust_col] < 0.4).mean()

        health_score = (
            0.5 * avg_trust
            + 0.3 * (1 - noise_ratio)
            + 0.2 * (1 - low_trust_ratio)
        )

        return {
            "dataset_health_score": round(health_score * 100, 2),
            "avg_trust": round(avg_trust, 3),
            "noise_ratio": round(noise_ratio, 3),
            "low_trust_ratio": round(low_trust_ratio, 3),
        }
