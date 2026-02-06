import pandas as pd


class ClassTrustProfiler:
    """
    Aggregates trust metrics at class level
    """

    def compute_class_trust(
        self,
        df,
        label_col,
        trust_col="trust_score",
        label_error_col="label_error_prob",
    ):
        if label_col not in df.columns:
            raise ValueError("Label column missing")

        class_stats = (
            df.groupby(label_col)
            .agg(
                avg_trust=(trust_col, "mean"),
                noise_rate=(label_error_col, "mean"),
                sample_count=(label_col, "count"),
                low_trust_pct=(
                    trust_col,
                    lambda x: (x < 0.4).mean(),
                ),
            )
            .reset_index()
        )

        class_stats["class_health"] = (
            0.6 * class_stats["avg_trust"]
            + 0.4 * (1 - class_stats["noise_rate"])
        )

        class_stats = class_stats.sort_values(
            "class_health", ascending=True
        )

        return class_stats
