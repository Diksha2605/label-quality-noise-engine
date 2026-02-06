import os
import json


class AutoCleaningAdvisor:
    def __init__(
        self,
        output_path="reports/cleaning_recommendations.json",
        trust_threshold=0.25,
        drop_fraction=0.01,
    ):
        self.output_path = output_path
        self.trust_threshold = trust_threshold
        self.drop_fraction = drop_fraction
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def generate(self, df):
        """
        df must contain:
        - trust_score
        - label_error_prob
        - class_noise_rate
        - relabel_priority (optional)
        """

        total_samples = len(df)

        low_trust_df = df[df["trust_score"] < self.trust_threshold]
        low_trust_frac = len(low_trust_df) / total_samples

        # Drop candidates (worst trust only)
        drop_n = max(1, int(total_samples * self.drop_fraction))
        drop_candidates = (
            df.sort_values("trust_score")
            .head(drop_n)
            .index
            .tolist()
        )

        recommendations = {
            "summary": {
                "total_samples": total_samples,
                "low_trust_fraction": round(low_trust_frac, 3),
                "suggested_drop_fraction": self.drop_fraction,
                "suggested_drop_count": drop_n,
            },
            "recommendations": [
                {
                    "action": "relabel",
                    "description": "Relabel low-trust samples first",
                    "affected_samples": len(low_trust_df),
                    "rule": f"trust_score < {self.trust_threshold}",
                },
                {
                    "action": "drop",
                    "description": "Consider dropping worst samples (after review)",
                    "affected_samples": drop_n,
                    "rule": f"lowest {self.drop_fraction*100:.1f}% trust scores",
                },
                {
                    "action": "reweight",
                    "description": "Down-weight samples inversely proportional to trust",
                    "rule": "sample_weight = trust_score",
                },
            ],
            "drop_candidate_indices": drop_candidates,
        }

        with open(self.output_path, "w") as f:
            json.dump(recommendations, f, indent=2)

        return recommendations
