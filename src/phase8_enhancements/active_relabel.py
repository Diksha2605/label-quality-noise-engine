import os
import pandas as pd


class ActiveRelabelQueue:
    def __init__(
        self,
        output_path="reports/relabel_queue.csv",
        min_priority=0.05,
    ):
        self.output_path = output_path
        self.min_priority = min_priority
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def generate(self, df, top_k=500):
        """
        df must contain:
        - trust_score
        - label
        - label_error_prob
        - class_noise_rate
        """

        df = df.copy()

        df["relabel_priority"] = (
            df["label_error_prob"]
            * df["class_noise_rate"]
            * (1 - df["trust_score"])
        )

        df = df[df["relabel_priority"] >= self.min_priority]

        df = df.sort_values(
            by="relabel_priority",
            ascending=False,
        )

        output_cols = [
            "relabel_priority",
            "trust_score",
            "label",
            "label_error_prob",
            "class_noise_rate",
        ]

        df[output_cols].head(top_k).to_csv(
            self.output_path,
            index=False,
        )

        return df[output_cols].head(top_k)
