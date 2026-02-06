import numpy as np
import pandas as pd


class SampleTrustScorer:
    """
    Computes trust score (0–1) per sample
    """

    def __init__(self):
        self.w_confidence = 0.35
        self.w_label_error = 0.30
        self.w_disagreement = 0.20
        self.w_class_noise = 0.15

    def _normalize(self, x):
        x = np.asarray(x)
        if x.max() == x.min():
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    def compute_trust_score(
        self,
        df,
        confidence_col,
        label_error_col,
        disagreement_col,
        class_noise_col,
    ):
        for col in [
            confidence_col,
            label_error_col,
            disagreement_col,
            class_noise_col,
        ]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        conf = self._normalize(df[confidence_col])
        label_err = self._normalize(df[label_error_col])
        disagree = self._normalize(df[disagreement_col])
        class_noise = self._normalize(df[class_noise_col])

        trust = (
            self.w_confidence * conf
            + self.w_label_error * (1 - label_err)
            + self.w_disagreement * (1 - disagree)
            + self.w_class_noise * (1 - class_noise)
        )

        df = df.copy()
        df["trust_score"] = np.clip(trust, 0, 1)

        df["trust_bucket"] = pd.cut(
            df["trust_score"],
            bins=[0.0, 0.4, 0.7, 1.0],
            labels=["low", "medium", "high"],
        )

        return df
