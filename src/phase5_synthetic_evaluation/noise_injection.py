import numpy as np
import pandas as pd


class SyntheticNoiseInjector:
    """
    Injects controlled synthetic label noise.
    """

    def inject_random_noise(
        self,
        df: pd.DataFrame,
        label_col: str,
        noise_rate: float = 0.2,
        seed: int = 42,
    ) -> pd.DataFrame:
        np.random.seed(seed)
        df = df.copy()

        n = len(df)
        noisy_count = int(noise_rate * n)
        noisy_idx = np.random.choice(df.index, noisy_count, replace=False)

        df["is_noisy_true"] = 0
        df.loc[noisy_idx, "is_noisy_true"] = 1

        labels = df[label_col].unique()

        for idx in noisy_idx:
            current = df.at[idx, label_col]
            df.at[idx, label_col] = np.random.choice(labels[labels != current])

        return df
