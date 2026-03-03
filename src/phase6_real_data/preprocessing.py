import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class RealDatasetPreprocessor:


    # Load dataset
    def load_bank_marketing(self, path):

        # IMPORTANT: Bank dataset uses ;
        df = pd.read_csv(path, sep=';')

        print("\nDataset Loaded Successfully")
        print("Shape:", df.shape)

        print("\nColumns Found:")
        print(df.columns)

        return df



    # Preprocess dataset
    def preprocess(self, df):

        df = df.copy()

        # Target column = y
        target_column = "y"

        print("\nTarget Column Detected:", target_column)


        # Convert yes/no → 1/0
        df["label"] = df[target_column].map({
            "yes": 1,
            "no": 0
        })


        df = df.drop(columns=[target_column])


        # Encode categorical columns
        for col in df.select_dtypes(include="object").columns:

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(df[col])


        print("\nPreprocessing Completed")

        return df



    # Add noise
    def add_noise_labels(self, df, noise_rate=0.15):

        df = df.copy()

        n = int(len(df) * noise_rate)

        noisy_indices = np.random.choice(
            df.index,
            n,
            replace=False
        )

        df.loc[noisy_indices, "label"] = \
            1 - df.loc[noisy_indices, "label"]

        print("\nNoise Injection Completed")

        return df



    # Trust scores
    def generate_trust_scores(self, df):

        df = df.copy()

        df["trust_score"] = np.random.uniform(
            0.4,
            1.0,
            len(df)
        )

        df["label_error_prob"] = \
            1 - df["trust_score"]

        print("\nTrust Scores Generated")

        return df