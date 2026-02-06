# src/phase6_real_data/datasets.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_bank_marketing_dataset():
    """
    Loads and preprocesses the Bank Marketing dataset
    """
    df = pd.read_csv("data/raw/bank_marketing.csv", sep=";")

    y = df["y"].map({"yes": 1, "no": 0})
    X = df.drop(columns=["y"])

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    return X.values, y.values
