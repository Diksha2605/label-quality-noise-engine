import sys
import os
sys.path.append(os.path.abspath("."))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.phase4_dataset_health.sample_trust import SampleTrustScorer


# --------------------------------------------------
# 1. Load German Credit Dataset
# --------------------------------------------------
def load_german_credit():
    url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/statlog/"
        "german/german.data"
    )

    columns = [
        "status", "duration", "credit_history", "purpose",
        "credit_amount", "savings", "employment",
        "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property",
        "age", "other_installment_plans", "housing",
        "existing_credits", "job", "num_dependents",
        "telephone", "foreign_worker", "label"
    ]

    df = pd.read_csv(url, sep=" ", names=columns)

    # Convert label: 1=good, 2=bad → binary
    df["label"] = (df["label"] == 2).astype(int)

    return df


# --------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------
def preprocess(df, label_col="label"):
    X = df.drop(columns=[label_col])
    y = df[label_col].values

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y


# --------------------------------------------------
# 3. Main Experiment
# --------------------------------------------------
def run_german_experiment():
    print("\n=== Phase 6: German Credit Dataset ===")

    df = load_german_credit()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 4. Baseline model
    # --------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    baseline_acc = accuracy_score(
        y_test, model.predict(X_test)
    )

    # --------------------------------------------------
    # 5. Trust signals
    # --------------------------------------------------
    probs = model.predict_proba(X_train).max(axis=1)
    disagreement = 1 - probs

    df_train = pd.DataFrame({
        "label": y_train,
        "pred_proba_max": probs,
        "label_error_prob": disagreement,
        "cv_disagreement": disagreement,
    })

    df_train["class_noise_rate"] = (
        df_train.groupby("label")["label_error_prob"].transform("mean")
    )

    # --------------------------------------------------
    # 6. Trust scoring
    # --------------------------------------------------
    scorer = SampleTrustScorer()
    df_train = scorer.compute_trust_score(
        df=df_train,
        confidence_col="pred_proba_max",
        label_error_col="label_error_prob",
        disagreement_col="cv_disagreement",
        class_noise_col="class_noise_rate",
    )

    # --------------------------------------------------
    # 7. Retrain after cleaning
    # --------------------------------------------------
    clean_mask = (df_train["trust_score"] >= 0.4).values

    clean_model = LogisticRegression(max_iter=1000)
    clean_model.fit(
        X_train[clean_mask],
        y_train[clean_mask],
    )

    clean_acc = accuracy_score(
        y_test, clean_model.predict(X_test)
    )

    # --------------------------------------------------
    # 8. Results
    # --------------------------------------------------
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    print(f"After cleaning accuracy: {clean_acc:.3f}")
    print(f"Delta: {clean_acc - baseline_acc:.3f}")

    print("\nTop suspicious samples:")
    print(
        df_train.sort_values("trust_score")
        .head(10)[["trust_score", "label"]]
    )

    # --------------------------------------------------
    # 9. Visualization
    # --------------------------------------------------
    plt.figure(figsize=(7, 4))
    sns.histplot(df_train["trust_score"], bins=25, kde=True)
    plt.title("Trust Score Distribution (German Credit)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_train["label"], y=df_train["trust_score"])
    plt.title("Trust Score by Risk Class")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 10. Run
# --------------------------------------------------
if __name__ == "__main__":
    run_german_experiment()
