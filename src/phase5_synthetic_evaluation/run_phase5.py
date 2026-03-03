import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

from src.phase5_synthetic_evaluation.noise_evaluation import evaluate_noise_pipeline
from src.phase2_noise_detection.clean_pipeline import clean_labels


print("\nRunning Synthetic Noise Experiments...\n")


# =============================
# Project Root Path
# =============================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


# =============================
# Dataset Path
# =============================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "bank_marketing.csv"
)


print("Dataset Path:")
print(DATA_PATH)


# =============================
# Load Dataset
# =============================

df = pd.read_csv(DATA_PATH, sep=";")


print("\nDataset Loaded Successfully")

print("Shape:", df.shape)


# =============================
# Encode Categorical Columns
# =============================

label_encoders = {}

for column in df.columns:

    if df[column].dtype == "object":

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column])

        label_encoders[column] = le


print("\nCategorical Encoding Done")


# =============================
# Features and Labels
# =============================

X = df.drop("y", axis=1).values
y = df["y"].values


print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)


# =============================
# Run Synthetic Experiments
# =============================

results = evaluate_noise_pipeline(

    X,
    y,
    clean_labels

)


# =============================
# Save Results
# =============================

REPORT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "synthetic_noise_results.csv"
)


results_df = pd.DataFrame(results)

results_df.to_csv(

    REPORT_PATH,
    index=False

)


print("\nResults Saved Successfully")

print("Location:", REPORT_PATH)