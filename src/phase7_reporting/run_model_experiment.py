import pandas as pd

from model_training import ModelTrainer
from accuracy_plot import plot_accuracy


# ===== Load Data =====

# Raw dataset → semicolon separator
raw_df = pd.read_csv(
"C:/Users/User/LQNE_Project/data/raw/bank_marketing.csv",
sep=";"
)

# Cleaned dataset → comma separator
clean_df = pd.read_csv(
"C:/Users/User/LQNE_Project/data/processed/bank_cleaned.csv"
)


# ===== Train Models =====

trainer = ModelTrainer()

print("\nTraining BEFORE cleaning")
before_accuracy = trainer.train_model(raw_df)

print("\nTraining AFTER cleaning")
after_accuracy = trainer.train_model(clean_df)


# ===== Plot Results =====

plot_accuracy(before_accuracy, after_accuracy)

print("\nExperiment Completed")