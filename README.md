# Label Quality Noise Engine (LQNE)

## Overview

Label Quality Noise Engine (LQNE) is an end-to-end machine learning pipeline designed to detect and correct noisy labels in supervised datasets.

Real-world datasets often contain incorrect labels due to human error, automation mistakes, or data collection issues. These noisy labels reduce model accuracy and reliability.

This project builds a complete pipeline that:

- Detects suspicious labels
- Estimates label reliability
- Corrects incorrect labels
- Retrains models on cleaned data
- Evaluates performance improvements

The system demonstrates how label cleaning improves dataset quality and machine learning performance.


----------------------------------------------------

## Key Features

- Automated label noise detection
- Label correction pipeline
- Synthetic noise experiments
- Real dataset evaluation
- Dataset health analysis
- End-to-end ML workflow
- Reproducible experiments


----------------------------------------------------

## Project Structure

LQNE_Project/

data/

    raw/
        bank_marketing.csv

    processed/

    synthetic/


src/

    phase0_data_loading/

    phase1_eda/

    phase2_noise_detection/

    phase3_confident_learning/

    phase4_dataset_health/

    phase5_synthetic_evaluation/

    phase6_real_data/

    phase7_reporting/


reports/


----------------------------------------------------

## Problem Statement

Machine learning models assume training labels are correct.

In real-world datasets:

- Human labeling mistakes occur
- Automated labeling systems fail
- Old datasets contain errors

Even small amounts of label noise can significantly reduce model accuracy.

This project builds a system that automatically improves dataset quality before training.


----------------------------------------------------

## Methodology

Step 1 — Data Loading

Dataset is loaded and prepared for training.


Step 2 — Noise Detection

Suspicious labels are detected using:

- Model confidence scores
- Prediction disagreement
- Probability thresholds


Step 3 — Label Cleaning

Low-confidence labels are corrected using model predictions.


Step 4 — Model Training

Models are trained:

- Before cleaning
- After cleaning


Step 5 — Synthetic Noise Experiments

Artificial label noise is injected into datasets:

- 10% noise
- 20% noise
- 30% noise

The cleaning pipeline is evaluated against known noise.


Step 6 — Evaluation

Metrics used:

- Accuracy
- Precision
- Recall


----------------------------------------------------

## Dataset

Bank Marketing Dataset

- 41,188 samples
- 20 features
- Binary classification problem

Target variable:

y = yes/no

Dataset location:

data/raw/bank_marketing.csv


----------------------------------------------------

## Results

Training BEFORE cleaning

Accuracy ≈ 0.91


Training AFTER cleaning

Accuracy ≈ 0.95


Synthetic Noise Experiments

10% Noise → Accuracy Improved

20% Noise → Accuracy Improved

30% Noise → Accuracy Improved


----------------------------------------------------

## Installation

Clone repository:

git clone https://github.com/YOUR_USERNAME/LQNE_Project.git


Install dependencies:

pip install -r requirements.txt


----------------------------------------------------

## How to Run

Run Synthetic Noise Experiments:

python -m src.phase5_synthetic_evaluation.run_phase5


Run Final Pipeline:

python -m src.phase7_reporting.run_model


----------------------------------------------------

## Example Output

Training BEFORE cleaning Accuracy: 0.91

Training AFTER cleaning Accuracy: 0.95

Noise Level: 10%
Accuracy after cleaning: Improved

Noise Level: 20%
Accuracy after cleaning: Improved

Noise Level: 30%
Accuracy after cleaning: Improved


----------------------------------------------------

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib


----------------------------------------------------

## Research Contribution

This project demonstrates a practical approach to:

- Dataset Quality Engineering
- Label Noise Detection
- Robust Machine Learning Pipelines

The system simulates real-world data science problems.


----------------------------------------------------

## Author

Diksha Singh

Machine Learning Student


----------------------------------------------------

## License

MIT License
