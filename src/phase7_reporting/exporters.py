# src/phase7_reporting/exporters.py

import os
import json
import pandas as pd
import numpy as np


def export_sample_trust(trust_scores, labels, output_dir):
    """
    Export per-sample trust scores
    """
    df = pd.DataFrame({
        "sample_id": range(len(trust_scores)),
        "trust_score": trust_scores,
        "label": labels
    })

    csv_path = os.path.join(output_dir, "sample_trust_report.csv")
    json_path = os.path.join(output_dir, "sample_trust_report.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"✔ Sample trust report saved → {csv_path}")


def export_class_noise(labels, output_dir):
    """
    Estimate class-level noise (simple heuristic)
    """
    df = pd.DataFrame({"label": labels})
    noise_report = df["label"].value_counts(normalize=True).reset_index()
    noise_report.columns = ["label", "class_fraction"]

    csv_path = os.path.join(output_dir, "class_noise_report.csv")
    noise_report.to_csv(csv_path, index=False)

    print(f"✔ Class noise report saved → {csv_path}")


def export_dataset_health(trust_scores, output_dir):
    """
    Overall dataset health score
    """
    health_score = float(np.mean(trust_scores))

    report = {
        "dataset_health_score": round(health_score, 4),
        "status": "healthy" if health_score > 0.7 else "needs_review"
    }

    json_path = os.path.join(output_dir, "dataset_health.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✔ Dataset health report saved → {json_path}")
