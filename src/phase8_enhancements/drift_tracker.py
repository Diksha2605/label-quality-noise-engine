import os
from datetime import datetime
import pandas as pd


class DatasetDriftTracker:
    def __init__(self, history_path="reports/health_history.csv"):
        self.history_path = history_path
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

    def log_snapshot(self, health_json):
        """
        health_json example:
        {
            "dataset_health_score": 0.66,
            "health_grade": "Good",
            "total_samples": 22792,
            "low_trust_fraction": 0.14
        }
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": health_json["dataset_health_score"],
            "low_trust_fraction": health_json["low_trust_fraction"],
            "total_samples": health_json["total_samples"],
        }

        df_new = pd.DataFrame([record])

        if os.path.exists(self.history_path):
            df_old = pd.read_csv(self.history_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv(self.history_path, index=False)
