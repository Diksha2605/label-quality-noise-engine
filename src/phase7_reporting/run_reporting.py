# src/phase7_reporting/run_reporting.py

import os

from src.phase6_real_data.datasets import load_bank_marketing_dataset
from src.phase7_reporting.exporters import (
    export_sample_trust,
    export_class_noise,
    export_dataset_health
)


def run_reporting(
    dataset_name: str,
    output_dir: str,
    trust_threshold: float = 0.3,
):
    """
    Phase 7 Reporting entry point
    """

    print("\n📊 Running Phase 7: Reporting Module")
    print(f"Dataset: {dataset_name}")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Dataset selection (future-proof) ----
    if dataset_name == "bank_marketing":
        X, y = load_bank_marketing_dataset()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # ---- Placeholder trust scores (already computed in earlier phases) ----
    trust_scores = [1.0 - trust_threshold] * len(y)

    # ---- Export reports ----
    export_sample_trust(trust_scores, y, output_dir)
    export_class_noise(y, output_dir)
    export_dataset_health(trust_scores, output_dir)

    print(f"✅ Reports generated in {output_dir}")
