import argparse
import yaml
import os
import sys

# Ensure src is discoverable
sys.path.append(os.path.abspath("src"))

from phase7_reporting.run_reporting import run_reporting


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LQNE – Label Quality & Noise Explorer")
    parser.add_argument("--dataset", required=True, help="Dataset name (adult | bank_marketing)")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")

    args = parser.parse_args()
    config = load_config(args.config)

    dataset_name = args.dataset
    output_dir = os.path.join(
        config["output"]["base_dir"], dataset_name
    )

    print(f"\n🚀 Running LQNE on dataset: {dataset_name}")
    print(f"📁 Output directory: {output_dir}\n")

    run_reporting(
        dataset_name=dataset_name,
        output_dir=output_dir,
        trust_threshold=config["trust"]["low_threshold"],
    )

    print("\n✅ LQNE run complete.")


if __name__ == "__main__":
    main()
