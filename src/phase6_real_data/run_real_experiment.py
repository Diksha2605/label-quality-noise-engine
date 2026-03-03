import sys
import os


# Fix imports
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


from preprocessing import RealDatasetPreprocessor
from phase4_dataset_health.dataset_health import DatasetHealthScorer



def run_real_experiment():

    print("\n==============================")
    print("LQNE REAL DATA EXPERIMENT")
    print("==============================\n")


    processor = RealDatasetPreprocessor()


    # YOUR DATASET PATH
    dataset_path = r"C:\Users\User\LQNE_Project\data\raw\bank_marketing.csv"


    # Load dataset
    df = processor.load_bank_marketing(dataset_path)


    # Preprocess
    df = processor.preprocess(df)


    # Add Noise
    df = processor.add_noise_labels(df)


    # Trust Scores
    df = processor.generate_trust_scores(df)


    # Dataset Health

    scorer = DatasetHealthScorer()

    health = scorer.compute_dataset_health(df)


    print("\n==============================")
    print("DATASET HEALTH REPORT")
    print("==============================")

    print("Dataset Health Score:", health["dataset_health_score"])
    print("Average Trust:", health["avg_trust"])
    print("Noise Ratio:", health["noise_ratio"])
    print("Low Trust Ratio:", health["low_trust_ratio"])



    # Save cleaned dataset

    output_path = r"C:\Users\User\LQNE_Project\data\processed\bank_cleaned.csv"

    df.to_csv(output_path, index=False)


    print("\nSaved Clean Dataset To:")
    print(output_path)



if __name__ == "__main__":

    run_real_experiment()