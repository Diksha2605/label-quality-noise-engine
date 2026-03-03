from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.phase5_synthetic_evaluation.noise_injection import inject_label_noise


def evaluate_noise_pipeline(X, y, clean_function):

    noise_levels = [0.1, 0.2, 0.3]

    results = []

    for noise in noise_levels:

        print("\n====================")
        print("Noise Level:", noise)

        # Inject noise
        X_noise, y_noise, noisy_idx = inject_label_noise(
            X,
            y,
            noise
        )

        # Train BEFORE cleaning

        X_train, X_test, y_train, y_test = train_test_split(
            X_noise,
            y_noise,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred_before = model.predict(X_test)

        acc_before = accuracy_score(
            y_test,
            pred_before
        )

        print("Accuracy Before Cleaning:", acc_before)

        # Apply LQNE cleaning

        X_clean, y_clean = clean_function(
            X_noise,
            y_noise
        )

        # Train AFTER cleaning

        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X_clean,
            y_clean,
            test_size=0.2,
            random_state=42
        )

        model2 = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model2.fit(X_train2, y_train2)

        pred_after = model2.predict(X_test2)

        acc_after = accuracy_score(
            y_test2,
            pred_after
        )

        print("Accuracy After Cleaning:", acc_after)

        results.append({

            "Noise_Level": noise,
            "Accuracy_Before": acc_before,
            "Accuracy_After": acc_after

        })

    return results