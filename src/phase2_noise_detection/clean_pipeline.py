import numpy as np
from sklearn.ensemble import RandomForestClassifier


def clean_labels(X, y):
    """
    LQNE Label Cleaning Pipeline

    Steps:
    1 Detect noisy labels using model confidence
    2 Correct labels
    3 Return cleaned dataset
    """

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    confidence = np.max(probabilities, axis=1)

    predicted_labels = model.predict(X)

    # Threshold for noisy labels

    threshold = 0.7

    y_clean = y.copy()

    noisy_count = 0

    for i in range(len(y)):

        if confidence[i] < threshold:

            y_clean[i] = predicted_labels[i]

            noisy_count += 1

    print("Labels corrected:", noisy_count)

    return X, y_clean