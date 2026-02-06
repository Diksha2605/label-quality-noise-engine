from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class ModelPerformanceEvaluator:
    """
    Measures model accuracy before and after trust-based cleaning.
    """

    def evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        train_trust_scores,
        trust_threshold=0.4,
    ):
        # Baseline
        base_model = LogisticRegression(max_iter=1000)
        base_model.fit(X_train, y_train)
        base_acc = accuracy_score(y_test, base_model.predict(X_test))

        # Cleaned
        clean_mask = train_trust_scores >= trust_threshold
        clean_model = LogisticRegression(max_iter=1000)
        clean_model.fit(X_train[clean_mask], y_train[clean_mask])
        clean_acc = accuracy_score(y_test, clean_model.predict(X_test))

        return {
            "baseline_accuracy": round(base_acc, 3),
            "after_cleaning_accuracy": round(clean_acc, 3),
            "accuracy_delta": round(clean_acc - base_acc, 3),
        }
