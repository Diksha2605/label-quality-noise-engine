from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class ModelImpactEvaluator:
    """
    Measures model performance before & after cleaning
    """

    def evaluate_model_impact(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        trust_scores,
        threshold=0.4,
    ):
        # Baseline model
        base_model = LogisticRegression(max_iter=1000)
        base_model.fit(X_train, y_train)
        base_acc = accuracy_score(y_test, base_model.predict(X_test))

        # Cleaned data
        mask = trust_scores >= threshold
        clean_model = LogisticRegression(max_iter=1000)
        clean_model.fit(X_train[mask], y_train[mask])
        clean_acc = accuracy_score(y_test, clean_model.predict(X_test))

        return {
            "baseline_accuracy": round(base_acc, 3),
            "after_cleaning_accuracy": round(clean_acc, 3),
            "delta": round(clean_acc - base_acc, 3),
        }
