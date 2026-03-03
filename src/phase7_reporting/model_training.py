import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class ModelTrainer:

    def train_model(self, df):

        df = df.copy()

        # ===== Detect Target Column =====

        if "y" in df.columns:
            target_col = "y"
        elif "label" in df.columns:
            target_col = "label"
        elif "target" in df.columns:
            target_col = "target"
        else:
            raise Exception(
                f"No target column found. Columns available: {list(df.columns)}"
            )

        # ===== Separate X and Y =====

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ===== Encode Target =====

        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

        # ===== Encode Categorical Columns =====

        for col in X.columns:
            if X[col].dtype == "object":
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))

        # ===== Train Test Split =====

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ===== Train Model =====

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

        # ===== Prediction =====

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)

        return accuracy