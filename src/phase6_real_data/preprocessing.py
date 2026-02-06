from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class TabularPreprocessor:
    def preprocess(self, df, label_col="label"):
        X = df.drop(columns=[label_col])
        y = df[label_col]

        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = X.select_dtypes(include=["object"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        return X_processed, y.values, preprocessor
