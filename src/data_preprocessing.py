import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    df["Car_Age"] = 2024 - df["Year"]
    df.drop(["Car_Name", "Year"], axis=1, inplace=True)

    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    categorical_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    numeric_cols = ["Present_Price", "Driven_kms", "Owner", "Car_Age"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    return X, y, preprocessor


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)