import joblib
import pandas as pd


def load_model(path="models/model.pkl"):
    return joblib.load(path)


def predict(model, input_dict: dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]