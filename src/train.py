import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def train_models(X_train, y_train, preprocessor):
    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "DecisionTree": DecisionTreeRegressor()
    }

    trained_models = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

    return trained_models


def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {"R2": r2, "RMSE": rmse}

    return results


def save_best_model(models, results, path="models/model.pkl"):
    best = max(results, key=lambda x: results[x]["R2"])
    joblib.dump(models[best], path)
    return best