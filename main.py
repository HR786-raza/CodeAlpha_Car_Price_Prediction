from src.data_preprocessing import load_data, preprocess_data, split_data
from src.train import train_models, evaluate_models, save_best_model


def main():
    df = load_data("data/car_data.csv")

    X, y, pre = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_models(X_train, y_train, pre)
    results = evaluate_models(models, X_test, y_test)

    print(results)

    best = save_best_model(models, results)
    print("Best model:", best)


if __name__ == "__main__":
    main()