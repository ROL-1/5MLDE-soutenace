import mlflow
import mlflow.keras

from model_predictions import load_and_prepare_data, preprocess_data, build_model, train_model, evaluate_model


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Wine Quality Prediction")


def run_mlflow_pipeline(dataset_path: str):
    # .fn() pour appeler la fonction sous-jacente de la tâche Prefect
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data.fn(dataset_path)
    X_train_processed, X_val_processed, X_test_processed = preprocess_data.fn(X_train, X_val, X_test)

    input_dim = X_train_processed.shape[1]
    output_dim = y_train.shape[1]

    model = build_model.fn(input_dim, output_dim)
    history = train_model.fn(model, X_train_processed, y_train, X_val_processed, y_val, epochs=2, batch_size=32)
    loss, accuracy = evaluate_model.fn(model, X_test_processed, y_test)

    with mlflow.start_run():     

        mlflow.log_param("epochs", 2)
        mlflow.log_param("batch_size", 32)
        mlflow.log_metrics({"train_accuracy": history.history["accuracy"][-1], "val_accuracy": history.history["val_accuracy"][-1]})

        mlflow.keras.log_model(model, "models")

if __name__ == "__main__":
    run_mlflow_pipeline("data/winequality.csv")