import mlflow
import mlflow.keras

from model_predictions import load_and_prepare_data, preprocess_data, build_model, train_model, evaluate_model
from config import logger, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, REGISTERED_MODEL_NAME

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

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)  

    with mlflow.start_run() as run:  
        run_id = run.info.run_id
        logger.info(f"run id: {run_id}")
        logger.info(f"artifact_uri: {mlflow.get_artifact_uri()}")
        logger.info(f"registry_uri: {mlflow.get_registry_uri()}")

        mlflow.log_param("epochs", 2)
        mlflow.log_param("batch_size", 32)
        mlflow.log_metrics({"train_accuracy": history.history["accuracy"][-1], "val_accuracy": history.history["val_accuracy"][-1]})

        logger.info(f"Logging and register the model {REGISTERED_MODEL_NAME}...")
        mlflow.keras.log_model(model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME) 

        # Create a text file to log as an artifact
        # with open("metrics.txt", "w") as f:
        #     f.write(f"Loss: {loss}\n")
        #     f.write(f"Accuracy: {accuracy}\n")
        # mlflow.log_artifact("metrics.txt")

if __name__ == "__main__":
    run_mlflow_pipeline("data/winequality.csv")