import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from typing import List, Dict, Union, Tuple
from tensorflow.keras.callbacks import History
import mlflow
from prefect import task, flow

# Configuration de MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Wine Quality Prediction")

@task(name='load_data_3', tags=['model-prediction'], retries=3, retry_delay_seconds=10)
def load_and_prepare_data(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and prepares data for training and evaluation.

    Args:
    dataset_path (str): Path to the dataset file.

    Returns:
    Tuple containing split training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    df = pd.read_csv(dataset_path)
    df["quality"] = df["quality"].apply(lambda x: 0 if x < 6 else (1 if x == 6 else 2))
    y = to_categorical(df["quality"])
    X = df.drop("quality", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_val, X_test, y_train, y_val, y_test

@task(name='data_preprocessing', tags=['model-prediction'], retries=2, retry_delay_seconds=5)
def preprocess_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the data by applying normalization and encoding.

    Args:
    X_train, X_val, X_test (DataFrame): Datasets to preprocess.

    Returns:
    Tuple of preprocessed datasets (X_train_processed, X_val_processed, X_test_processed).
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, numeric_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_val_processed, X_test_processed

@task(name='model_declaration', tags=['model-prediction'])
def build_model(input_dim: int, output_dim: int) -> Sequential:
    """
    Builds a neural network model.

    Args:
    input_dim (int): Input dimension.
    output_dim (int): Output dimension.

    Returns:
    Sequential: Neural network model.
    """
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@task(name='model_training', tags=['model-prediction'], retries=1, retry_delay_seconds=30)
def train_model(model: Sequential, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, epochs: int = 2, batch_size: int = 32) -> History:
    """
    Trains the neural network model.

    Args:
    model (Sequential): The model to train.
    X_train, y_train, X_val, y_val (ndarray): Training and validation data.
    epochs (int): Number of epochs for training.
    batch_size (int): Batch size for training.

    Returns:
    History: Training history.
    """
    with mlflow.start_run():
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
        
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_metrics({"train_accuracy": history.history["accuracy"][-1], "val_accuracy": history.history["val_accuracy"][-1]})
        
        return history

@task(name='model_evaluation', tags=['model-prediction'], retries=1, retry_delay_seconds=5)
def evaluate_model(model: Sequential, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[float, float]:
    """
    Evaluates the model on the test set.

    Args:
    model (Sequential): The model to evaluate.
    X_test, y_test (ndarray): Test data.

    Returns:
    Tuple: Test loss and accuracy.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    with mlflow.start_run():
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)
        
        mlflow.keras.log_model(model, "model")
        
    return loss, accuracy

@flow(name="Wine Quality Prediction Flow")
def wine_quality_prediction_flow(dataset_path: str):
    """
    Orchestrates a workflow for loading, preprocessing, training, and evaluating a wine quality prediction model.

    Args:
    dataset_path (str): Path to the wine quality dataset file.

    The workflow executes several tasks in sequence:
    1. Load and split the dataset into training, validation, and test sets.
    2. Preprocess the data by normalizing numeric features and encoding categorical features.
    3. Build a neural network model suitable for the classification task.
    4. Train the model with the training dataset.
    5. Evaluate the model's performance with the test dataset.

    The workflow is designed to be executed with Prefect, managing the execution of each task and handling their dependencies.
    The output of the workflow includes the model evaluation results on the test dataset, comprising loss and accuracy metrics.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(dataset_path)
    X_train_processed, X_val_processed, X_test_processed = preprocess_data(X_train, X_val, X_test)
    input_dim = X_train_processed.shape[1]
    output_dim = y_train.shape[1]
    model = build_model(input_dim, output_dim)
    history = train_model(model, X_train_processed, y_train, X_val_processed, y_val, epochs=2, batch_size=32)
    loss, accuracy = evaluate_model(model, X_test_processed, y_test)

if __name__ == "__main__":
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'app', 'data', 'winequality.csv')
    )
    wine_quality_prediction_flow(dataset_path)
