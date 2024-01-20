import pandas as pd
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

from prefect import task, flow

@task(retries=3, retry_delay_seconds=10)
def load_and_prepare_data(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge et prépare les données pour l'entraînement et l'évaluation.

    Returns:
    tuple: Retourne les ensembles de données divisés en train, validation et test.
    """
    df = pd.read_csv(dataset_path)

    # Réduction des classes de qualité
    df["quality"] = df["quality"].apply(lambda x: 0 if x < 6 else (1 if x == 6 else 2))

    y = to_categorical(df["quality"])
    X = df.drop("quality", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    return X_train, X_val, X_test, y_train, y_val, y_test

@task(retries=2, retry_delay_seconds=5)
def preprocess_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prétraite les données en appliquant la normalisation et le codage.

    Args:
    X_train, X_val, X_test (DataFrame): Ensembles de données à prétraiter.

    Returns:
    tuple: Retourne les ensembles de données prétraités.
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

@task
def build_model(input_dim: int, output_dim: int) -> Sequential:
    """
    Construit un modèle de réseau de neurones.

    Args:
    input_dim (int): Dimension de l'entrée.
    output_dim (int): Dimension de la sortie.

    Returns:
    Sequential: Un modèle de réseau de neurones.
    """
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@task(retries=1, retry_delay_seconds=30)
def train_model(model: Sequential, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, epochs: int = 2, batch_size: int = 32) -> History:
    """
    Entraîne le modèle de réseau de neurones.

    Args:
    model (Sequential): Le modèle à entraîner.
    X_train, y_train, X_val, y_val (ndarray): Données d'entraînement et de validation.
    epochs (int): Nombre d'époques pour l'entraînement.
    batch_size (int): Taille du batch pour l'entraînement.

    Returns:
    History: L'historique de l'entraînement.
    """
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return history

@task(retries=1, retry_delay_seconds=5)
def evaluate_model(model: Sequential, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[float, float]:
    """
    Évalue le modèle sur l'ensemble de test.

    Args:
    model (Sequential): Le modèle à évaluer.
    X_test, y_test (ndarray): Données de test.

    Returns:
    tuple: Perte et précision sur l'ensemble de test.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    return loss, accuracy

@flow(name="Wine quality prediction flow")
def wine_quality_flow(dataset_path: str):
    """
    Orchestre un workflow pour charger, prétraiter, entraîner et évaluer un modèle de prédiction de la qualité du vin.

    Args:
    dataset_path (str): Chemin vers le fichier du dataset de qualité du vin.

    Ce workflow exécute plusieurs tâches en séquence :
    1. Charge et divise le dataset en ensembles d'entraînement, de validation et de test.
    2. Prétraite les données en normalisant les caractéristiques numériques et en encodant les caractéristiques catégorielles.
    3. Construit un modèle de réseau de neurones adapté à la tâche de classification.
    4. Entraîne le modèle avec l'ensemble de données d'entraînement.
    5. Évalue les performances du modèle avec l'ensemble de données de test.

    Le workflow est conçu pour être exécuté avec Prefect, qui gère l'exécution de chaque tâche et s'occupe de leurs dépendances.
    La sortie du workflow est constituée des résultats de l'évaluation du modèle sur l'ensemble de test, incluant les métriques de perte et de précision.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(dataset_path)
    X_train_processed, X_val_processed, X_test_processed = preprocess_data(X_train, X_val, X_test)
    input_dim = X_train_processed.shape[1]
    output_dim = y_train.shape[1]
    model = build_model(input_dim, output_dim)
    history = train_model(model, X_train_processed, y_train, X_val_processed, y_val, epochs=2, batch_size=32)
    loss, accuracy = evaluate_model(model, X_test_processed, y_test)


if __name__ == "__main__":
    # wine_quality_flow("winequality.csv")
     wine_quality_flow.from_source(
        source="https://github.com/ROL-1/5MLDE-soutenace.git", 
        entrypoint="5MLDE_proj_wine_light.py:wine_quality_flow"
    ).deploy(
        name="WineQualityDeployment",
        work_pool_name="my-managed-pool",
    )
    # wine_quality_flow.serve(
    #     name="WineServeDeployment",
    #     tags=["ml", "wine-quality"],
    #     # cron="0 1 * * *",  # Exécution quotidienne à 1h du matin
    #     description="Deployment for wine quality prediction model",
    # )

    