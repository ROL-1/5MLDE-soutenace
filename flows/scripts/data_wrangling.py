from prefect import task, flow
from prefect import get_run_logger

from sklearn.impute import SimpleImputer
import pandas as pd
import os
import logging

@task(name='load_data_1', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def load_data(file_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: The loaded data, or raises an exception if loading fails.
    """
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            logger.warning("The loaded data is empty.")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

@task(name='check_df', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def check_dataframe(df: pd.DataFrame, step: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Task to check if a DataFrame is empty.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        step (str): Description of the current step for logging.

    Raises:
        ValueError: If the DataFrame is empty.

    Returns:
        pd.DataFrame: The original DataFrame if not empty, or raises an exception if the check fails.
    """
    try:
        if df.empty:
            raise ValueError(f"The DataFrame is empty after {step}.")
        return df
    except Exception as e:
        logger.error(f"DataFrame check failed at {step}: {e}")
        raise

@task(name='data_types', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def explore_data(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Explore data types and nature of the dataset.

    Args:
        df (DataFrame): The DataFrame to explore.

    Raises:
        Exception: If data exploration fails.
    """
    try:
        print(df.dtypes)
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            print(f"{column} has {df[column].nunique()} unique values: {df[column].unique()}")
    except Exception as e:
        logger.error(f"Data exploration failed: {e}")
        raise

@task(name='duplicates_data', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def remove_duplicates(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (DataFrame): The DataFrame to clean.

    Returns:
        DataFrame: The DataFrame without duplicates, or raises an exception if the removal fails.
    """
    try:
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - df.shape[0]} duplicate rows.")
        return df
    except Exception as e:
        logger.error(f"Duplicate removal failed: {e}")
        raise

@task(name='missing_rows', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def missing_data_percentage(df: pd.DataFrame, logger: logging.Logger) -> pd.Series:
    """
    Calculate the percentage of missing data for each column.

    Args:
        df (DataFrame): The DataFrame to analyze.

    Returns:
        Series: A Series with the percentage of missing data for each column, or raises an exception if the calculation fails.
    """
    try:
        missing_percentage = df.isnull().mean() * 100
        return missing_percentage[missing_percentage > 0].sort_values(ascending=False)
    except Exception as e:
        logger.error(f"Calculation of missing data percentage failed: {e}")
        raise

@task(name='quantitative_columns_imputer', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def impute_numeric_data(df: pd.DataFrame, logger: logging.Logger, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute missing values in numeric columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to impute.
        strategy (str, optional): The imputation strategy for numeric data ('mean', 'median', 'most_frequent', or 'constant'). Defaults to 'median'.

    Returns:
        pd.DataFrame: The DataFrame with imputed numeric values, or raises an exception if imputation fails.
    """
    try:
        numeric_columns = df.select_dtypes(include=['number']).columns
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        return df
    except Exception as e:
        logger.error(f"Numeric data imputation failed: {e}")
        raise

@task(name='categorical_columns_imputer', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def impute_categorical_data(df: pd.DataFrame, logger: logging.Logger, strategy: str = 'most_frequent', fill_value: str = None) -> pd.DataFrame:
    """
    Impute missing values in categorical columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to impute.
        strategy (str, optional): The imputation strategy for categorical data ('most_frequent' or 'constant'). Defaults to 'most_frequent'.
        fill_value (str, optional): The constant value to use for imputation if strategy is 'constant'. If None, 'missing' is used.

    Returns:
        pd.DataFrame: The DataFrame with imputed categorical values, or raises an exception if imputation fails.
    """
    
    try:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if strategy == 'constant' and fill_value is None:
            fill_value = 'missing'

        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
        return df
    except Exception as e:
        logger.error(f"Categorical data imputation failed: {e}")
        raise

@flow(name="Data Cleaning Workflow")
def clean_data_flow(file_path: str):
    logger = get_run_logger()

    data = load_data(file_path, logger)
    if data is None or data.empty:
        return

    data = check_dataframe(data, "loading data", logger)
    explore_data(data, logger)
    deduped_data = remove_duplicates(data, logger)
    deduped_data = check_dataframe(deduped_data, "removing duplicates", logger)
    missing_perc = missing_data_percentage(deduped_data, logger)
    imputed_numeric_data = impute_numeric_data(deduped_data, logger)
    imputed_numeric_data = check_dataframe(imputed_numeric_data, "imputing numeric data", logger)
    final_cleaned_data = impute_categorical_data(imputed_numeric_data, logger)
    final_cleaned_data = check_dataframe(final_cleaned_data, "imputing categorical data", logger)

    return final_cleaned_data

if __name__ == "__main__":
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'data', 'winequality.csv')
    )
    clean_data_flow(dataset_path)