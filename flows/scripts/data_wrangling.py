from prefect import task, flow
from prefect import get_run_logger

from sklearn.impute import SimpleImputer
import pandas as pd
import os

@task(name='load_data', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

@task(name='check_df', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def check_dataframe(df: pd.DataFrame, step: str) -> pd.DataFrame:
    """
    Task to check if a DataFrame is empty.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        step (str): Description of the current step for logging.

    Raises:
        ValueError: If the DataFrame is empty.

    Returns:
        pd.DataFrame: The original DataFrame if not empty.
    """
    if df.empty:
        raise ValueError(f"The DataFrame is empty after {step}.")
    return df

@task(name='data_types', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def explore_data(df: pd.DataFrame) -> None:
    """
    Explore data types and nature of the dataset.

    Args:
        df (DataFrame): The DataFrame to explore.
    """
    print(df.dtypes)
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        print(f"{column} has {df[column].nunique()} unique values: {df[column].unique()}")

@task(name='duplicates_data', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (DataFrame): The DataFrame to clean.

    Returns:
        DataFrame: The DataFrame without duplicates.
    """
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - df.shape[0]} duplicate rows.")
    return df

@task(name='missing_rows', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def missing_data_percentage(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the percentage of missing data for each column.

    Args:
        df (DataFrame): The DataFrame to analyze.

    Returns:
        Series: A Series with the percentage of missing data for each column.
    """
    missing_percentage = df.isnull().mean() * 100
    return missing_percentage[missing_percentage > 0].sort_values(ascending=False)

@task(name='quantitative_columns_imputer', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def impute_numeric_data(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute missing values in numeric columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to impute.
        strategy (str, optional): The imputation strategy for numeric data ('mean', 'median', 'most_frequent', or 'constant'). Defaults to 'median'.

    Returns:
        pd.DataFrame: The DataFrame with imputed numeric values.
    """
    numeric_columns = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

@task(name='categorical_columns_imputer', tags=['data-cleaning'], retries=2, retry_delay_seconds=60)
def impute_categorical_data(df: pd.DataFrame, strategy: str = 'most_frequent', fill_value: str = None) -> pd.DataFrame:
    """
    Impute missing values in categorical columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to impute.
        strategy (str, optional): The imputation strategy for categorical data ('most_frequent' or 'constant'). Defaults to 'most_frequent'.
        fill_value (str, optional): The constant value to use for imputation if strategy is 'constant'. If None, 'missing' is used.

    Returns:
        pd.DataFrame: The DataFrame with imputed categorical values.
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if strategy == 'constant' and fill_value is None:
        fill_value = 'missing'

    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
    return df

@flow(name="Data Cleaning Workflow")
def clean_data_flow(file_path: str):
    """
    Flow to clean the data by performing various cleaning steps.

    Args:
        file_path (str): The path to the data file.
    """
    logger = get_run_logger()

    try:
        data = load_data(file_path)
        if data.empty:
            logger.warning("The loaded data is empty. Exiting flow.")
            return
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    try:
        explore_data(data)
    except Exception as e:
        logger.error(f"Failed to explore data: {e}")
        return

    try:
        deduped_data = remove_duplicates(data)
        deduped_data = check_dataframe(deduped_data, "removing duplicates")
    except Exception as e:
        logger.error(f"Failed to remove duplicates: {e}")
        return

    try:
        missing_perc = missing_data_percentage(deduped_data)
        if missing_perc.empty:
            logger.info("No missing data found.")
    except Exception as e:
        logger.error(f"Failed to calculate missing data percentage: {e}")
        return

    try:
        imputed_numeric_data = impute_numeric_data(deduped_data)
        imputed_numeric_data = check_dataframe(imputed_numeric_data, "imputing numeric data")
    except Exception as e:
        logger.error(f"Failed to impute numeric data: {e}")
        return

    try:
        final_cleaned_data = impute_categorical_data(imputed_numeric_data)
        final_cleaned_data = check_dataframe(final_cleaned_data, "imputing categorical data")
    except Exception as e:
        logger.error(f"Failed to impute categorical data: {e}")
        return

    logger.info("Data cleaning flow completed successfully.")
    return final_cleaned_data

if __name__ == "__main__":
    dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'data', 'winequality.csv')
        )
    clean_data_flow(file_path=dataset_path)
