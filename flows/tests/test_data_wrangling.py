import pytest
import pandas as pd
import os
import sys

PREFECT_FLOW_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PREFECT_FLOW_PATH)

from scripts.prefect.data_wrangling import (
    load_data, 
    check_dataframe,
    explore_data,
    remove_duplicates, 
    missing_data_percentage,
    impute_numeric_data,
    impute_categorical_data
)

from config import (
    get_logger
)

TEST_CSV_PATH = "../../data/winequality.csv"
EMPTY_CSV_PATH = "fake_dataset.csv"

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return pd.DataFrame({
            'type': ['white', 'white', 'white', 'white', 'red', 'white', 'white', 'white', 'red', 'white', 'white', 'red', 'white', 'white', 'white', 'white'],
            'fixed_acidity': [7.0, 6.3, 8.1, 7.2, 7.2, 8.1, 6.2, 7.0, 6.3, 8.1, 8.1, 8.6, 7.9, 6.6, 8.3, 6.6],
            'volatile_acidity': [0.27, 0.3, 0.28, 0.23, 0.23, 0.28, 0.32, 0.27, 0.3, 0.22, 0.27, 0.23, 0.18, 0.16, 0.42, 0.17],
            'citric_acid': [0.36, 0.34, 0.4, 0.32, 0.32, 0.4, 0.16, 0.36, 0.34, 0.43, 0.41, 0.4, 0.37, 0.4, None, 0.38],
            'residual_sugar': [20.7, 1.6, 6.9, 8.5, 8.5, 6.9, 7.0, 20.7, 1.6, 1.5, 1.45, 4.2, 1.2, 1.5, 19.25, 1.5],
            'chlorides': [0.045, 0.049, 0.05, 0.058, 0.058, 0.05, 0.045, 0.045, 0.049, 0.044, 0.033, 0.035, 0.04, 0.044, 0.04, 0.032],
            'free_sulfur_dioxide': [45, 14, 30, 47, 47, 30, 30, 45, 14, 28, 11, 17, 16, 48, 41, 28],
            'total_sulfur_dioxide': [170, 132, 97, 186, 186, 97, 136, 170, None, 129, 63, 109, 75, 143, 172, 112],
            'density': [1.001, 0.994, 0.9951, 0.9956, 0.9956, 0.9951, 0.9949, 1.001, 0.994, 0.9938, 0.9908, 0.9947, 0.992, 0.9912, 1.0002, 0.9914],
            'pH': [3.0, 3.3, 3.26, 3.19, 3.19, 3.26, 3.18, 3.0, 3.3, 3.22, 2.99, 3.14, 3.18, 3.54, 2.98, 3.25],
            'sulphates': [0.45, 0.49, 0.44, 0.4, 0.4, 0.44, 0.47, 0.45, 0.49, 0.45, 0.56, 0.53, 0.63, 0.52, 0.67, 0.55],
            'alcohol': [8.8, 9.5, 10.1, 9.9, 9.9, 10.1, 9.6, None, 9.5, 11, 12, 9.7, 10.8, 12.4, 9.7, 11.4],
            'quality': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 7, 5, 7]
    })

@pytest.fixture
def empty_data():
    """Provide empty data for testing."""
    return pd.DataFrame()

def test_load_data_success():
    """Test loading data successfully."""
    data = load_data.fn(TEST_CSV_PATH, get_logger())
    assert not data.empty

def test_load_data_failure():
    """Test loading data with an invalid path."""
    with pytest.raises(Exception):
        load_data.fn("invalid/path.csv", get_logger())

def test_check_dataframe_non_empty(sample_data):
    """Test checking a non-empty DataFrame."""
    result = check_dataframe.fn(sample_data, "testing non-empty", get_logger())
    assert not result.empty

def test_check_dataframe_empty(empty_data):
    """Test checking an empty DataFrame."""
    with pytest.raises(ValueError):
        check_dataframe.fn(empty_data, "testing empty", get_logger())

def test_explore_data_types(sample_data, capsys):
    """Test exploring data types."""
    explore_data.fn(sample_data, get_logger())
    captured = capsys.readouterr()
    assert "fixed_acidity" in captured.out
    assert "volatile_acidity" in captured.out
    assert "citric_acid" in captured.out
    assert "type" in captured.out

def test_remove_duplicates(sample_data):
    """Test removing duplicate rows."""
    deduped_data = remove_duplicates.fn(sample_data, get_logger())
    assert not deduped_data.shape[0] == 3

@pytest.mark.parametrize("strategy, expected_value", [("mean", 0.25), ("median", 0.25)])
def test_impute_numeric_data_strategy(sample_data, strategy, expected_value):
    """Test imputing numeric data with different strategies."""
    imputed_data = impute_numeric_data.fn(sample_data, get_logger(), strategy=strategy)
    assert imputed_data['volatile_acidity'].fillna(0).mean() == expected_value

def test_impute_categorical_data(sample_data):
    """Test imputing categorical data."""
    imputed_data = impute_categorical_data.fn(sample_data, get_logger())
    assert not imputed_data['type'].isnull().any()

def test_missing_data_percentage_no_missing(sample_data):
    """Test calculating missing data percentage with no missing data."""
    sample_data = sample_data.fillna(0)
    missing_perc = missing_data_percentage.fn(sample_data, get_logger())
    assert missing_perc.empty

if __name__ == "__main__":
    pytest.main()
