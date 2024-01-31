import pytest
from unittest.mock import Mock, patch
import pandas as pd
import os
import sys

PREFECT_FLOW_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PREFECT_FLOW_PATH)

from scripts.prefect.exploratory_data_analysis import (
    load_data,
    plot_boxplot,
    plot_histogram,
    plot_grouped_histogram,
    plot_pie_chart,
    plot_correlation_heatmap,
    plot_bubble_chart,
    plot_quality_histogram,
    plot_scatter_with_regression
)

@pytest.fixture
def sample_data():
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


logger = Mock()

def test_load_data(sample_data):
    with patch('pandas.read_csv', return_value=sample_data):
        data = load_data.fn('../../data/winequality.csv', logger)
        assert not data.empty

def test_plot_boxplot(sample_data):
    with patch('matplotlib.pyplot.show'):
        plot_boxplot.fn(sample_data, ['alcohol', 'pH'], logger)

def test_plot_histogram(sample_data):
    with patch('matplotlib.pyplot.show'):
        plot_histogram.fn(sample_data, 'alcohol', logger)

def test_plot_grouped_histogram(sample_data):
    with patch('matplotlib.pyplot.show'):
        plot_grouped_histogram.fn(sample_data, 'alcohol', 'quality', logger)

def test_plot_pie_chart(sample_data):
    with patch('matplotlib.pyplot.show'):
        plot_pie_chart.fn(sample_data, 'quality', logger)

def test_plot_correlation_heatmap(sample_data):
    with patch('matplotlib.pyplot.show'):
        plot_correlation_heatmap.fn(sample_data, ['alcohol', 'pH'], logger)

def test_plot_bubble_chart(sample_data):
    with patch('matplotlib.pyplot.show'):
        plot_bubble_chart.fn(sample_data, 'pH', 'alcohol', 'quality', logger)

def test_plot_quality_histogram(sample_data):
    with patch('matplotlib.pyplot.show'):
        plot_quality_histogram.fn(sample_data, 'alcohol', 'quality', logger)

def test_plot_scatter_with_regression(sample_data):
    with patch('matplotlib.pyplot.show'):
        plot_scatter_with_regression.fn(sample_data, ['alcohol'], 'quality', logger)

if __name__ == "__main__":
    pytest.main()
