import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prefect import task, flow
from prefect import get_run_logger
import os
import logging

@task(name='load_data_2', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def load_data(file_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load data from a CSV file.
    Args:
        file_path (str): The path to the CSV file.
        logger (logging.Logger): Logger for logging events.
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

@task(name='plot_boxplot', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def plot_boxplot(df: pd.DataFrame, columns: list, logger: logging.Logger) -> None:
    """
    Generate box plots for specified columns in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        columns (list): Columns to plot.
        logger (logging.Logger): Logger for logging events.
    """
    try:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[columns], orient='h', palette='Set2')
        plt.title(f"Box Plot of {', '.join(columns)}")
        plt.xlabel("Value")
        plt.ylabel("Column")
        plt.xticks(rotation=45)
        plt.show()
    except Exception as e:
        logger.error(f"Error generating box plot: {e}")

@task(name='plot_histogram', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def plot_histogram(df: pd.DataFrame, column: str, logger: logging.Logger) -> None:
    """
    Generate a histogram for a specified column in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        column (str): Column to plot.
        logger (logging.Logger): Logger for logging events.
    """
    try:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[column], kde=True, color='skyblue')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
    except Exception as e:
        logger.error(f"Error generating histogram for {column}: {e}")

@task(name='plot_grouped_histogram', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def plot_grouped_histogram(df: pd.DataFrame, column: str, group_by: str, logger: logging.Logger) -> None:
    """
    Generate grouped histograms for a specified column in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        column (str): Column to plot.
        group_by (str): Column to group by.
        logger (logging.Logger): Logger for logging events.
    """
    try:
        g = sns.FacetGrid(df, col=group_by)
        g.map(plt.hist, column, color='purple')
        plt.show()
    except Exception as e:
        logger.error(f"Error generating grouped histogram for {column}: {e}")

@task(name='plot_pie_chart', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def plot_pie_chart(df: pd.DataFrame, column: str, logger: logging.Logger) -> None:
    """
    Generate a pie chart for a specified column in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        column (str): Column to plot.
        logger (logging.Logger): Logger for logging events.
    """
    try:
        value_counts = df[column].value_counts()
        if len(value_counts) > 10:
            logger.warning(f"Too many unique values in {column} for a pie chart.")
            return

        plt.figure(figsize=(8, 8))
        value_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title(f"Pie Chart of {column}")
        plt.show()
    except Exception as e:
        logger.error(f"Error generating pie chart for {column}: {e}")

@task(name='plot_correlation_heatmap', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list, logger: logging.Logger) -> None:
    """
    Generate a correlation heatmap for specified numeric columns in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        numeric_columns (list): Numeric columns to include in the heatmap.
    """
    try:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {e}")

@task(name='plot_bubble_chart', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def plot_bubble_chart(df: pd.DataFrame, x_1: str, x_2: str, y: str, logger: logging.Logger) -> None:
    """
    Generate a bubble chart for two numeric columns and use a third column for bubble size.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        x_1 (str): First numeric column.
        x_2 (str): Second numeric column.
        y (str): Column for bubble size.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_1], df[x_2], s=df[y]*10, c=df[y], cmap='viridis', alpha=0.7)
        plt.xlabel(x_1)
        plt.ylabel(x_2)
        plt.title(f"Bubble Chart of '{x_1}' vs '{x_2}' with Size by '{y}'")
        plt.colorbar(label=y)
        plt.show()
    except Exception as e:
        logger.error(f"Error generating bubble chart: {e}")

@task(name='plot_quality_histogram', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def plot_quality_histogram(df: pd.DataFrame, x: str, y: str, logger: logging.Logger) -> None:
    """
    Generate histograms for a specified column grouped by another column.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        x (str): Column for the histogram.
        y (str): Column to group by.
    """
    try:
        plt.figure(figsize=(8, 4))
        for quality in sorted(df[y].unique()):
            sns.histplot(df[df[y] == quality][x], kde=True, label=f'Quality {quality}', alpha=0.6)
        plt.title(f"Histogram of '{x}' by Quality")
        plt.xlabel(x)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    except Exception as e:
        logger.error(f"Error generating quality histograms: {e}")

@task(name='plot_scatter_with_regression', tags=['data-analysis'], retries=2, retry_delay_seconds=60)
def plot_scatter_with_regression(df: pd.DataFrame, x_columns: list, y_column: str, logger: logging.Logger) -> None:
    """
    Generate scatter plots with regression lines for specified columns.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        x_columns (list): List of columns to use as the x-axis.
        y_column (str): Column to use as the y-axis.
        logger (logging.Logger): Logger for logging events.
    """
    try:
        num_plots = len(x_columns)
        num_cols = 3
        num_rows = (num_plots - 1) // num_cols + 1
        
        colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray"]

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

        for i, x_column in enumerate(x_columns):
            if x_column not in df.columns or y_column not in df.columns or x_column == y_column:
                logger.error(f"Invalid columns '{x_column}' or '{y_column}' or x_column equals y_column. Skipping.")
                continue

            row = i // num_cols
            col = i % num_cols

            ax = axes[row, col]
            sns.regplot(data=df, x=x_column, y=y_column, ax=ax, color='black', scatter_kws={"color": colors[i % len(colors)]})
            ax.set_title(f"Regression Plot: {x_column} vs {y_column}")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        for i in range(num_plots, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

        logger.info('Showing the regression plot...')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error generating scatter plots with regression for columns {x_columns} vs {y_column}: {e}")

@flow(name="Exploratory Data Analysis Workflow")
def data_visualization_flow(file_path: str):
    """
    Flow to perform data visualization on a dataset.
    Args:
        file_path (str): File path to the dataset.
    """
    sns.set_style("whitegrid")
    logger = get_run_logger()

    try:
        df = load_data(file_path, logger)
        plot_boxplot(df, ['alcohol', 'pH'], logger)
        plot_histogram(df, 'alcohol', logger)
        plot_grouped_histogram(df, 'alcohol', 'quality', logger)
        plot_pie_chart(df, 'quality', logger)

        numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        plot_correlation_heatmap(df, numeric_columns, logger)
        plot_scatter_with_regression(df, numeric_columns, 'quality', logger)

        plot_bubble_chart(df, 'pH', 'alcohol', 'quality', logger)
        plot_quality_histogram(df, 'alcohol', 'quality', logger)
    except Exception as e:
        logger.error(f"Error in data visualization flow: {e}")

if __name__ == "__main__":
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'app', 'data', 'winequality.csv')
    )
    data_visualization_flow(dataset_path)