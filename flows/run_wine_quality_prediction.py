from scripts.prefect.model_predictions import wine_quality_prediction_flow
import os

def main():
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'winequality.csv')
        # os.path.join(os.path.dirname(__file__), '..', 'data', 'winequality.csv')
    )
    wine_quality_prediction_flow(dataset_path)

if __name__ == "__main__":
    main()
