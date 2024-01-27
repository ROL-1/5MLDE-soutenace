from prefect import flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from scripts.data_wrangling import clean_data_flow
from scripts.exploratory_data_analysis import data_visualization_flow
from scripts.model_predictions import wine_quality_prediction_flow

import os

@flow
def data_cleaning_visualization_and_prediction():
    dataset_path = os.path.abspath(
                     os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'winequality.csv')
                    )
    clean_data_flow(dataset_path)
    data_visualization_flow(dataset_path,  wait_for=clean_data_flow)
    wine_quality_prediction_flow(dataset_path, wait_for=data_visualization_flow)

mlde_project_deployment = Deployment.build_from_flow(
    flow=data_cleaning_visualization_and_prediction,
    name="MLDE Project Deployment",
    version="1.0",
    schedule=CronSchedule(cron="0 0 * * 0"),  
    tags=["mlde-project"]
)

def main_flow():
    mlde_project_deployment.apply()
    
if __name__ == "__main__":
    main_flow()
