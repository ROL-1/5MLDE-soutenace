
# MLOps Project Overview

## Introduction
This project is an end-to-end MLOps pipeline designed to automate and streamline the processes of data ingestion, quality checking, model training, performance logging, and deployment. Our approach leverages the latest technologies in data processing and machine learning operations to ensure efficiency and accuracy.

## Environment
- **Docker Desktop & WSL:** Our infrastructure is containerized using Docker, providing a consistent and isolated environment for development and deployment. WSL (Windows Subsystem for Linux) enables us to seamlessly integrate Linux-based tools within a Windows environment.
- **Prefect Cloud:** For workflow orchestration, we utilize Prefect Cloud to manage our data pipelines. This choice eliminates the need for local infrastructure management. Users must create an account, generate an API key, and manage workflows via the Prefect dashboard.

## Deployment
The deployment process is automated and scheduled to run weekly. It involves the following sequential steps:
1. **Data Wrangling:** Cleansing and preparing the data for analysis.
2. **Preliminary Analysis:** Analyzing the data to extract insights and inform model training.
3. **Data Prediction:** Utilizing the trained model to make predictions on new data.

This sequence is initiated through a Docker image's entry point, ensuring a streamlined and consistent execution.

## Great Expectations
This tool is integral for maintaining data quality. Through a YAML configuration file, it defines and executes a series of tests on the dataset, generating JSON files that represent test cases for each dataset column. This process is ideal for validating new datasets as they are ingested.

## MLFlow
MLFlow serves as our primary tool for model management. It provides a user interface for monitoring and comparing model performances, logging parameters, and maintaining a registry of models. This system enhances the observability and traceability of our model training processes.

## API
Our API is built using FastAPI, which offers a lightweight and efficient way to handle requests. The primary endpoint, `predict()`, accepts input data and utilizes the trained model to return predictions. Pydantic is used for robust JSON data validation, ensuring that all inputs meet the predefined schema requirements.

## Frontend Client
The frontend is a Single Page Application (SPA) designed to interact with the API. It allows users to input features into a form, sends these to the API for prediction, and displays the results. The SPA enhances user experience by providing an intuitive interface for interacting with the model.

## Testing
Testing is handled via PyTest, ensuring reliability and robustness of our data processing scripts. We employ unit tests, fixtures, and mocks to validate the functionality of the data-wrangling and exploratory-data-analysis components.

## Setup Instructions
To set up the project, follow these steps:
1. Ensure Docker Desktop and WSL are installed and configured.
2. Clone the project repository.
3. Navigate to the project directory and run `docker composer --build` to build and start the services.

## Service Overviews
- **Prefect:** Automates and schedules the execution of data pipelines.
- **Great Expectations:** Ensures data quality and integrity through automated testing.
- **MLFlow:** Manages and tracks machine learning models, facilitating model selection and deployment.
- **API (FastAPI):** Serves as the interface for model predictions, handling requests and responses efficiently.
- **Frontend (SPA):** Provides a user-friendly interface for interacting with the model, enhancing accessibility and usability.
- **PyTest:** Guarantees the reliability of our data processing and analysis scripts through comprehensive testing.
