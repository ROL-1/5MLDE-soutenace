from pathlib import Path
import logging

def get_logger(logging_level=logging.INFO, logger_name: str = "model_deployment_logger"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def is_docker() -> bool:
    cgroup = Path("/proc/self/cgroup")
    return Path('/.dockerenv').is_file() or cgroup.is_file() and cgroup.read_text().find("docker") > -1

# LOGGING
LOGGER_LEVEL = "INFO"
logger = get_logger(logging_level=getattr(logging, LOGGER_LEVEL))

# MLFLOW
MLFLOW_TRACKING_URI = "http://mlflow:5000" if is_docker() else "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "Wine Quality Prediction"
REGISTERED_MODEL_NAME = "wine_quality_model"