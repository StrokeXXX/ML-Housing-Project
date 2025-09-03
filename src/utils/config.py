"""
Configuration du projet
"""

import os
from pathlib import Path

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Configuration MLflow
EXPERIMENT_NAME = "housing-price-prediction"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # ou "http://localhost:5000"

# Configuration du modèle
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Configuration des hyperparamètres par défaut
DEFAULT_LINEAR_PARAMS = {
    "fit_intercept": True,
    "normalize": False
}

DEFAULT_RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}

# Configuration logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
