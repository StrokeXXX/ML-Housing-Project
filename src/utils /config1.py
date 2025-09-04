import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class Environment(Enum):
    LOCAL = "local"
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class Config:
    # Chemins
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
    TESTS_DIR: Path = PROJECT_ROOT / "tests"
    
    # MLflow
    EXPERIMENT_NAME: str = "housing-price-prediction"
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    
    # Modèle
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.2
    
    # Environnement
    ENV: Environment = Environment.LOCAL
    
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    def __post_init__(self):
        # Créer les dossiers s'ils n'existent pas
        for directory in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, 
                         self.MODELS_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Instance de configuration globale
config = Config()
