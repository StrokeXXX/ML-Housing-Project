# src/data/data_loader.py
"""
Module pour le chargement et préprocessing des données
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_housing_data(return_df=False):
    """
    Charge le dataset California Housing
    
    Args:
        return_df (bool): Si True, retourne un DataFrame combiné
        
    Returns:
        tuple: (X, y) ou DataFrame si return_df=True
    """
    logger.info("Chargement du dataset California Housing...")
    
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='MedHouseVal')
    
    logger.info(f"Dataset chargé: {X.shape[0]} échantillons, {X.shape[1]} features")
    
    if return_df:
        return pd.concat([X, y], axis=1)
    
    return X, y

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Crée un split train/test
    
    Args:
        X: Features
        y: Target
        test_size (float): Proportion du test set
        random_state (int): Seed pour la reproductibilité
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info(f"Création du split train/test (test_size={test_size})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train: {len(X_train)} échantillons")
    logger.info(f"Test: {len(X_test)} échantillons")
    
    return X_train, X_test, y_train, y_test

def prepare_data_for_linear_model(X_train, X_test):
    """
    Prépare les données pour un modèle linéaire (normalisation)
    
    Args:
        X_train: Features d'entraînement
        X_test: Features de test
        
    Returns:
        tuple: X_train_scaled, X_test_scaled, scaler
    """
    logger.info("Normalisation des données pour modèle linéaire")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

# ================================================================
# src/models/training.py
"""
Module pour l'entraînement des modèles ML avec MLflow tracking
"""

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import logging

from ..utils.config import EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurer MLflow
mlflow.set_experiment(EXPERIMENT_NAME)

class ModelTrainer:
    """Classe pour l'entraînement des modèles avec MLflow tracking"""
    
    def __init__(self):
        self.scaler = None
        self.model = None
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test, 
                              run_name="linear_regression"):
        """
        Entraîne un modèle de régression linéaire
        
        Args:
            X_train, X_test, y_train, y_test: Données d'entraînement et test
            run_name (str): Nom de l'expérience MLflow
            
        Returns:
            dict: Métriques du modèle
        """
        with mlflow.start_run(run_name=run_name):
            logger.info(f"Entraînement: {run_name}")
            
            # Preprocessing
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Modèle
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            
            # Évaluation
            from .evaluation import ModelEvaluator
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(
                self.model, X_test_scaled, y_test
            )
            
            # Logging MLflow
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("scaling", "StandardScaler")
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log des modèles
            y_pred = self.model.predict(X_test_scaled)
            signature = infer_signature(X_train, y_pred)
            
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )
            mlflow.sklearn.log_model(self.scaler, "scaler")
            
            logger.info(f"Métriques: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            
            return metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test,
                          run_name="random_forest", 
                          param_grid=None):
        """
        Entraîne un Random Forest avec hyperparameter tuning
        
        Args:
            X_train, X_test, y_train, y_test: Données d'entraînement et test
            run_name (str): Nom de l'expérience MLflow
            param_grid (dict): Paramètres pour GridSearchCV
            
        Returns:
            dict: Métriques du modèle
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        with mlflow.start_run(run_name=run_name):
            logger.info(f"Entraînement: {run_name}")
            
            # GridSearch
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, 
                scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            # Évaluation
            from .evaluation import ModelEvaluator
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(
                self.model, X_test, y_test
            )
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Logging MLflow
            mlflow.log_param("model_type", "RandomForestRegressor")
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(param, value)
            
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_metric("cv_score", -grid_search.best_score_)
            
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log du modèle
            y_pred = self.model.predict(X_test)
            signature = infer_signature(X_train, y_pred)
            
            mlflow.sklearn.log_model(
                self.model,
                "model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )
            
            # Log feature importance
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            logger.info(f"Métriques: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            logger.info(f"Meilleurs paramètres: {grid_search.best_params_}")
            
            return metrics

# ================================================================
# src/models/evaluation.py
"""
Module pour l'évaluation des modèles ML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Classe pour l'évaluation des modèles"""
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Évalue un modèle et retourne les métriques
        
        Args:
            model: Modèle entraîné
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            dict: Dictionnaire des métriques
        """
        logger.info("Évaluation du modèle...")
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        
        return metrics
    
    def plot_predictions(self, model, X_test, y_test, save_path=None):
        """
        Crée un graphique des prédictions vs valeurs réelles
        
        Args:
            model: Modèle entraîné
            X_test: Features de test
            y_test: Target de test
            save_path (str): Chemin pour sauvegarder le graphique
        """
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title('Prédictions vs Valeurs Réelles')
        
        # Ajouter les métriques sur le graphique
        metrics = self.evaluate_model(model, X_test, y_test)
        textstr = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}"
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return 'predictions_plot.png'
    
    def plot_residuals(self, model, X_test, y_test, save_path=None):
        """
        Crée un graphique des résidus
        
        Args:
            model: Modèle entraîné
            X_test: Features de test
            y_test: Target de test
            save_path (str): Chemin pour sauvegarder le graphique
        """
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Résidus vs prédictions
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Prédictions')
        axes[0].set_ylabel('Résidus')
        axes[0].set_title('Résidus vs Prédictions')
        
        # Distribution des résidus
        axes[1].hist(residuals, bins=30, alpha=0.7)
        axes[1].set_xlabel('Résidus')
        axes[1].set_ylabel('Fréquence')
        axes[1].set_title('Distribution des Résidus')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return 'residuals_plot.png'

# ================================================================
# src/utils/config.py
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

# ================================================================
# tests/test_models.py
"""
Tests unitaires pour les modèles
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

from src.data.data_loader import load_housing_data, create_train_test_split
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator

class TestDataLoader:
    """Tests pour le chargement des données"""
    
    def test_load_housing_data(self):
        """Test le chargement du dataset California Housing"""
        X, y = load_housing_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] > 0
        assert X.shape[1] == 8  # 8 features dans California Housing
        assert len(X) == len(y)
    
    def test_train_test_split(self):
        """Test la création du split train/test"""
        X, y = load_housing_data()
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_test) == int(0.2 * len(X))

class TestModelTrainer:
    """Tests pour l'entraînement des modèles"""
    
    @pytest.fixture
    def sample_data(self):
        """Crée des données synthétiques pour les tests"""
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(y, name='target')
        
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test
    
    def test_linear_regression_training(self, sample_data):
        """Test l'entraînement du modèle de régression linéaire"""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer()
        metrics = trainer.train_linear_regression(
            X_train, X_test, y_train, y_test, run_name="test_linear"
        )
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mae' in metrics
        assert metrics['r2'] > 0  # R² devrait être positif pour des données cohérentes
    
    def test_random_forest_training(self, sample_data):
        """Test l'entraînement du Random Forest"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Utilise une grille réduite pour les tests
        simple_param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [5, 10]
        }
        
        trainer = ModelTrainer()
        metrics = trainer.train_random_forest(
            X_train, X_test, y_train, y_test, 
            run_name="test_rf",
            param_grid=simple_param_grid
        )
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mae' in metrics
        assert metrics['r2'] > 0

class TestModelEvaluator:
    """Tests pour l'évaluation des modèles"""
    
    @pytest.fixture
    def sample_model_and_data(self):
        """Crée un modèle simple et des données pour les tests"""
        from sklearn.linear_model import LinearRegression
        
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(y, name='target')
        
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    def test_model_evaluation(self, sample_model_and_data):
        """Test l'évaluation d'un modèle"""
        model, X_test, y_test = sample_model_and_data
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Vérifications de cohérence
        assert metrics['rmse'] == np.sqrt(metrics['mse'])
        assert metrics['mse'] > 0
        assert metrics['mae'] > 0

if __name__ == "__main__":
    pytest.main([__file__])

# ================================================================
# Script principal : main.py (à mettre à la racine)
"""
Script principal pour lancer l'entraînement des modèles
"""

import argparse
import logging
from pathlib import Path

from src.data.data_loader import load_housing_data, create_train_test_split
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.utils.config import LOG_FORMAT, LOG_LEVEL

def setup_logging():
    """Configure le logging"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Entraînement des modèles de prix immobilier")
    parser.add_argument("--model", choices=["linear", "rf", "all"], default="all",
                       help="Type de modèle à entraîner")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Taille du test set")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🏠 Début de l'entraînement des modèles de prix immobilier")
    
    # Chargement des données
    X, y = load_housing_data()
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=args.test_size
    )
    
    # Entraînement des modèles
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    results = {}
    
    if args.model in ["linear", "all"]:
        logger.info("Entraînement du modèle de régression linéaire...")
        metrics = trainer.train_linear_regression(X_train, X_test, y_train, y_test)
        results['Linear Regression'] = metrics
        
        # Graphiques d'évaluation
        evaluator.plot_predictions(trainer.model, X_test, y_test, "linear_predictions.png")
    
    if args.model in ["rf", "all"]:
        logger.info("Entraînement du Random Forest...")
        metrics = trainer.train_random_forest(X_train, X_test, y_train, y_test)
        results['Random Forest'] = metrics
        
        # Graphiques d'évaluation
        evaluator.plot_predictions(trainer.model, X_test, y_test, "rf_predictions.png")
        evaluator.plot_residuals(trainer.model, X_test, y_test, "rf_residuals.png")
    
    # Comparaison des résultats
    if len(results) > 1:
        logger.info("\n📊 Comparaison des modèles:")
        comparison_df = pd.DataFrame(results).T
        print(comparison_df.round(4))
        
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        logger.info(f"🏆 Meilleur modèle: {best_model} (RMSE: {results[best_model]['rmse']:.4f})")
    
    logger.info("✅ Entraînement terminé! Consultez MLflow UI pour voir les résultats.")
    logger.info("   Commande: mlflow ui")

if __name__ == "__main__":
    main()
