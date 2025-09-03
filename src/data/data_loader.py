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
            for param, value
