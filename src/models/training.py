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
