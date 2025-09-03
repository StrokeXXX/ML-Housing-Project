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

