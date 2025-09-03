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

