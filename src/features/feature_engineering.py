"""
Feature engineering pour le dataset California Housing
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Classe pour l'ingénierie des features"""
    
    def __init__(self, add_rooms_per_household=True, add_population_per_household=True,
                 add_bedrooms_per_room=True):
        self.add_rooms_per_household = add_rooms_per_household
        self.add_population_per_household = add_population_per_household
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        if self.add_rooms_per_household:
            X["rooms_per_household"] = X["AveRooms"] / X["AveOccup"]
        
        if self.add_population_per_household:
            X["population_per_household"] = X["Population"] / X["AveOccup"]
        
        if self.add_bedrooms_per_room:
            X["bedrooms_per_room"] = X["AveBedrms"] / X["AveRooms"]
        
        # Interaction features
        X["income_per_population"] = X["MedInc"] / X["Population"]
        
        logger.info(f"Features ajoutées. Nouvelle shape: {X.shape}")
        return X
