import pytest
import pandas as pd
from src.data.data_loader import load_housing_data, create_train_test_split

def test_load_housing_data():
    """Test le chargement des données"""
    X, y = load_housing_data()
    
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 8  # 8 features dans California Housing
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_test_split():
    """Test la création du split train/test"""
    X, y = load_housing_data()
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(X)
