import pytest
from src.data.data_loader import load_housing_data

def test_load_housing_data():
    X, y = load_housing_data()
    assert X.shape[0] > 0
    assert X.shape[1] == 8
    assert len(y) == X.shape[0]
