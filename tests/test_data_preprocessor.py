import numpy as np
import pytest
from diamajax_utils.data_preprocessor import DataPreprocessor

def test_preprocess_invalid_input():
    dp = DataPreprocessor()
    with pytest.raises(ValueError):
        dp.preprocess([1, 2, 3])  # pas un tableau 2D

def test_preprocess_standardize_and_normalize():
    # Données simples 2×2
    data = [[0, 10], [5, 20]]
    dp = DataPreprocessor(normalize=True, standardize=True)
    out = dp.preprocess(data)

    # Après standardisation : mean=0, std=1
    # Puis normalisation : min→0 et max→1
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert np.allclose(out.min(axis=0), 0.0)
    assert np.allclose(out.max(axis=0), 1.0)
