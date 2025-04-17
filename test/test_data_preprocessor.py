import numpy as np
import pandas as pd
import pytest
from diamajax_utils.data_preprocessor import DataPreprocessor

def test_preprocess_list_input():
    data = [[1, 2], [3, 4], [5, 6]]
    dp = DataPreprocessor(normalize=True, standardize=True)
    out = dp.preprocess(data)
    # Après standardisation puis normalisation, chaque colonne ∈ [0,1]
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 2)
    assert np.all(out >= 0) and np.all(out <= 1)

def test_preprocess_invalid_input():
    dp = DataPreprocessor()
    with pytest.raises(ValueError):
        dp.preprocess([1, 2, 3])  # pas un 2D
