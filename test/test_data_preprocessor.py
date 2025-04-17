import numpy as np
import pandas as pd
import pytest
from diamajax_utils.data_preprocessor import DataPreprocessor

@pytest.fixture
def df():
    # DataFrame simple
    return pd.DataFrame({
        "a": [1, 2, None],
        "b": ["x", None, "z"]
    })

def test_clean_and_encode(df):
    proc = DataPreprocessor()
    cleaned = proc.clean_missing(df)
    # Plus de NaN en a ou b
    assert not cleaned.isnull().any().any()

    encoded = proc.encode_categorical(cleaned, columns=["b"])
    # colonne b transformée
    assert "b" in encoded.columns
    assert encoded["b"].dtype == int or encoded["b"].dtype == np.int64
