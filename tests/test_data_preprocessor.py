import pytest
from diamajax_utils.data_preprocessor import DataPreprocessor

def test_preprocess_invalid_input():
    dp = DataPreprocessor()
    # Toute liste non‑2D doit déclencher une ValueError
    with pytest.raises(ValueError):
        dp.preprocess([1, 2, 3])  # pas un tableau 2D
