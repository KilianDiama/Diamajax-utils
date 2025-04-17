import numpy as np
import pytest
from diamajax_utils.onnx_wrapper import ONNXModelWrapper

@pytest.fixture
def dummy_model(tmp_path):
    # Crée un fichier vide pour stub ONNX
    path = tmp_path / "model.onnx"
    path.write_bytes(b"")
    return str(path)

def test_import_and_methods_exist():
    assert hasattr(ONNXModelWrapper, "__init__")
    assert hasattr(ONNXModelWrapper, "predict")
    assert hasattr(ONNXModelWrapper, "warmup")

def test_predict_returns_dict(dummy_model):
    wrapper = ONNXModelWrapper(model_path=dummy_model, device_preference="cpu")
    # On passe un dict conforme, même si le stub renvoie vide/dépendra de ton code
    fake = np.zeros((1, 3, 224, 224), dtype=np.float32)
    out = wrapper.predict({"input": fake})
    assert isinstance(out, dict)
