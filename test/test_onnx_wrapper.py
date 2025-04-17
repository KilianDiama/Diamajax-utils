import numpy as np
import pytest
from diamajax_utils.onnx_wrapper import ONNXModelWrapper

@pytest.fixture
def dummy_model_path(tmp_path):
    # Génère un stub ONNX vide pour ne pas planter l'import
    path = tmp_path / "model.onnx"
    path.write_bytes(b"")
    return str(path)

def test_importable_and_has_methods():
    assert hasattr(ONNXModelWrapper, "__init__")
    assert hasattr(ONNXModelWrapper, "predict")
    assert hasattr(ONNXModelWrapper, "warmup")

def test_predict_returns_dict(dummy_model_path):
    wrapper = ONNXModelWrapper(model_path=dummy_model_path, device_preference="cpu")
    # stub d’entrée compatible (1×3×224×224)
    fake_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
    out = wrapper.predict({"input": fake_input})
    assert isinstance(out, dict)
