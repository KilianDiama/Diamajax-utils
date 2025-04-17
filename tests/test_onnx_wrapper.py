import pytest
import numpy as np
import onnxruntime as ort

from diamajax_utils.onnx_wrapper import ONNXModelWrapper

# Stub pour InferenceSession
class DummyMeta:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

class DummySession:
    def __init__(self, model_path, providers):
        self._inputs = [DummyMeta("input", [1, 3, 224, 224])]
        self._outputs = [DummyMeta("output", [1, 1000])]
    def get_inputs(self):
        return self._inputs
    def get_outputs(self):
        return self._outputs
    def run(self, *args, **kwargs):
        return [np.zeros((1, 1000), dtype=np.float32)]

@pytest.fixture(autouse=True)
def patch_onnx(monkeypatch):
    monkeypatch.setattr(ort, "get_device", lambda: "CPU")
    monkeypatch.setattr(ort, "InferenceSession", DummySession)

def test_get_model_metadata_and_validate(tmp_path):
    # on crée un fichier dummy.onnx pour l'initialisation
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"")
    wrapper = ONNXModelWrapper(str(model_path), device_preference="auto")

    meta = wrapper.get_model_metadata()
    assert "inputs" in meta and "outputs" in meta
    assert meta["inputs"]["input"] == [1, 3, 224, 224]

    # valid input
    valid = np.zeros((1, 3, 224, 224), dtype=np.float32)
    assert wrapper.validate_input({"input": valid})
    # invalid key
    assert not wrapper.validate_input({"bad": valid})

def test_predict_and_warmup(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"")
    wrapper = ONNXModelWrapper(str(model_path), device_preference="cpu")

    inp = np.zeros((1, 3, 224, 224), dtype=np.float32)
    out = wrapper.predict({"input": inp})
    assert isinstance(out, list) and isinstance(out[0], np.ndarray)

    # warmup doit juste appeler predict sans erreur
    wrapper.warmup({"input": inp})
