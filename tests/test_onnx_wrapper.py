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
        # Inputs et outputs factices
        self._inputs = [DummyMeta("input", [1, 3, 224, 224])]
        self._outputs = [DummyMeta("output", [1, 1000])]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, input_data):
        # Renvoie un array compatible avec output shape
        return [np.zeros((1, 1000), dtype=np.float32)]

# Stub qui échoue en inference
class ErrorSession(DummySession):
    def run(self, *args, **kwargs):
        raise RuntimeError("Inference failed")

def setup_session(monkeypatch, session_cls=DummySession, device="CPU"):
    # Force le device et la session ONNX
    monkeypatch.setattr(ort, "get_device", lambda: device)
    monkeypatch.setattr(ort, "InferenceSession", session_cls)

def test_select_device(monkeypatch):
    setup_session(monkeypatch, session_cls=DummySession, device="GPU")
    o = ONNXModelWrapper.__new__(ONNXModelWrapper)
    # GPU disponible → doit choisir CUDAExecutionProvider
    assert o._select_device("auto") == "CUDAExecutionProvider"
    # Précision CPU
    assert o._select_device("cpu") == "CPUExecutionProvider"
    # Forcer GPU
    assert o._select_device("gpu") == "CUDAExecutionProvider"

def test_init_and_get_model_metadata(monkeypatch, tmp_path):
    setup_session(monkeypatch, session_cls=DummySession, device="CPU")
    model_path = str(tmp_path / "model.onnx")
    # Crée un fichier stub (inutile pour DummySession mais bon)
    (tmp_path / "model.onnx").write_bytes(b"")
    wrapper = ONNXModelWrapper(model_path, device_preference="auto")

    md = wrapper.get_model_metadata()
    assert set(md.keys()) == {"inputs", "outputs"}
    assert md["inputs"] == {"input": [1, 3, 224, 224]}
    assert md["outputs"] == {"output": [1, 1000]}

def test_validate_input(monkeypatch):
    setup_session(monkeypatch)
    wrapper = ONNXModelWrapper.__new__(ONNXModelWrapper)
    # On injecte manuellement métadonnées pour tester
    wrapper.input_metadata = {"in": [2, 2]}
    # Forme correcte
    valid = np.zeros((2, 2), dtype=np.float32)
    assert wrapper.validate_input({"in": valid})
    # Clé invalide
    assert not wrapper.validate_input({"bad": valid})
    # Mauvaise forme
    bad = np.zeros((2, 2, 2), dtype=np.float32)
    assert not wrapper.validate_input({"in": bad})

def test_predict_success(monkeypatch):
    setup_session(monkeypatch, session_cls=DummySession)
    # Initialisation normale
    wrapper = ONNXModelWrapper.__new__(ONNXModelWrapper)
    setup_session(monkeypatch)
    # On contourne __init__ pour injecter attributs
    wrapper.validate_input = lambda x: True
    wrapper.session = DummySession("p", ["CPU"])
    # Doit renvoyer une liste
    inp = np.zeros((1, 3, 224, 224), dtype=np.float32)
    result = wrapper.predict({"input": inp})
    assert isinstance(result, list)
    assert result and isinstance(result[0], np.ndarray)

def test_predict_invalid_input_raises(monkeypatch):
    setup_session(monkeypatch, session_cls=DummySession)
    wrapper = ONNXModelWrapper.__new__(ONNXModelWrapper)
    wrapper.validate_input = lambda x: False
    with pytest.raises(ValueError):
        wrapper.predict({"input": None})

def test_predict_error_returns_empty(monkeypatch):
    setup_session(monkeypatch, session_cls=ErrorSession)
    wrapper = ONNXModelWrapper.__new__(ONNXModelWrapper)
    # Injecte validate_input OK
    wrapper.validate_input = lambda x: True
    wrapper.session = ErrorSession("p", ["CPU"])
    out = wrapper.predict({"input": np.zeros((1,3,224,224), dtype=np.float32)})
    assert out == []

def test_warmup_catches(monkeypatch):
    setup_session(monkeypatch, session_cls=ErrorSession)
    wrapper = ONNXModelWrapper.__new__(ONNXModelWrapper)
    wrapper.predict = lambda x: (_ for _ in ()).throw(RuntimeError("oops"))
    # Ne doit pas remonter l'exception
    wrapper.warmup({"input": np.zeros((1,3,224,224), dtype=np.float32)})

