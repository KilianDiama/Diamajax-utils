# tests/test_onnx_wrapper.py
import pytest
from diamajax_utils.onnx_wrapper import ONNXModelWrapper

def test_wrapper_import():
    wrapper = ONNXModelWrapper
    assert hasattr(wrapper, "predict")
