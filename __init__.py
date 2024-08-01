from caffe2onnx.cli import main

try:
    from caffe2onnx.version import __version__  # noqa: F401
except ImportError:
    __version__ = "0+unknown"
