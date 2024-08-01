# Copyright (c) 2023, Axera Tech. All rights reserved.

from onnx import ModelProto, shape_inference
from onnxsim import simplify

import caffe2onnx.graphsurgeon as gs


def toposort_and_simlify(onnx_model: ModelProto) -> ModelProto:
    """
    Cleanup, toposort and simlify.
    """
    ir_version = onnx_model.ir_version
    graph: gs.Graph = gs.import_onnx(onnx_model)
    print("Simplifier: Cleanup and toposort...")
    graph = graph.cleanup().toposort()

    onnx_graph = gs.export_onnx(graph)
    onnx_graph.ir_version = ir_version
    onnx_graph = shape_inference.infer_shapes(onnx_graph)

    print("Simplifier: Doing onnxsim...")
    model_simp, check_ok = simplify(onnx_graph)
    if check_ok:
        print("Simplifier: onnxsim succeeded :)")
        return model_simp

    print("Simplifier: onnxsim failed :(")
    return onnx_graph


def convert_caffe_to_onnx(
    caffe_graph_path: str,
    caffe_params_path: str,
    *,
    onnx_name: str = "model",
    opset_version: int = 16,
    skip_onnxsim: bool = False,
    batch_size: int = 1
):
    from caffe2onnx.caffe_converter import CaffeToOnnx
    from caffe2onnx.utils.load_caffe_model import load_caffe_model

    print("Converter: Loading caffe graph and params...")
    graph, params = load_caffe_model(caffe_graph_path, caffe_params_path)
    print("Converter: Starting model conversion...")
    caffe_convert = CaffeToOnnx(graph, params, onnx_name, caffe_graph_path, batch_size)
    print("Converter: Creating onnx model...")
    onnx_model = caffe_convert.createOnnxModel(opset_version=opset_version)

    if not skip_onnxsim:
        onnx_model = toposort_and_simlify(onnx_model)

    return onnx_model
