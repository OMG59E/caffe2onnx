import copy

from caffe2onnx.graph import CaffeToOnnxNode


def get_reduction_caffe_param(layer):
    return {
        "operation": layer.reduction_param.operation,
        "axis": layer.reduction_param.axis,
        "coeff": layer.reduction_param.coeff,
        "domain": "ax.caffe2onnx",
    }


def shape_inference_reduction(input_shape, attrs):
    axis = attrs["axis"]
    output_shape = copy.deepcopy(input_shape[0])
    dim = axis + len(input_shape[0]) if axis < 0 else axis
    del output_shape[dim]
    output_shape = [1] if len(input_shape[0]) == 1 else output_shape
    return [output_shape]


def create_reduction_node(layer, node_name, input_name, output_name, input_shape):
    caffe_params = get_reduction_caffe_param(layer)
    onnx_operation = "Reduction"
    output_shape = shape_inference_reduction(input_shape, caffe_params)
    node = CaffeToOnnxNode(
        layer,
        node_name,
        onnx_operation,
        input_name,
        output_name,
        input_shape,
        output_shape,
        caffe_params,
    )
    return node
