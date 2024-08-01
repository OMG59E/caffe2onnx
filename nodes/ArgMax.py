import copy

from caffe2onnx.graph import CaffeToOnnxNode


def get_argmax_attributes(layer):
    axis = layer.argmax_param.axis
    keepdims = 1
    attributes = {"axis": axis, "keepdims": keepdims}
    return attributes


def shape_inference(input_shape, attributes):
    output_shape = copy.deepcopy(input_shape[0])

    keepdims = attributes["keepdims"]
    axis = attributes["axis"]

    if keepdims == 1:
        output_shape[axis] = 1
    else:
        del output_shape[axis]

    return [output_shape]


def create_argmax_node(layer, node_name, input_name, output_name, input_shape):
    attrs = get_argmax_attributes(layer)
    output_shape = shape_inference(input_shape, attrs)

    node = CaffeToOnnxNode(layer, node_name, "ArgMax", input_name, output_name, input_shape, output_shape, attrs)

    return node
