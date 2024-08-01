from caffe2onnx.graph import CaffeToOnnxNode


def get_exp_attributes(layer):
    base = layer.exp_param.base
    scale = layer.exp_param.scale
    shift = layer.exp_param.shift
    return base, scale, shift


def get_exp_output_shape(input_shape):
    output_shape = input_shape
    return output_shape


def create_exp_node(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_exp_output_shape(input_shape)

    node = CaffeToOnnxNode(layer, node_name, "Exp", input_name, output_name, input_shape, output_shape)

    return node
