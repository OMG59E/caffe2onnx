from caffe2onnx.graph import CaffeToOnnxNode


def get_constant_output_shape(value):
    return [value.dims]


# 构建节点
def createConstant(layer, node_name, input_name, output_name, value, input_shape=[]):
    # 构建node
    attributes = {"value": value}
    output_shape = get_constant_output_shape(value)

    node = CaffeToOnnxNode(
        layer,
        node_name,
        "Constant",
        input_name,
        output_name,
        input_shape,
        output_shape,
        attributes,
    )
    return node
