from caffe2onnx.graph import CaffeToOnnxNode


def getPReluOutShape(input_shape):
    output_shape = input_shape
    return output_shape


def createPRelu(layer, nodename, inname, outname, input_shape):
    output_shape = getPReluOutShape(input_shape)
    node = CaffeToOnnxNode(layer, nodename, "PRelu", inname, outname, input_shape, output_shape)
    return node
