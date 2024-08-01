from caffe2onnx.graph import CaffeToOnnxNode


def getOutShape(input_shape):
    # 获取output_shape
    return input_shape


# 构建节点
def createSoftplus(layer, nodename, inname, outname, input_shape):
    output_shape = getOutShape(input_shape)

    node = CaffeToOnnxNode(layer, nodename, "Softplus", inname, outname, input_shape, output_shape)

    return node