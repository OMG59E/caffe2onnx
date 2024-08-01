from caffe2onnx.graph import CaffeToOnnxNode


def getIdentityOutShape(input_shape):
    output_shape = input_shape
    return output_shape


# 构建节点
def createIdentity(layer, nodename, inname, outname, input_shape):
    output_shape = getIdentityOutShape(input_shape)
    node = CaffeToOnnxNode(
        layer,
        nodename,
        "Identity",
        inname,
        outname,
        input_shape,
        output_shape,
    )
    return node
