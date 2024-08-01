from caffe2onnx.graph import CaffeToOnnxNode


def getDropoutOutShape(input_shape):
    # 计算输出维度output_shape
    output_shape = input_shape  # 与输入维度一样
    return output_shape


# 构建节点
def createDropout(layer, nodename, inname, outname, input_shape):
    output_shape = getDropoutOutShape(input_shape)
    # 构建node
    node = CaffeToOnnxNode(
        layer,
        nodename,
        "Dropout",
        inname,
        outname,
        input_shape,
        output_shape,
    )
    return node
