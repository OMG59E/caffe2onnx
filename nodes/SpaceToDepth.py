from caffe2onnx.graph import CaffeToOnnxNode


def getAttri(layer):
    attr = {"blocksize": layer.reorg_param.stride}
    return attr


# 计算输出维度
def get_output_shape(input_shape, attr):
    blocksize = attr["blocksize"]

    output_shape = [
        [
            input_shape[0][0],
            input_shape[0][1] * blocksize * blocksize,
            int(input_shape[0][2] / blocksize),
            int(input_shape[0][3] / blocksize),
        ]
    ]

    return output_shape


# 构建节点
def createSpaceToDepth(layer, nodename, inname, outname, input_shape):
    attr = getAttri(layer)
    output_shape = get_output_shape(input_shape, attr)

    node = CaffeToOnnxNode(
        layer,
        nodename,
        "SpaceToDepth",
        inname,
        outname,
        input_shape,
        output_shape,
        dict=attr,
    )

    return node
