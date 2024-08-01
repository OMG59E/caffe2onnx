import numpy as np

from caffe2onnx.graph import CaffeToOnnxNode


# 获取超参数
def get_upsample_attri(layer):
    # scale = layer.upsample_param.scale
    # scales = [1.0,1.0,scale,scale]
    # dict = {"scales":scales,"mode":"nearest"}#Upsample将scales放入参数里面了
    # dict = {"width_scale": scale,"height_scale":scale, "mode": "nearest"}#在OpenVINO读onnx的时候要求用width_scale和height_scale
    try:
        scale = layer.upsample_param.scale
    except:
        try:
            scale = layer.upsample_param.stride
        except:
            raise NotImplementedError("Only support 'scale' or 'stride' in 'Upsample'!")
    scales = [1.0, 1.0, scale, scale]

    attributes = {"mode": "linear", "scales": scales}

    return attributes


def get_upsample_outputshape(input_shape, layer):
    scales = get_upsample_attri(layer)["scales"]
    output_shape = [np.multiply(np.array(scales, dtype=np.int), np.array(input_shape[0])).tolist()]
    return output_shape


def create_upsample_node(layer, node_name, input_name, output_name, input_shape):
    attributes = get_upsample_attri(layer)
    output_shape = get_upsample_outputshape(input_shape, layer)

    # print(output_shape)
    node = CaffeToOnnxNode(
        layer,
        node_name,
        "Upsample",
        input_name,
        output_name,
        input_shape,
        output_shape,
        attributes,
    )
    return node
