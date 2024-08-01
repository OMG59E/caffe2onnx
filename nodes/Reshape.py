from functools import reduce
from operator import mul

from typing import List

from caffe2onnx.graph import CaffeToOnnxNode


# 计算输出维度
def getReshapeOutShape(layer, input_shape: List) -> List:
    if layer.type == "InnerProduct":
        dims = input_shape[0]
        total_dims = len(dims)

        axis = (layer.inner_product_param.axis + total_dims) % total_dims

        prod0 = 1
        prod1 = 1
        for i in range(axis):
            prod0 *= dims[i]
        for i in range(axis, total_dims):
            prod1 *= dims[i]

        output_shape = [prod0, prod1]
        return [output_shape]

    elif layer.type == "ShuffleChannel":
        # change [N, C, H, W] -> [N, G, C', H, W] tensor
        group = layer.shuffle_channel_param.group
        n, g, c, h, w = (
            input_shape[0][0],
            group,
            int(input_shape[0][1] / group),
            input_shape[0][2],
            input_shape[0][3],
        )
        out_shape = [[n, g, c, h, w]]
        return out_shape

    elif layer.type == "DeReshape":
        n, c, h, w = (
            input_shape[0][0],
            input_shape[0][1] * input_shape[0][2],
            input_shape[0][3],
            input_shape[0][4],
        )
        out_shape = [[n, c, h, w]]
        return out_shape

    elif layer.type == "Flatten":
        num_axis = len(input_shape[0])
        start_axis = layer.flatten_param.axis
        end_axis = layer.flatten_param.end_axis

        end_axis = (end_axis + num_axis) % num_axis + 1

        shape_prod = 1
        for i in range(start_axis, end_axis):
            shape_prod *= input_shape[0][i]
        output_shape = [input_shape[0][0:start_axis] + [shape_prod] + input_shape[0][end_axis:]]

        return output_shape

    elif layer.type == "Scale":
        return input_shape

    elif layer.type == "Reshape":
        shape = input_shape[0]
        re_shape = layer.reshape_param.shape.dim
        new_shape_list = []
        for j in range(len(re_shape)):
            if re_shape[j] == 0:
                # if value = 0 ; then use original
                new_shape_list.append(shape[j])
            else:
                new_shape_list.append(re_shape[j])
        if -1 in new_shape_list:
            index = new_shape_list.index(-1)
            if index == 0:
                prod = reduce(mul, new_shape_list[1:], 1)
            elif index == (len(new_shape_list) - 1):
                prod = reduce(mul, new_shape_list[0:index])
            else:
                prod = reduce(mul, new_shape_list[0:index]) * reduce(mul, new_shape_list[index + 1 :], 1)
            new_shape_list[index] = int(reduce(mul, shape, 1) / prod)
        output_shape = [new_shape_list]
        return output_shape


def get_reshape_param(layer, input_shape: List[int]) -> List[int]:
    re_shape = layer.reshape_param.shape.dim
    return re_shape


# 构建节点
def createReshape(layer, node_name, input_name, output_name, input_shape, output_shape={}):
    # 获取output_shape
    if layer.type == "Scale" and output_shape != {}:
        node = CaffeToOnnxNode(
            layer,
            node_name,
            "Reshape",
            input_name,
            output_name,
            input_shape,
            output_shape,
        )
        return node
    elif layer.type == "LSTM" and output_shape != {}:
        node = CaffeToOnnxNode(
            layer,
            node_name,
            "Reshape",
            input_name,
            output_name,
            input_shape,
            output_shape,
        )
        return node
    elif output_shape == {}:
        output_shape = getReshapeOutShape(layer, input_shape)

    # 构建node
    node = CaffeToOnnxNode(layer, node_name, "Reshape", input_name, output_name, input_shape, output_shape)
    return node
