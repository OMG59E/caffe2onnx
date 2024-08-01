# Tencent is pleased to support the open source community by making TNN available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from copy import deepcopy

from caffe2onnx.graph import CaffeToOnnxNode


def analyzeLayer(layer, input_shape, output_num):
    if len(layer.slice_param.slice_point) > 0:
        # 获取到 slice_point
        axis = layer.slice_param.axis
        starts = [0]
        axes = [axis]
        for step in layer.slice_param.slice_point:
            starts.append(step)
            axes.append(axis)
        # 获取需要进行操作的轴
        ends = []
        for step in layer.slice_param.slice_point:
            ends.append(step)
        # 这个地方搞了一个小 trick, 使用输入的 axis 作为最后一个
        ends.append(input_shape[0][axis])
    else:
        axis = layer.slice_param.axis
        assert output_num is not None and input_shape[0][axis] % output_num == 0
        interval = input_shape[0][axis] // output_num

        starts, ends, axes = [], [], []
        for step in range(output_num):
            starts.append(step * interval)
            ends.append((step + 1) * interval)
            axes.append(axis)

    return starts, ends, axes


# 计算输出维度
# def getSliceOutShape(layer, input_shape, output_name):
#     # TODO:
#     steps = []
#     for step in layer.slice_param.slice_point:
#         steps.append(step)
#     # slice point
#     assert len(steps) == len(output_name) - 1
#     # 轴
#     axis = layer.concat_param.axis
#     start = 0
#     n, c, w, h = input_shape[0][0], 0, input_shape[0][2], input_shape[0][3]
#     # 计算总体的值
#     output_shape = [[]]
#     sum = input_shape[0][1]
#     if (axis == 1):
#         for step in steps:
#             # update start
#             c = step - start
#             output_shape.append([n, c, w, h])
#             start = step
#     output_shape.append([n, sum - start, w, h])
#     return output_shape[1:]


# def getSliceAttri(layer, start, end, axes):
#     attributs = {
#         'starts': [start],
#         'ends': [end],
#         'axes': [axes],
#     }
#     return attributs


def getSliceOutShape(input_shape, start, end, axis, step):
    if not step:
        step = [1 for i in range(len(start))]
    output_shape = [deepcopy(input_shape[0])]
    for i in range(len(start)):
        # remap start to [0, input_shape[0][axis[i]]]
        if start[i] < -input_shape[0][axis[i]]:
            if step[i] > 0:
                start[i] = 0
            else:
                output_shape[0][axis[i]] = 0
                continue
        elif start[i] < 0:
            start[i] += input_shape[0][axis[i]]
        elif start[i] >= input_shape[0][axis[i]]:
            if step[i] > 0:
                output_shape[0][axis[i]] = 0
                continue
            else:
                start[i] = input_shape[0][axis[i]] - 1
        # remap end to [-1, input_shape[0][axis[i]]]
        if end[i] < -input_shape[0][axis[i]]:
            if step[i] > 0:
                output_shape[0][axis[i]] = 0
                continue
            else:
                end[i] = -1
        elif end[i] < 0:
            end[i] += input_shape[0][axis[i]]
        elif end[i] > input_shape[0][axis[i]]:
            if step[i] > 0:
                end[i] = input_shape[0][axis[i]]
            else:
                output_shape[0][axis[i]] = 0
                continue

        output_shape[0][axis[i]] = (end[i] - start[i]) // step[i]
    return output_shape


# 构建节点
def createSlice(layer, node_name, input_name, output_name, input_shape, start, end, axis, step=None):
    output_shape = getSliceOutShape(input_shape, start, end, axis, step)

    node = CaffeToOnnxNode(
        layer,
        node_name,
        "Slice",
        input_name,
        output_name,
        input_shape,
        output_shape,
        Flag=True,
    )
    return node
