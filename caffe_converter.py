import copy
import math

import numpy as np

import onnx
from onnx import defs, helper

from caffe2onnx import nodes, op_layer_info
from caffe2onnx.graph import CaffeToOnnxGraph


class CaffeToOnnx:
    def __init__(self, net, model, onnx_path: str, proto_path, batch_size=1):
        # 初始化一个c2oGraph对象
        self.onnxmodel = CaffeToOnnxGraph(onnx_path)
        # 网络和参数
        self.netLayerCaffe = self.GetNetLayerCaffe(net)
        self.netModelCaffe = self.GetNetModelCaffe(model)
        self.proto_path = proto_path

        # 模型的输入名和输入维度
        self.model_input_name = []
        self.model_input_shape = []

        # 节点列表
        self.onnxNodeList = []

        # 获取层列表
        LayerList = self.AddInputsTVIAndGetLayerList(net, batch_size)
        self.GenerateOnnxNodeList(LayerList)
        self.DeleteUnusedInput()
        self.AddOutputsTVIAndValueInfo()

        try:
            self.LayerParamFreqStatistics(LayerList)
        except Exception as e:
            print(f"Warning: {e}")

    # 获取网络层
    def GetNetLayerCaffe(self, net):
        if len(net.layer) == 0 and len(net.layers) != 0:
            return net.layers
        if len(net.layer) != 0 and len(net.layers) == 0:
            return net.layer
        raise ValueError(f"No layer found in prototxt file: '{net}'!")

    # 获取参数层
    def GetNetModelCaffe(self, model):
        if len(model.layer) == 0 and len(model.layers) != 0:
            return model.layers
        if len(model.layer) != 0 and len(model.layers) == 0:
            return model.layer
        raise ValueError(f"No layer found in caffemodel file: '{model}'!")

    # 将模型输入信息添加到Inputs中并获取后续层列表
    def AddInputsTVIAndGetLayerList(self, net, batch_size=1):
        # 如果第一个layer的类型为Input,且没有net.input存在
        if net.input == [] and self.netLayerCaffe[0].type == "Input":
            layer_list = []
            # 考虑到整个网络会有多输入情况
            for lay in self.netLayerCaffe:
                if lay.type == "Input":
                    # Input layer may have multiple tops
                    for i in range(len(lay.top)):
                        in_tvi = helper.make_tensor_value_info(
                            lay.top[i],
                            onnx.TensorProto.FLOAT,
                            lay.input_param.shape[i].dim,
                        )
                        self.model_input_name.append(lay.top[i])
                        self.model_input_shape.append(lay.input_param.shape[i].dim)
                        self.onnxmodel.addInputsTVI(in_tvi)
                else:
                    layer_list.append(lay)
            return layer_list

        # 如果存在net.input
        if net.input != []:
            for i in range(len(net.input)):
                if len(net.input_dim) > 0:
                    input_dims = net.input_dim
                else:
                    input_dims = net.input_shape[i].dim
                input_dims[0] = batch_size
                in_tvi = helper.make_tensor_value_info(net.input[i], onnx.TensorProto.FLOAT, input_dims)
                self.model_input_name.append(net.input[i])
                self.model_input_shape.append(input_dims)
                self.onnxmodel.addInputsTVI(in_tvi)
            return self.netLayerCaffe

        # 以上情况都不是,则该caffe模型没有输入,存在问题
        raise ValueError("The model has no input!")

    # 得到layer的参数shape
    def GetParamsShapeAndData(self, layer):
        ParamShape = []
        ParamData = []
        # 根据这个layer名找出对应的caffemodel中的参数
        for model_layer in self.netModelCaffe:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                ParamShape = self.get_param_shape(Params)
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN":
                    if len(ParamShape) == 3:
                        # 如果是bn层，则不用最后一层的滑动系数
                        ParamShape = ParamShape[:-1]
                        ParamData = ParamData[:-1]
                    elif len(ParamShape) == 2 and len(ParamShape[0]) != 1:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = ParamData
        return ParamShape, ParamData

    def get_param_shape(self, params):
        shapes = []
        for p in params:
            if p.shape.dim != []:
                shape = p.shape.dim
                shapes.append(shape)
            elif p.num == 0 or p.channels == 0 or p.height == 0 or p.width == 0:
                shapes.append([len(p.data)])
            else:
                shape = [p.num, p.channels, p.height, p.width]
                shapes.append(shape)
        return shapes

    # 将参数添加到Inputs中,并生成tensor存储数据
    def AddInputsTVIFromParams(self, layer, ParamName, ParamType):
        ParamShape = []
        ParamData = []
        # 根据这个layer名找出对应的caffemodel中的参数
        for model_layer in self.netModelCaffe:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                # ParamShape = [p.shape.dim for p in Params]
                ParamShape = self.get_param_shape(Params)
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN":
                    if len(ParamShape) == 3:
                        # 如果是bn层，params为[mean, var, s]，则需要把mean和var除以滑动系数s
                        ParamShape = ParamShape[:-1]
                        scale_factor = 0 if Params[-1].data[0] == 0 else 1 / Params[-1].data[0]
                        ParamData = [[q * scale_factor for q in p.data] for i, p in enumerate(Params[:-1])]
                    elif len(ParamShape) == 2 and len(ParamShape[0]) == 4:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = [[q * 1.0 for q in p.data] for i, p in enumerate(Params)]
                if layer.type == "Reshape":
                    ParamShape = [[len(model_layer.reshape_param.shape.dim)]]
                    ParamData = [model_layer.reshape_param.shape.dim]
                if layer.type == "Convolution" or layer.type == "ConvolutionDepthwise":
                    if len(ParamShape) == 2:
                        ParamShape[1] = [ParamShape[0][0]]
                if layer.type == "InnerProduct":
                    if len(ParamShape[0]) > 2:
                        ParamShape[0] = [ParamShape[0][2], ParamShape[0][3]]
                    if len(ParamShape) == 2:
                        if len(ParamShape[1]) > 2:
                            ParamShape[1] = [ParamShape[1][2], ParamShape[1][3]]
                if layer.type == "Normalize":
                    if len(ParamShape) == 1:
                        ParamShape[0] = [1, ParamShape[0][0], 1, 1]
                if layer.type == "LSTM":
                    # TODO: BiLSTM
                    is_bilstm = False
                    if is_bilstm:
                        raise Exception("Not implemented!")
                    else:
                        """
                        caffe的参数为:
                           W: [4*hidden_size, input_size], ifoc
                           R: [4*hidden_size, hidden_size], ifoc
                           B: [4*hidden_size], ifoc
                        onnx的参数为:
                           W: [num_directions, 4*hidden_size, input_size], iofc
                           R: [num_directions, 4*hidden_size, hidden_size], iofc
                           B: [num_directions, 8*hidden_size], iofciofc
                        """
                        hidden_size = int(ParamShape[0][0] / 4)
                        input_size = int(ParamShape[0][1])

                        # LSTM_W
                        tmp = ParamData[0][hidden_size * input_size * 2 : hidden_size * input_size * 3].copy()
                        ParamData[0][hidden_size * input_size * 2 : hidden_size * input_size * 3] = ParamData[0][
                            hidden_size * input_size : hidden_size * input_size * 2
                        ]
                        ParamData[0][hidden_size * input_size : hidden_size * input_size * 2] = tmp
                        ParamShape[0].insert(0, 1)

                        # LSTM_B
                        tmp = ParamData[1][hidden_size * 2 : hidden_size * 3].copy()
                        ParamData[1][hidden_size * 2 : hidden_size * 3] = ParamData[1][hidden_size : hidden_size * 2]
                        ParamData[1][hidden_size : hidden_size * 2] = tmp
                        # onnx has w_b and r_b, caffe only have w_b
                        ParamData[1].extend([0.0 for i in range(ParamShape[1][0])])
                        ParamShape[1][0] *= 2
                        ParamShape[1].insert(0, 1)

                        # LSTM_R
                        tmp = ParamData[2][hidden_size * hidden_size * 2 : hidden_size * hidden_size * 3].copy()
                        ParamData[2][hidden_size * hidden_size * 2 : hidden_size * hidden_size * 3] = ParamData[2][
                            hidden_size * hidden_size : hidden_size * hidden_size * 2
                        ]
                        ParamData[2][hidden_size * hidden_size : hidden_size * hidden_size * 2] = tmp
                        ParamShape[2].insert(0, 1)

                # comment it for tvm because tvm use broadcast at prelu layer
                # 个人感觉如果不用 tvm，就不需要使用 Prelu
                # if layer.type == 'PReLU':
                #     ParamShape = [[ParamShape[0][0], 1, 1]]

                break
        # 判断是否有Param
        if ParamShape != []:
            ParamName = ParamName[0 : len(ParamShape)]
            ParamType = ParamType[0 : len(ParamShape)]
            for i in range(len(ParamShape)):
                ParamName[i] = layer.name + ParamName[i]
                p_tvi = helper.make_tensor_value_info(ParamName[i], ParamType[i], ParamShape[i])
                p_t = helper.make_tensor(ParamName[i], ParamType[i], ParamShape[i], ParamData[i])
                self.onnxmodel.addInputsTVI(p_tvi)
                self.onnxmodel.addInitTensor(p_t)
                # print("add param " + ParamName[i] + "input and tensor data")
        if layer.type in ["BatchNorm", "BN", "Scale", "LSTM"]:
            return ParamName, ParamShape
        return ParamName

    # 手动将参数添加到输入信息中,并生成tensor存储数据
    def AddInputsTVIMannul(self, layer, param_names, param_types, param_shapes, param_data):
        node_names = copy.deepcopy(param_names)
        for i in range(len(param_shapes)):
            node_names[i] = layer.name + param_names[i]
            p_tvi = helper.make_tensor_value_info(node_names[i], param_types[i], param_shapes[i])
            p_t = helper.make_tensor(node_names[i], param_types[i], param_shapes[i], param_data[i])
            self.onnxmodel.addInputsTVI(p_tvi)
            self.onnxmodel.addInitTensor(p_t)
        return node_names
        # # 由于 Slice 的 input 情况特殊，所以需要特殊处理
        # if layer.type == 'Slice':
        #     for i in range(len(ParamShape)):
        #         p_tvi = helper.make_tensor_value_info(Param_Name[i], ParamType[i], ParamShape[i])
        #         p_t = helper.make_tensor(Param_Name[i], ParamType[i], ParamShape[i], ParamData[i])
        #         self.onnxmodel.addInputsTVI(p_tvi)
        #         self.onnxmodel.addInitTensor(p_t)
        #     return Param_Name
        # else:
        #     for i in range(len(ParamShape)):
        #         Param_Name[i] = layer.name + ParamName[i]
        #         p_tvi = helper.make_tensor_value_info(Param_Name[i], ParamType[i], ParamShape[i])
        #         p_t = helper.make_tensor(Param_Name[i], ParamType[i], ParamShape[i], ParamData[i])
        #         self.onnxmodel.addInputsTVI(p_tvi)
        #         self.onnxmodel.addInitTensor(p_t)
        #     return Param_Name

    # 获取上一层的输出名(即当前层的输入)
    def GetLastLayerOutNameAndShape(self, layer):
        output_name = []
        output_shape = []
        # flag is True: 模型的输入没有被覆盖
        # flag is False: 模型的输入已经被覆盖
        flag = True

        # 在模型输入里搜寻当前层输入
        for i in range(len(self.model_input_name)):
            for j in range(len(layer.bottom)):
                if self.model_input_name[i] == layer.bottom[j]:
                    output_name.append(self.model_input_name[i])
                    output_shape.append(self.model_input_shape[i])

        # 在模型其余层中搜寻当前层输入
        for i in range(len(layer.bottom)):

            # 因为prototxt中存在top和bottom同名的情况，但是layer.bottom只能对应一个node，所以对每个layer.bottom，找到最末的那个同名节点作为上一层节点
            name = None
            shape = None
            for node in self.onnxNodeList:
                for j in range(len(node.top) if node.node.op_type != "MaxPool" else 1):
                    if layer.bottom[i] == node.top[j]:
                        name = node.outputs_name[j]
                        shape = node.outputs_shape[j]
                    for k in range(len(node.bottom)):
                        if node.top[j] == node.bottom[k]:
                            for w in range(len(self.model_input_name)):
                                if node.top[j] + "_input" == self.model_input_name[w]:
                                    flag = False
            for j in range(len(self.model_input_name)):
                if layer.bottom[i] + "_input" == self.model_input_name[j] and flag:
                    output_name.append(self.model_input_name[j])
                    output_shape.append(self.model_input_shape[j])

            if name:
                output_name.append(name)
                output_shape.append(shape)

        assert output_name, "Failed at layer %s, layer's bottom not detected ..." % (layer.name)
        return output_name, output_shape

    # 获取当前层的输出名，即layername
    def GetCurrentLayerOutName(self, layer):
        # return [layer.name]
        # 考虑有多个输出的情况
        if layer.top == layer.bottom and len(layer.top) == 1:
            if layer.name == layer.top[0]:
                return [layer.name + "_" + layer.name]
            else:
                return [layer.name]
        return [out for out in layer.top]

    def GetLayerOutputShape(self, layer):
        for node in self.onnxNodeList:
            if node.node.name == layer.name:
                return node.outputs_shape[0]

    def LayerParamFreqStatistics(self, Layers):
        def get_param(x, x_h=None, x_w=None, default=None):
            is_empty = lambda x: x is None or not x
            if not is_empty(x_h) and not is_empty(x_w):
                if x_h == x_w:
                    return x_h
                else:
                    return (x_h, x_w)
            else:
                if isinstance(x, (int, float, str)):
                    x = [x]
                else:
                    x = list(x)
                if len(x) == 0:
                    return default
                elif len(x) == 1:
                    if is_empty(x[0]):
                        return default
                    else:
                        return x[0]
                else:
                    if x[1:] == x[:-1]:
                        if is_empty(x[0]):
                            return default
                        else:
                            return x[0]
                    else:
                        return tuple(x)

        print("==========================================")
        print("Parameter frequency statistics:")
        param_set = dict()
        unsupported_type = [
            "ContinuationIndicator",
        ]
        for i in range(len(Layers)):
            if Layers[i].type == "Scale" and i > 0 and Layers[i - 1].type == "BatchNorm":
                # 如果当前层为被吸掉的bn后的scale层，跳过
                continue
            if Layers[i].type in unsupported_type:
                continue
            layer = Layers[i]
            _, input_shapes = self.GetLastLayerOutNameAndShape(layer)
            output_shape = self.GetLayerOutputShape(layer)
            s = ""
            p = ""
            if layer.type in ["Convolution", op_layer_info.Layer_CONVOLUTION]:
                param = layer.convolution_param
                ks = get_param(param.kernel_size, param.kernel_h, param.kernel_w)
                stride = get_param(param.stride, param.stride_h, param.stride_w, default=1)
                if isinstance(ks, tuple):
                    default_pad = ((ks[0] - 1) // 2, (ks[1] - 1) // 2)
                else:
                    default_pad = (ks - 1) // 2
                pad = get_param(param.pad, param.pad_h, param.pad_w, default=default_pad)
                dilation = get_param(param.dilation, default=1)
                group = param.group
                if group is None:
                    group = 1
                ic = input_shapes[0][1]
                oc = output_shape[1]
                if group == ic and ic == oc:
                    group = "C"

                s = "Conv"
                s += "-ks{}".format(ks)
                s += "-s{}".format(stride)
                s += "-p{}".format(pad)
                s += "-d{}".format(dilation)
                s += "-g{}".format(group)

                p += "ic{}".format(ic)
                p += "-oc{}".format(oc)

                if (
                    ks in [1, 3]
                    and stride in [1, 2, 3]
                    and pad == (ks - 1) // 2
                    and dilation == 1
                    and group == 1
                    and ic % 16 == 0
                    and oc % 16 == 0
                ):
                    if ic % 16 == 0 and oc % 16 == 0:
                        continue
            elif layer.type in ["ReLU", op_layer_info.Layer_RELU]:
                param = layer.relu_param
                s = "ReLU"
                ns = math.floor(param.negative_slope * 100 + 0.5) // 100
                s += "-ns{}".format(ns)

                if abs(ns) < 1e-3:
                    continue
            elif layer.type in ["Pooling", op_layer_info.Layer_POOLING]:
                param = layer.pooling_param
                s = ""
                if param.global_pooling:
                    s += "G"
                if param.pool == 0:
                    s += "MaxPool"
                elif param.pool == 1:
                    s += "AvgPool"
                else:
                    s += "StcPool"
                if not param.global_pooling:
                    pad = get_param(param.pad, param.pad_h, param.pad_w, default=0)
                    ks = get_param(param.kernel_size, param.kernel_h, param.kernel_w)
                    stride = get_param(param.stride, param.stride_h, param.stride_w, default=1)
                    s += "-p{}".format(pad)
                    s += "-ks{}".format(ks)
                    s += "-s{}".format(stride)
                    ic = input_shapes[0][1]
                    oc = output_shape[1]
                    p += "ic{}".format(ic)
                    p += "-oc{}".format(oc)
                    if param.pool == 0:
                        if ic >= 16 and ic <= 128 and stride == 2:
                            continue
                else:
                    h, w = input_shapes[0][2:]
                    if h == w:
                        ks = h
                    else:
                        ks = (h, w)
                    s += "-ks{}".format(ks)

            elif layer.type in ["InnerProduct", op_layer_info.Layer_INNER_PRODUCT]:
                param = layer.inner_product_param
                s = "InnerProduct"

                ic = input_shapes[0][1]
                oc = output_shape[1]
                p += "ic{}".format(ic)
                p += "-oc{}".format(oc)
                if ic <= 2048 and ic % 16 == 0 and oc <= 1024 and oc % 16 == 0:
                    continue
            elif layer.type in ["Concat", op_layer_info.Layer_CONCAT]:
                param = layer.concat_param
                s = "Concat"
                dim = param.axis
                if dim is None:
                    dim = param.concat_dim
                s += "-d{}".format(dim)

                n = len(input_shapes)
                p += "n{}".format(n)
                if n >= 3 and n <= 6 and dim == 1:
                    continue
            elif layer.type in ["Eltwise", op_layer_info.Layer_ELTWISE]:
                param = layer.eltwise_param
                s = "Eltwise"
                if param.operation == 0:
                    s += "-PROD"
                elif param.operation == 1:
                    s += "-SUM"
                else:
                    s += "-MAX"

                if param.operation == 1:
                    continue
            elif layer.type in ["Split"]:
                s = "Split"
            elif layer.type in ["Permute"]:
                param = layer.permute_param
                s = "Permute"
                s += "-od{}".format(param.order)
            elif layer.type in ["Slice"]:
                param = layer.slice_param
                s = "Slice"
                dim = param.axis
                if dim is None:
                    dim = param.slice_dim
                s += "-d{}".format(param.axis)
                s += "-sp{}".format(param.slice_point)
            else:
                s += "{}".format(layer.type)

            if s in ["Reshape", "Sigmoid", "BatchNorm", "Scale"]:
                continue

            if s != "":
                if s in param_set:
                    if p in param_set[s]:
                        param_set[s][p] += 1
                    else:
                        param_set[s][p] = 1
                else:
                    param_set[s] = dict()
                    param_set[s][p] = 1

        max_display = 30
        sorted_params = sorted(param_set.items(), key=lambda item: -len(item[1]))
        for i in range(min(max_display, len(sorted_params))):
            param, subparam = sorted_params[i]
            print("{}: ".format(param), end="")
            if len(subparam) == 1 and "" in subparam:
                print(subparam[""])
            else:
                print(sorted(subparam.items(), key=lambda item: -item[1]))
        print("==========================================")

    def GenerateOnnxNodeList(self, Layers):
        for i in range(len(Layers)):
            print("=================================================================")
            print(f"Converting layer: {Layers[i].name} | {Layers[i].type}")
            print("Input: ", Layers[i].bottom)
            print("Output: ", Layers[i].top)
            # Convolution
            if Layers[i].type == "Convolution" or Layers[i].type == op_layer_info.Layer_CONVOLUTION:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                conv_pname = self.AddInputsTVIFromParams(
                    Layers[i], op_layer_info.op_pname["Conv"], op_layer_info.op_ptype["Conv"]
                )
                input_name.extend(conv_pname)

                # 3.构建conv_node
                conv_node = nodes.createConv(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(conv_node)

            elif Layers[i].type == "ConvolutionDepthwise" or Layers[i].type == op_layer_info.Layer_CONVOLUTION:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                conv_pname = self.AddInputsTVIFromParams(
                    Layers[i], op_layer_info.op_pname["Conv"], op_layer_info.op_ptype["Conv"]
                )
                input_name.extend(conv_pname)

                # 3.构建conv_node
                conv_node = nodes.createConv(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(conv_node)

            # LSTM
            elif Layers[i].type == "LSTM":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])

                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.1 判断输入是否大于三维，若是，则需要进行pre-reshape
                need_pre_reshape = False
                for k in range(len(input_shape)):
                    if len(input_shape[k]) > 3:
                        need_pre_reshape = True

                        # 获取pre_reshape的参数
                        pre_reshape_input_shape = copy.deepcopy(input_shape[k])
                        pre_reshape_input_name = [copy.deepcopy(input_name[k])]
                        pre_reshape_output_name = [node_name + "_pre_reshape"]
                        prod = 1
                        for j in range(2, len(pre_reshape_input_shape)):
                            prod *= pre_reshape_input_shape[j]
                        pre_reshape_output_shape = [
                            pre_reshape_input_shape[0],
                            pre_reshape_input_shape[1],
                            prod,
                        ]

                        # 替换lstm的输入
                        input_name[k] = pre_reshape_output_name[0]
                        input_shape[k] = pre_reshape_output_shape
                        break

                # 2.2 增加pre_reshape 节点
                if need_pre_reshape:
                    # 2.2 pre_reshape 参数
                    pre_reshape_node = copy.deepcopy(Layers[i])
                    pre_reshape_node.name += "_pre_reshape"
                    pre_reshape_input_shape = [pre_reshape_input_shape]
                    pre_reshape_output_shape = [pre_reshape_output_shape]
                    pre_reshape_param_data = pre_reshape_output_shape
                    pre_reshape_param_shape = [np.shape(pre_reshape_output_shape[0])]

                    # 2.3 生成节点参数tensor value info，获取节点参数名并加入节点输入名列表
                    pre_reshape_pname = self.AddInputsTVIMannul(
                        pre_reshape_node,
                        op_layer_info.op_pname["Reshape"],
                        op_layer_info.op_ptype["Reshape"],
                        pre_reshape_param_shape,
                        pre_reshape_param_data,
                    )
                    pre_reshape_input_name.extend(pre_reshape_pname)

                    # 2.4 构建pre_reshape_node
                    pre_reshape_node = nodes.createReshape(
                        pre_reshape_node,
                        pre_reshape_output_name[0],
                        pre_reshape_input_name,
                        pre_reshape_output_name,
                        pre_reshape_output_shape,
                        pre_reshape_param_data,
                    )

                    # 2.5 添加到节点列表
                    self.onnxNodeList.append(pre_reshape_node)

                # 3.1 判断是否存在第二个输入clip，若存在则转换为全0，加到第一个输入上
                # 具体来说, data = onnx.Add(data, onnx.Reshape(onnx.Mul(clip, onnx.Constant(value=0)), shape=[0, -1, 1])
                # 这么做的原因是onnx.lstm不需要第二输入clip，为了简单起见进行如此转换
                has_clip = len(input_name) > 1
                if has_clip:
                    # 注：仅第一个lstm与clip相连
                    if hasattr(self, "first_lstm"):
                        input_name = input_name[0:1]
                        input_shape = input_shape[0:1]
                    else:
                        self.first_lstm = True
                        # 3.2 构造 Constant
                        clip_node = copy.deepcopy(Layers[i])
                        clip_node.name += "_clip"
                        const_node_name = clip_node.name + "_constant"
                        const_input_name = []
                        const_output_name = [const_node_name]
                        const_input_shape = []
                        const_value = helper.make_tensor(
                            name=f"{const_node_name}_value",
                            data_type=onnx.TensorProto.FLOAT,
                            dims=np.zeros(1).shape,
                            vals=np.zeros(1),
                        )
                        const_output_shape = [const_value.dims]
                        const_node = nodes.createConstant(
                            clip_node,
                            const_node_name,
                            const_input_name,
                            const_output_name,
                            const_value,
                            const_input_shape,
                        )
                        self.onnxNodeList.append(const_node)
                        # 3.3 构造Mul节点
                        mul_node_name = clip_node.name + "_mul"
                        mul_input_name = [input_name[1], const_node_name]
                        mul_input_shape = [input_shape[1], const_output_shape]
                        mul_output_name = [mul_node_name]
                        mul_output_shape = [input_shape[1]]
                        mul_node = nodes.create_mul_node(
                            clip_node,
                            mul_node_name,
                            mul_input_name,
                            mul_output_name,
                            mul_input_shape,
                        )
                        self.onnxNodeList.append(mul_node)
                        # 3.4.1 构造reshape节点
                        clip_reshape_node_name = clip_node.name + "_reshape"
                        clip_reshape_input_name = [mul_node_name]
                        clip_reshape_input_shape = mul_output_shape
                        clip_reshape_output_name = [clip_reshape_node_name]
                        clip_reshape_output_shape = [tuple(mul_output_shape[0]) + (1,)]
                        # 3.4.2 构造reshape节点的tensor value info
                        clip_reshape_param_data = clip_reshape_output_shape
                        clip_reshape_param_shape = [np.shape(clip_reshape_param_data[0])]
                        clip_reshape_pname = self.AddInputsTVIMannul(
                            clip_node,
                            op_layer_info.op_pname["Reshape"],
                            op_layer_info.op_ptype["Reshape"],
                            clip_reshape_param_shape,
                            clip_reshape_param_data,
                        )
                        clip_reshape_input_name.extend(clip_reshape_pname)
                        # 3.4.3 构造reshape_node
                        clip_reshape_node = nodes.createReshape(
                            clip_node,
                            clip_reshape_node_name,
                            clip_reshape_input_name,
                            clip_reshape_output_name,
                            clip_reshape_input_shape,
                            clip_reshape_output_shape,
                        )
                        self.onnxNodeList.append(clip_reshape_node)

                        # 3.5 构造Add节点
                        add_node_name = clip_node.name + "_add"
                        add_input_name = [input_name[0], clip_reshape_node_name]
                        add_input_shape = [input_shape[0], clip_reshape_output_shape[0]]
                        add_output_name = [add_node_name]
                        add_output_shape = [input_shape[0]]
                        add_node = nodes.create_add_node(
                            clip_node,
                            add_node_name,
                            add_input_name,
                            add_output_name,
                            add_input_shape,
                        )
                        self.onnxNodeList.append(add_node)

                        # 3.5 更新lstm的输入
                        input_name = add_output_name
                        input_shape = add_output_shape

                # 4.1 生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                lstm_pname, lstm_pshape = self.AddInputsTVIFromParams(
                    Layers[i], op_layer_info.op_pname["LSTM"], op_layer_info.op_ptype["LSTM"]
                )
                input_name.extend(lstm_pname)
                input_shape.extend(lstm_pshape)

                # 4.2 构建lstm_node
                lstm_node, lstm_output_name, lstm_output_shape = nodes.createLSTM(
                    Layers[i], node_name, input_name, output_name, input_shape
                )

                # 4.3 添加节点到节点列表
                self.onnxNodeList.append(lstm_node)

                # 5. 由于 caffe lstm 输出尺寸为(T, B, H), onnx lstm 输出尺寸为 (T, D, B, H)，D为方向数即为1，因此再接一个reshape层
                # 5.1 获取reshape层的输入尺寸
                reshape_param_data = copy.deepcopy(lstm_output_shape[0:1])
                del reshape_param_data[0][1]
                reshape_param_shape = [np.shape(reshape_param_data[0])]

                # 5.2 生成reshape层节点参数
                reshape_pname = self.AddInputsTVIMannul(
                    Layers[i],
                    op_layer_info.op_pname["Reshape"],
                    op_layer_info.op_ptype["Reshape"],
                    reshape_param_shape,
                    reshape_param_data,
                )

                # 5.3 构建reshape_node
                reshape_input_name = copy.deepcopy(lstm_output_name[0:1])
                reshape_input_name.extend(reshape_pname)
                reshape_node_name = node_name + "_reshape"
                reshape_output_name = copy.deepcopy(output_name[0:1])
                reshape_output_shape = copy.deepcopy(lstm_output_shape[0:1])
                reshape_node = nodes.createReshape(
                    Layers[i],
                    reshape_node_name,
                    reshape_input_name,
                    reshape_output_name,
                    reshape_output_shape,
                    reshape_param_data,
                )

                # 5.4 添加节点到节点列表
                self.onnxNodeList.append(reshape_node)

            # BatchNorm+Scale
            elif Layers[i].type == "BatchNorm" or Layers[i].type == "BN":
                # 1.获取节点输入名、输入维度、输出名、节点名
                bn_node = copy.deepcopy(Layers[i])
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                if i < len(Layers) - 1 and Layers[i + 1].type == "Scale":
                    scale_pname, scale_pshape = self.AddInputsTVIFromParams(
                        Layers[i + 1], op_layer_info.op_pname["Scale"], op_layer_info.op_ptype["Scale"]
                    )
                    bn_pname, bn_pshape = self.AddInputsTVIFromParams(
                        Layers[i], op_layer_info.op_pname["BatchNorm"], op_layer_info.op_ptype["BatchNorm"]
                    )
                    assert bn_pshape == scale_pshape, "BatchNorm and Scale params should share the same shape"
                    input_name.extend(scale_pname)
                    input_name.extend(bn_pname)

                    # 替换bn层的output为scale层的output
                    scale_output_name = self.GetCurrentLayerOutName(Layers[i + 1])
                    output_name = scale_output_name
                    del bn_node.top[:]
                    bn_node.top.extend(Layers[i + 1].top)

                else:
                    bn_pshape, _ = self.GetParamsShapeAndData(Layers[i])
                    custom_params = [
                        np.ones(shape=bn_pshape[0], dtype=np.float),
                        np.zeros(shape=bn_pshape[1], dtype=np.float),
                    ]
                    scale_pname = self.AddInputsTVIMannul(
                        Layers[i],
                        op_layer_info.op_pname["Scale"],
                        op_layer_info.op_ptype["Scale"],
                        bn_pshape,
                        custom_params,
                    )
                    bn_pname, bn_pshape = self.AddInputsTVIFromParams(
                        Layers[i], op_layer_info.op_pname["BatchNorm"], op_layer_info.op_ptype["BatchNorm"]
                    )
                    input_name.extend(scale_pname)
                    input_name.extend(bn_pname)

                # 3.构建bn_node
                bn_node = nodes.createBN(bn_node, node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(bn_node)

            elif Layers[i].type == "Scale":
                if i > 0 and (Layers[i - 1].type == "BatchNorm" or Layers[i - 1].type == "BN"):
                    # bn + scale
                    continue
                # signal scale
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name
                has_two_input = len(input_name) > 1

                if has_two_input:
                    # 相当于Mul
                    if nodes.need_add_reshape(input_shape):
                        # 第二个输入的shape必须符合onnx的broadcast规则，与caffe的broadcast不同
                        reshape_layer = copy.deepcopy(Layers[i])
                        # add reshape layer
                        reshape_node_name = input_name[1] + "_reshape"

                        reshape_input_name = input_name[1]
                        reshape_input_shape = input_shape[1]

                        reshape_shape_data = nodes.get_param_shape(input_shape)
                        reshape_shape_shape = np.shape(reshape_shape_data)

                        reshape_params = self.AddInputsTVIMannul(
                            Layers[i],
                            [reshape_node_name + "shape"],
                            [onnx.TensorProto.INT64],
                            [reshape_shape_shape],
                            [reshape_shape_data],
                        )

                        reshape_output_name = [reshape_input_name + "_output_name"]

                        reshape_node = nodes.createReshape(
                            reshape_layer,
                            reshape_node_name,
                            [reshape_input_name, reshape_params[0]],
                            reshape_output_name,
                            reshape_input_shape,
                            output_shape=[reshape_shape_data],
                        )

                        self.onnxNodeList.append(reshape_node)
                        input_name[1] = reshape_output_name[0]
                        input_shape[1] = reshape_shape_data

                    # create mul node
                    mul_node = nodes.create_mul_node(Layers[i], node_name, input_name, output_name, input_shape)

                    self.onnxNodeList.append(mul_node)
                else:
                    # 带有可学习参数的Scale层 = Mul
                    param_shape, param_data = self.GetParamsShapeAndData(Layers[i])
                    # scale的param永远是channel维的
                    # Scale = Mul + Add
                    if len(param_shape) == 2:
                        # create mul
                        param_scale_shape = [1 for i in range(len(input_shape[0]))]
                        param_scale_shape[1] = param_shape[0][0]
                        param_scale_data = param_data[0]
                        param_scale_name = self.AddInputsTVIMannul(
                            Layers[i],
                            ["_scale"],
                            [onnx.TensorProto.FLOAT],
                            [param_scale_shape],
                            [param_scale_data],
                        )

                        mul_node_name = node_name + "_mul"
                        mul_input_name = [input_name[0], param_scale_name[0]]
                        mul_output_name = [output_name[0] + "_mul"]
                        mul_input_shape = [input_shape[0], param_scale_shape]

                        mul_node = nodes.create_mul_node(
                            Layers[i],
                            mul_node_name,
                            mul_input_name,
                            mul_output_name,
                            mul_input_shape,
                        )
                        self.onnxNodeList.append(mul_node)

                        param_bias_shape = [1 for i in range(len(input_shape[0]))]
                        param_bias_shape[1] = param_shape[1][0]
                        param_bias_data = param_data[1]
                        param_bias_name = self.AddInputsTVIMannul(
                            Layers[i],
                            ["_bias"],
                            [onnx.TensorProto.FLOAT],
                            [param_bias_shape],
                            [param_bias_data],
                        )

                        add_node_name = node_name + "_add"
                        add_input_name = [mul_output_name[0], param_bias_name[0]]
                        add_output_name = output_name
                        add_input_shape = [input_shape[0], param_bias_shape]
                        add_node = nodes.create_add_node(
                            Layers[i],
                            add_node_name,
                            add_input_name,
                            add_output_name,
                            add_input_shape,
                        )
                        self.onnxNodeList.append(add_node)
                    if len(param_shape) == 1:
                        # create mul
                        param_scale_shape = [1 for i in range(len(input_shape[0]))]
                        param_scale_shape[1] = param_shape[0][0]
                        param_scale_data = param_data[0]
                        param_scale_name = self.AddInputsTVIMannul(
                            Layers[i],
                            ["_scale"],
                            [onnx.TensorProto.FLOAT],
                            [param_scale_shape],
                            [param_scale_data],
                        )

                        mul_input_name = [input_name[0], param_scale_name[0]]
                        mul_input_shape = [input_shape[0], param_scale_shape]

                        mul_node = nodes.create_mul_node(
                            Layers[i],
                            node_name,
                            mul_input_name,
                            output_name,
                            mul_input_shape,
                        )
                        self.onnxNodeList.append(mul_node)

            # Pooling
            elif Layers[i].type == "Pooling" or Layers[i].type == op_layer_info.Layer_POOLING:
                # NOTE： 由于 Caffe 和 ONNX 对 AveragePool 的处理的方式的不同，所以需要在avgpool node 之前添加 Pad node
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name
                pool_name = node_name
                pool_input_name = input_name
                pool_output_name = output_name
                pool_input_shape = input_shape

                # 如果是avgpool，在pool前增加 pad node
                pool_type = nodes.pooling_type(Layers[i])
                if pool_type == "GlobalAveragePool" or pool_type == "AveragePool":
                    pads = nodes.get_pool_pads(Layers[i])
                    pads_shape = [np.shape(pads)]
                    pads_name = node_name + "_pre_pad"
                    pads_output_name = [node_name + "_pre_pad"]
                    pads_output_shape = nodes.calculate_pad_output_shape(input_shape, pads)
                    pads_param = self.AddInputsTVIMannul(
                        Layers[i], ["_pad"], [onnx.TensorProto.INT64], pads_shape, [pads]
                    )
                    input_name.extend(pads_param)

                    pad_node = nodes.create_pad_node(Layers[i], pads_name, input_name, pads_output_name, input_shape)
                    self.onnxNodeList.append(pad_node)

                    # 修改pool参数
                    pool_input_name = copy.deepcopy(pads_output_name)
                    pool_input_shape = copy.deepcopy(pads_output_shape)

                # 2. 在某种特殊情况下，需要在pool层后增加slice层, 判断是否需要slice
                need_slice, pool_output_shape, start, end, axis = nodes.need_slice(Layers[i], pool_type, input_shape)
                if need_slice:
                    slice_output_name = copy.deepcopy(output_name)
                    pool_name = node_name + "_pool"
                    pool_output_name = [output_name[0] + "_pool"]
                    slice_input_name = copy.deepcopy(pool_output_name)

                # 3.构建pool_node
                pool_node = nodes.create_pooling_node(
                    Layers[i],
                    pool_name,
                    pool_input_name,
                    pool_output_name,
                    pool_input_shape,
                )

                # 4.添加节点到节点列表
                self.onnxNodeList.append(pool_node)

                # 5. 在某种特殊情况下，需要在pool层后增加slice层
                if need_slice:

                    slice_name = node_name + "_slice"
                    # starts ends axes 的 shape 是相同的
                    slice_shape = [np.shape(start)]

                    starts_param = self.AddInputsTVIMannul(
                        Layers[i],
                        ["_starts"],
                        [onnx.TensorProto.INT64],
                        slice_shape,
                        [start],
                    )
                    ends_param = self.AddInputsTVIMannul(
                        Layers[i],
                        ["_ends"],
                        [onnx.TensorProto.INT64],
                        slice_shape,
                        [end],
                    )
                    axes_param = self.AddInputsTVIMannul(
                        Layers[i],
                        ["_axes"],
                        [onnx.TensorProto.INT64],
                        slice_shape,
                        [axis],
                    )
                    slice_input_name.extend(starts_param)
                    slice_input_name.extend(ends_param)
                    slice_input_name.extend(axes_param)

                    Slice_node = nodes.createSlice(
                        Layers[i],
                        slice_name,
                        slice_input_name,
                        slice_output_name,
                        pool_output_shape,
                        start,
                        end,
                        axis,
                    )
                    # 3. 添加节点到节点列表
                    self.onnxNodeList.append(Slice_node)

            # MaxUnPool
            elif Layers[i].type == "MaxUnpool":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建unpool_node
                unpool_node = nodes.createUnPooling(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(unpool_node)

            elif Layers[i].type == "Reorg":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                reorg_node = nodes.createSpaceToDepth(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(reorg_node)

            elif Layers[i].type == "Reverse":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                reverse_layer = copy.deepcopy(Layers[i])

                axis = reverse_layer.reverse_param.axis
                start = input_shape[0][axis] - 1
                end = -input_shape[0][axis] - 1
                step = -1
                shape = np.shape([1])

                # add param node
                param_nodes = self.AddInputsTVIMannul(
                    reverse_layer,
                    ["_starts", "_ends", "_axes", "_steps"],
                    [onnx.TensorProto.INT64 for i in range(4)],
                    [shape for i in range(4)],
                    [[start], [end], [axis], [step]],
                )
                input_name.extend(param_nodes)

                reverse_node = nodes.createSlice(
                    reverse_layer,
                    node_name,
                    input_name,
                    output_name,
                    input_shape,
                    [start],
                    [end],
                    [axis],
                    [step],
                )
                self.onnxNodeList.append(reverse_node)

            # Eltwise
            elif Layers[i].type == "Eltwise" or Layers[i].type == op_layer_info.Layer_ELTWISE:
                # 1.获取节点输入名、输入维度、输出名、节点名
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状

                node_name = Layers[i].name

                # 2.构建eltwise_node
                eltwise_node = nodes.createEltwise(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(eltwise_node)

            # Softmax
            elif Layers[i].type == "Softmax" or Layers[i].type == op_layer_info.Layer_SOFTMAX:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建softmax_node
                softmax_node = nodes.createSoftmax(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(softmax_node)

            # Relu
            elif Layers[i].type == "ReLU" or Layers[i].type == op_layer_info.Layer_RELU:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建relu_node
                relu_node = nodes.createRelu(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(relu_node)
            # PRelu
            elif Layers[i].type == "PReLU":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                param_shape, param_data = self.GetParamsShapeAndData(Layers[i])

                # broadcast param_data
                param_shape[0] = list(param_shape[0]) + [1 for i in range(len(input_shape[0]) - 2)]
                param_data[0] = np.asarray(param_data[0]).reshape(param_shape[0])

                pname = self.AddInputsTVIMannul(
                    Layers[i],
                    op_layer_info.op_pname["PRelu"],
                    op_layer_info.op_ptype["PRelu"],
                    param_shape,
                    param_data,
                )

                input_name.extend(pname)

                # 3.构建PRelu_node
                PRelu_node = nodes.createPRelu(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(PRelu_node)
            # relu6
            elif Layers[i].type == "ReLU6":
                # relu6 = clip(0, 6)
                # add relu node
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                min_value = np.float(0)
                max_value = np.float(6)
                min_param = self.AddInputsTVIMannul(Layers[i], ["_min"], [onnx.TensorProto.FLOAT], [[]], [[min_value]])
                input_name.extend(min_param)
                max_param = self.AddInputsTVIMannul(Layers[i], ["_max"], [onnx.TensorProto.FLOAT], [[]], [[max_value]])
                input_name.extend(max_param)
                relu6_node = nodes.create_clip_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(relu6_node)

            elif Layers[i].type == "Clip":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # get clip attribute: min, max
                clip_attr = nodes.get_clip_attribute(Layers[i])

                # make min node
                min_value = np.float(clip_attr["min"])
                min_param = self.AddInputsTVIMannul(Layers[i], ["_min"], [onnx.TensorProto.FLOAT], [[]], [[min_value]])
                input_name.extend(min_param)

                # make max node
                max_value = np.float(clip_attr["max"])
                max_param = self.AddInputsTVIMannul(Layers[i], ["_max"], [onnx.TensorProto.FLOAT], [[]], [[max_value]])
                input_name.extend(max_param)

                # make clip node
                clip_node = nodes.create_clip_node(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(clip_node)

            elif Layers[i].type == "Sigmoid":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建relu_node
                sigmoid_node = nodes.createSigmoid(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(sigmoid_node)

            elif Layers[i].type == "Mish":
                # mish(x) = x * tanh(softplus(x))
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # create softplus
                softplus_name = node_name + "_softplus"
                softplus_node = nodes.createSoftplus(Layers[i], softplus_name, input_name, [softplus_name], input_shape)
                self.onnxNodeList.append(softplus_node)

                # create tanh
                tanh_name = node_name + "_tanh"
                tanh_node = nodes.createTanh(Layers[i], tanh_name, [softplus_name], [tanh_name], input_shape)
                self.onnxNodeList.append(tanh_node)

                # create mul
                mul_input_names = input_name + [tanh_name]
                mul_input_shapes = input_shape * 2
                mul_name = node_name + "_mul"
                mul_node = nodes.create_mul_node(Layers[i], mul_name, mul_input_names, output_name, mul_input_shapes)
                self.onnxNodeList.append(mul_node)

            elif Layers[i].type == "Log":
                # log(x) = log_base(shift + scale * x)
                # base is set to e if base is set to the default (-1). and we only support "base == -1" here.
                base, scale, shift = nodes.get_log_attributes(Layers[i])
                if base != -1:
                    raise NotImplementedError("Currently only support default base e for Log Op.")

                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                intermediate_input_name = input_name[0]

                if scale != 1:
                    # create Mul node
                    scale = np.array([scale])
                    scale_node_name = self.AddInputsTVIMannul(
                        Layers[i],
                        ["_scale"],
                        [onnx.TensorProto.FLOAT],
                        [np.shape(scale)],
                        [scale],
                    )
                    mul_input_name = [intermediate_input_name, scale_node_name[0]]

                    intermediate_input_name = output_name[0] + "_mul"
                    mul_node = nodes.create_mul_node(
                        Layers[i],
                        node_name + "_mul",
                        mul_input_name,
                        [intermediate_input_name],
                        [input_shape[0], np.shape(scale)],
                    )
                    self.onnxNodeList.append(mul_node)

                if shift != 0:
                    # create Add node
                    shift = np.array([shift])
                    shift_node_name = self.AddInputsTVIMannul(
                        Layers[i],
                        ["_shift"],
                        [onnx.TensorProto.FLOAT],
                        [np.shape(scale)],
                        [shift],
                    )
                    add_input_name = [intermediate_input_name, shift_node_name[0]]
                    intermediate_input_name = output_name[0] + "_add"
                    add_node = nodes.create_add_node(
                        Layers[i],
                        node_name + "_add",
                        add_input_name,
                        [intermediate_input_name],
                        [input_shape[0], np.shape(shift)],
                    )
                    self.onnxNodeList.append(add_node)

                # create Log node
                log_name = node_name + "_log"
                log_node = nodes.create_log_node(
                    Layers[i],
                    log_name,
                    [intermediate_input_name],
                    output_name,
                    input_shape,
                )
                self.onnxNodeList.append(log_node)
            elif Layers[i].type == "Exp":
                # exp(x) = base ^ (shift + scale * x)
                # base is set to e if base is set to the default (-1). and we only support "base == -1" here.
                base, scale, shift = nodes.get_exp_attributes(Layers[i])
                if base != -1:
                    raise NotImplementedError(f"Currently only support default base e for Log Op, but get: {base}")

                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                intermediate_input_name = input_name[0]

                if scale != 1:
                    # create Mul node
                    scale = np.array([scale])
                    scale_node_name = self.AddInputsTVIMannul(
                        Layers[i],
                        ["_scale"],
                        [onnx.TensorProto.FLOAT],
                        [np.shape(scale)],
                        [scale],
                    )
                    mul_input_name = [intermediate_input_name, scale_node_name[0]]

                    intermediate_input_name = output_name[0] + "_mul"
                    mul_node = nodes.create_mul_node(
                        Layers[i],
                        node_name + "_mul",
                        mul_input_name,
                        [intermediate_input_name],
                        [input_shape[0], np.shape(scale)],
                    )
                    self.onnxNodeList.append(mul_node)

                if shift != 0:
                    # create Add node
                    shift = np.array([shift])
                    shift_node_name = self.AddInputsTVIMannul(
                        Layers[i],
                        ["_shift"],
                        [onnx.TensorProto.FLOAT],
                        [np.shape(scale)],
                        [shift],
                    )
                    add_input_name = [intermediate_input_name, shift_node_name[0]]
                    intermediate_input_name = output_name[0] + "_add"
                    add_node = nodes.create_add_node(
                        Layers[i],
                        node_name + "_add",
                        add_input_name,
                        [intermediate_input_name],
                        [input_shape[0], np.shape(shift)],
                    )
                    self.onnxNodeList.append(add_node)

                # create Log node
                exp_name = node_name + "_log"
                exp_node = nodes.create_exp_node(
                    Layers[i],
                    exp_name,
                    [intermediate_input_name],
                    output_name,
                    input_shape,
                )
                self.onnxNodeList.append(exp_node)
            # LRN
            elif Layers[i].type == "LRN" or Layers[i].type == op_layer_info.Layer_LRN:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.构建LRN_node
                LRN_node = nodes.createLRN(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(LRN_node)
            # Dropout
            elif Layers[i].type == "Dropout" or Layers[i].type == op_layer_info.Layer_DROPOUT:
                # Dropout层在测试时不生效，因此直接丢弃dropout层并修复拓扑
                node_name = Layers[i].name
                assert len(Layers[i].bottom) == 1 and len(Layers[i].top) == 1, Layers[i]
                node_bottom = Layers[i].bottom[0]
                node_top = Layers[i].top[0]
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                for j in range(i + 1, len(Layers)):
                    if node_top in Layers[j].bottom:
                        bottoms = []
                        for bottom in Layers[j].bottom:
                            if bottom == node_top:
                                bottoms.append(node_bottom)
                            else:
                                bottoms.append(bottom)
                        del Layers[j].bottom[:]
                        Layers[j].bottom.extend(bottoms)

            # Upsample
            elif Layers[i].type == "Upsample" or Layers[i].type == op_layer_info.Layer_UPSAMPLE:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                # add roi input

                # add scales input
                scales = nodes.get_upsample_attri(Layers[i])["scales"]
                paramshape = [[8, 1], [4, 1]]
                paramdata = [
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    scales,
                ]

                pname = self.AddInputsTVIMannul(
                    Layers[i],
                    op_layer_info.op_pname["Upsample"],
                    op_layer_info.op_ptype["Upsample"],
                    paramshape,
                    paramdata,
                )

                input_name.extend(pname)

                # 3.构建Upsample_node
                Upsample_node = nodes.create_resize_node(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(Upsample_node)

            elif Layers[i].type == "Interp":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                interp_node = nodes.create_interp_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(interp_node)

            # Concat
            elif Layers[i].type == "Concat" or Layers[i].type == op_layer_info.Layer_CONCAT:
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.构建Concat_node
                Concat_node = nodes.createConcat(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(Concat_node)

            elif Layers[i].type == "Slice":
                # 1. 获取节点书输入名，输入维度，输出名，节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name_list = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                starts, ends, axes = nodes.analyzeLayer(Layers[i], input_shape, len(output_name_list))

                SliceLayer = copy.deepcopy(Layers[i])

                for j in range(len(output_name_list)):
                    # 放在这里的原因是
                    slice_name = copy.deepcopy(input_name)
                    # starts ends axes 的 shape 是相同的
                    shape = [np.shape([1])]

                    starts_param = self.AddInputsTVIMannul(
                        SliceLayer,
                        ["_starts" + str(j)],
                        [onnx.TensorProto.INT64],
                        shape,
                        [[starts[j]]],
                    )
                    ends_param = self.AddInputsTVIMannul(
                        SliceLayer,
                        ["_ends" + str(j)],
                        [onnx.TensorProto.INT64],
                        shape,
                        [[ends[j]]],
                    )
                    axes_param = self.AddInputsTVIMannul(
                        SliceLayer,
                        ["_axes" + str(j)],
                        [onnx.TensorProto.INT64],
                        shape,
                        [[axes[j]]],
                    )
                    slice_name.extend(starts_param)
                    slice_name.extend(ends_param)
                    slice_name.extend(axes_param)

                    Slice_node = nodes.createSlice(
                        SliceLayer,
                        output_name_list[j],
                        slice_name,
                        [output_name_list[j]],
                        input_shape,
                        [starts[j]],
                        [ends[j]],
                        [axes[j]],
                    )
                    # 3. 添加节点到节点列表
                    self.onnxNodeList.append(Slice_node)
            # Reshape
            elif Layers[i].type == "Reshape":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                reshape_param = nodes.get_reshape_param(Layers[i], input_shape)
                reshape_param_shape = [np.shape(reshape_param)]
                pname = self.AddInputsTVIMannul(
                    Layers[i],
                    op_layer_info.op_pname["Reshape"],
                    op_layer_info.op_ptype["Reshape"],
                    reshape_param_shape,
                    [reshape_param],
                )
                input_name.extend(pname)

                # 3.构建reshape节点
                reshape_node = nodes.createReshape(Layers[i], node_name, input_name, output_name, input_shape)

                # 4.添加点到节点列表
                self.onnxNodeList.append(reshape_node)

            # InnerProduct
            # 由于onnx中没有全连接层，因此将其拆分为Reshape+Gemm(+Reshape)
            elif Layers[i].type == "InnerProduct" or Layers[i].type == op_layer_info.Layer_INNER_PRODUCT:
                node_layer = copy.deepcopy(Layers[i])  # 深拷贝
                node_input_name, node_input_shape = self.GetLastLayerOutNameAndShape(node_layer)  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                start_axis = node_layer.inner_product_param.axis
                need_post_reshape = start_axis != 1  # 是否需要在gemm后再reshape一次

                reshape_outname = ""
                reshape_output_shape = nodes.getReshapeOutShape(node_layer, node_input_shape)
                need_reshape = 0 if reshape_output_shape[0] == node_input_shape[0] else 1

                if need_reshape:
                    # 一、reshape
                    # 1.获取节点输入名、输入维度、输出名、节点名
                    reshape_outname = [node_layer.name + "_Reshape"]
                    reshape_nodename = node_layer.name + "_Reshape"

                    # 2.生成节点参数tensor value info,并获取节点参数名, 将参数名加入节点输入名列表
                    paramshape = [[2]]
                    reshape_pname = self.AddInputsTVIMannul(
                        node_layer,
                        op_layer_info.op_pname["Reshape"],
                        op_layer_info.op_ptype["Reshape"],
                        paramshape,
                        reshape_output_shape,
                    )
                    node_input_name.extend(reshape_pname)
                    # 3.构建reshape_node
                    reshape_node = nodes.createReshape(
                        node_layer,
                        reshape_nodename,
                        node_input_name,
                        reshape_outname,
                        node_input_shape,
                    )

                    # 4.添加节点到节点列表
                    self.onnxNodeList.append(reshape_node)

                # import ipdb; ipdb.set_trace()

                # 二、Gemm 最后一个node输出保持原名称
                gemm_layer = copy.deepcopy(Layers[i])  # 深拷贝
                # 1.获取节点输入名、输入维度、输出名、节点名
                gemm_inname = reshape_outname if need_reshape == 1 else node_input_name
                gemm_input_shape = reshape_output_shape if need_reshape == 1 else node_input_shape
                gemm_outname = output_name if not need_post_reshape else [node_layer.name + "_gemm"]
                gemm_nodename = gemm_outname[0]

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                gemm_pname = self.AddInputsTVIFromParams(
                    gemm_layer, op_layer_info.op_pname["InnerProduct"], op_layer_info.op_ptype["InnerProduct"]
                )  # 获取输入参数，对于add来说blobs[1]里存放的是bias不需要,所以直接获取blobs[0]
                gemm_inname.extend(gemm_pname)

                # 3.构建gemm_node
                gemm_node, gemm_output_shape = nodes.createGemm(
                    gemm_layer,
                    gemm_nodename,
                    gemm_inname,
                    gemm_outname,
                    gemm_input_shape,
                    gemm_layer.inner_product_param.num_output,
                )

                # 4.添加节点到节点列表
                self.onnxNodeList.append(gemm_node)

                # 三.判断是否需要post reshape
                if not need_post_reshape:
                    continue
                # 1. 获取节点输入名、输入维度、输出名、节点名
                post_reshape_node = copy.deepcopy(Layers[i])
                post_reshape_node.name += "_post_reshape"
                post_reshape_inname = gemm_outname
                post_reshape_input_shape = gemm_output_shape
                post_reshape_outname = output_name
                post_reshape_name = gemm_layer.name

                # 2. 获取节点输出尺寸
                post_reshape_output_shape = [[]]
                for j in range(0, start_axis):
                    post_reshape_output_shape[0].append(node_input_shape[0][j])
                post_reshape_output_shape[0].append(gemm_layer.inner_product_param.num_output)

                # 2. 生成节点参数tensor value info,并获取节点参数名, 将参数名加入节点输入名列表
                paramshape = [[len(post_reshape_output_shape[0])]]
                reshape_pname = self.AddInputsTVIMannul(
                    post_reshape_node,
                    op_layer_info.op_pname["Reshape"],
                    op_layer_info.op_ptype["Reshape"],
                    paramshape,
                    post_reshape_output_shape,
                )
                post_reshape_inname.extend(reshape_pname)
                # 3.构建reshape_node
                post_reshape_node = nodes.createReshape(
                    node_layer,
                    post_reshape_name,
                    post_reshape_inname,
                    post_reshape_outname,
                    post_reshape_input_shape,
                    post_reshape_output_shape,
                )

                # 4.添加节点到节点列表
                self.onnxNodeList.append(post_reshape_node)

            elif Layers[i].type == "ShuffleChannel":
                # TODO support ShuffleChannel
                # reshape  [N, C, H, W] tensor to [N, G, C', H, W]
                node_layer = copy.deepcopy(Layers[i])  # 深拷贝
                node_input_name, node_input_shape = self.GetLastLayerOutNameAndShape(node_layer)  # 获取输入名列表和输入形状

                reshape_outname = ""
                reshape_output_shape = nodes.getReshapeOutShape(Layers[i], node_input_shape)
                need_reshape = 0 if reshape_output_shape[0] == node_input_shape[0] else 1

                if need_reshape:
                    # 一. reshape  [N, C, H, W] tensor to [N, G, C', H, W]
                    # 1.获取节点输入名、输入维度、输出名、节点名
                    reshape_outname = [node_layer.name + "_Reshape"]
                    reshape_nodename = node_layer.name + "_Reshape"

                    # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                    param_data = nodes.getReshapeOutShape(node_layer, node_input_shape)
                    param_shape = np.array([1, 2, 3, 4, 5], np.int).shape
                    reshape_pname = self.AddInputsTVIMannul(
                        node_layer,
                        op_layer_info.op_pname["Reshape"],
                        op_layer_info.op_ptype["Reshape"],
                        [param_shape],
                        param_data,
                    )

                    node_input_name.extend(reshape_pname)
                    # 这里不用对输入进行拓展，因为输入没有增加
                    # node_input_name.extend(reshape_pname)
                    # 3.构建reshape_node
                    reshape_node = nodes.createReshape(
                        node_layer,
                        reshape_nodename,
                        node_input_name,
                        reshape_outname,
                        node_input_shape,
                    )

                    # 4.添加节点到节点列表
                    self.onnxNodeList.append(reshape_node)

                # 2. transpose  [N, C', G, H, W]
                transpose_layer = copy.deepcopy(Layers[i])  # 深拷贝
                # 1.获取节点输入名、输入维度、输出名、节点名
                transpose_input_name = reshape_outname if need_reshape == 1 else node_input_name
                transpose_input_shape = reshape_output_shape if need_reshape == 1 else node_input_shape
                transpose_output_name = [node_layer.name + "_Transpose"]
                transpose_node_name = node_layer.name + "_Transpose"

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                # 获取输入参数，对于add来说blobs[1]里存放的是bias不需要,所以直接获取blobs[0]

                # TODO 这地方为什么要选择使用AddInputsTVIMannul？取决于什么？
                # ANSWER: 取决于要转换的 onnx 的类型
                # TODO param_date 是什么？为什么要设置这个变量
                param_data = [[2]]
                # transpose_pname = self.AddInputsTVIMannul(transpose_layer,
                #                                      op_layer_info.op_pname["Transpose"],
                #                                      op_layer_info.op_ptype['Transpose'],
                #                                      param_data,
                #                                      transpose_input_shape)
                # transpose_input_name.extend(transpose_pname)
                # 3.
                transpose_node = nodes.createTranspose(
                    transpose_layer,
                    transpose_node_name,
                    transpose_input_name,
                    transpose_output_name,
                    transpose_input_shape,
                )
                # 4.添加节点到节点列表
                self.onnxNodeList.append(transpose_node)

                # 三、 Reshape [N, C', G, H, W] tensor to [N, C, H, W]
                #
                end_layer = copy.deepcopy(Layers[i])
                end_layer.type = "DeReshape"
                # 最后的输出的节点要保持原名称，这是为了生成该节点，保持链路畅通
                end_output_name = [end_layer.name]
                end_node_name = end_layer.name

                # 上一层的输出是这一层的输入
                end_input_name = transpose_node.outputs_name
                end_input_shape = transpose_node.outputs_shape
                # 最后保持输出和输入的形状是一致的
                end_output_shape = [
                    [
                        node_input_shape[0][0],
                        -1,
                        node_input_shape[0][2],
                        node_input_shape[0][3],
                    ]
                ]
                param_shape = [np.array([1, 2, 3, 4], dtype=np.int).shape]
                end_pname = self.AddInputsTVIMannul(
                    node_layer,
                    op_layer_info.op_pname["DouReshape"],
                    op_layer_info.op_ptype["DouReshape"],
                    param_shape,
                    end_output_shape,
                )

                end_input_name.extend(end_pname)
                # 构建
                end_node = nodes.createReshape(
                    end_layer,
                    end_node_name,
                    end_input_name,
                    end_output_name,
                    end_input_shape,
                )

                self.onnxNodeList.append(end_node)

            # Deconvolution
            elif Layers[i].type == "Deconvolution":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表

                conv_pname = self.AddInputsTVIFromParams(
                    Layers[i], op_layer_info.op_pname["ConvTranspose"], op_layer_info.op_ptype["ConvTranspose"]
                )
                input_name.extend(conv_pname)

                # 3.构建conv_node
                conv_node = nodes.createConvTranspose(Layers[i], node_name, input_name, output_name, input_shape)
                # if True:
                #     self.__print_debug_info(node_name, input_name, output_name, input_shape, conv_node.outputs_shape)

                # 4.添加节点到节点列表
                self.onnxNodeList.append(conv_node)

            # Flatten
            elif Layers[i].type == "Flatten":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                # 由于后面 Flatten 的优化有问题,所以目前先将 Flatten -> reshape
                # flatten_node = nodes.create_flatten_node(layers[i], node_name, input_name,
                #                                       output_name, input_shape)
                # self.onnxnodelist.append(flatten_nodelatten_node)
                # continue

                # Flatten -> Reshape
                # import ipdb; ipdb.set_trace()
                # # 2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                param_data = nodes.getReshapeOutShape(Layers[i], input_shape)
                param_shape = [np.shape(param_data)]
                # check reshape parameter legality
                if len(param_shape[0]) == 2 and param_shape[0][0] == 1:
                    param_shape = [tuple([param_shape[0][1]])]
                reshape_pname = self.AddInputsTVIMannul(
                    Layers[i],
                    op_layer_info.op_pname["Reshape"],
                    op_layer_info.op_ptype["Reshape"],
                    param_shape,
                    param_data,
                )
                input_name.extend(reshape_pname)

                # 3.构建reshape_node
                reshape_node = nodes.createReshape(Layers[i], node_name, input_name, output_name, input_shape)
                # 4.添加节点到节点列表
                self.onnxNodeList.append(reshape_node)

            elif Layers[i].type == "Permute":
                # Permute -> Transpose
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                transpose_node = nodes.createTranspose(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(transpose_node)
            elif Layers[i].type == "PriorBox":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                priorbox_node = nodes.create_priorbox_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(priorbox_node)

            elif Layers[i].type == "DetectionOutput":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                detection_output_node = nodes.create_detection_output(
                    Layers[i], node_name, input_name, output_name, input_shape
                )
                self.onnxNodeList.append(detection_output_node)

            elif Layers[i].type == "SpatialTransformer":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                output_node = nodes.create_spatial_transformer(
                    Layers[i], node_name, input_name, output_name, input_shape, self.proto_path
                )
                self.onnxNodeList.append(output_node)

            elif Layers[i].type == "Axpy":
                # axpy = mul + add
                # top = bottom[0] * bottom[1] + bottom[2]
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                # create mul node
                mul_node = nodes.create_axpy_mul_node(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(mul_node)

                # create add node
                add_node = nodes.create_axpy_add_node(Layers[i], node_name, input_name, output_name, input_shape)
                self.onnxNodeList.append(add_node)
            elif Layers[i].type == "Normalize":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                lp_normalization_output_name = [output_name[0] + "_lp"]
                lp_normalization_node = nodes.create_Lp_Normalization(
                    Layers[i],
                    node_name,
                    input_name,
                    lp_normalization_output_name,
                    input_shape,
                )
                self.onnxNodeList.append(lp_normalization_node)
                # get Normalize
                scale_shape, scale_data = self.GetParamsShapeAndData(Layers[i])
                scale_shape = [1, scale_shape[0][0], 1, 1]
                scale_input = self.AddInputsTVIFromParams(Layers[i], ["_scale"], [onnx.TensorProto.FLOAT])
                mul_input_name = [lp_normalization_output_name[0], node_name + "_scale"]
                mul_input_shape = [input_shape[0], scale_shape]
                mul_node = nodes.create_mul_node(
                    Layers[i],
                    node_name + "_mul",
                    mul_input_name,
                    output_name,
                    mul_input_shape,
                )
                self.onnxNodeList.append(mul_node)
            elif Layers[i].type == "Power":
                # Power: Mul + Add + Pow
                # create mul node
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                power, scale, shift = nodes.get_power_param(Layers[i])
                scale_node_name = self.AddInputsTVIMannul(
                    Layers[i],
                    ["_scale"],
                    [onnx.TensorProto.FLOAT],
                    [np.shape(scale)],
                    [scale],
                )
                mul_input_name = [input_name[0], scale_node_name[0]]
                mul_node = nodes.create_mul_node(
                    Layers[i],
                    node_name + "_mul",
                    mul_input_name,
                    [output_name[0] + "_mul"],
                    [input_shape[0], np.shape(power)],
                )
                self.onnxNodeList.append(mul_node)
                # create Add node
                shift_param_name = self.AddInputsTVIMannul(
                    Layers[i],
                    ["_shift"],
                    [onnx.TensorProto.FLOAT],
                    [np.shape(scale)],
                    [shift],
                )
                add_input_name = [output_name[0] + "_mul", shift_param_name[0]]
                add_node = nodes.create_add_node(
                    Layers[i],
                    node_name + "_add",
                    add_input_name,
                    [output_name[0] + "_add"],
                    [input_shape[0], np.shape(shift)],
                )
                self.onnxNodeList.append(add_node)

                # create Pow
                power_param_name = self.AddInputsTVIMannul(
                    Layers[i],
                    ["_param_power"],
                    [onnx.TensorProto.FLOAT],
                    [np.shape(power)],
                    [power],
                )
                power_input_name = [output_name[0] + "_add", power_param_name[0]]
                power_node = nodes.create_power_node(
                    Layers[i],
                    node_name + "_power",
                    power_input_name,
                    output_name,
                    [input_shape[0], np.shape(power)],
                )
                self.onnxNodeList.append(power_node)

            elif Layers[i].type == "TanH":
                # 1.获取节点输入名、输入维度、输出名、节点名
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])  # 获取输入名列表和输入形状
                output_name = self.GetCurrentLayerOutName(Layers[i])  # 获取输出名列表
                node_name = Layers[i].name

                # 2.构建tanh_node
                tanh_node = nodes.createTanh(Layers[i], node_name, input_name, output_name, input_shape)

                # 3.添加节点到节点列表
                self.onnxNodeList.append(tanh_node)

            elif Layers[i].type == "Crop":
                # Crop: Slice
                # create Slice node
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                starts, ends, axes = nodes.get_crop_param(Layers[i], input_shape)

                Crop_name = []
                Crop_name.append(input_name[0])

                starts_param = self.AddInputsTVIMannul(
                    Layers[i],
                    ["_starts" + str(i)],
                    [onnx.TensorProto.INT64],
                    [np.shape(starts)],
                    [starts],
                )
                ends_param = self.AddInputsTVIMannul(
                    Layers[i],
                    ["_ends" + str(i)],
                    [onnx.TensorProto.INT64],
                    [np.shape(ends)],
                    [ends],
                )
                axes_param = self.AddInputsTVIMannul(
                    Layers[i],
                    ["_axes" + str(i)],
                    [onnx.TensorProto.INT64],
                    [np.shape(axes)],
                    [axes],
                )

                Crop_name.extend(starts_param)
                Crop_name.extend(ends_param)
                Crop_name.extend(axes_param)
                crop_node = nodes.create_crop_node(Layers[i], node_name, Crop_name, output_name, input_shape)
                self.onnxNodeList.append(crop_node)

            # MVN
            elif Layers[i].type == "MVN":
                # MVN: InstanceNormalization
                # create InstanceNormalization
                if Layers[i].mvn_param.normalize_variance is False or Layers[i].mvn_param.across_channels is True:
                    print("Failed type not support: " + Layers[i].type)
                    exit(-1)

                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name

                MVN_name = []
                MVN_name.append(input_name[0])
                scale, bias = nodes.get_InstanceNorm_param(Layers[i], input_shape)

                scale_param = self.AddInputsTVIMannul(
                    Layers[i],
                    ["_scale" + str(i)],
                    [onnx.TensorProto.FLOAT],
                    [np.shape(scale)],
                    [scale],
                )
                bias_param = self.AddInputsTVIMannul(
                    Layers[i],
                    ["_bias" + str(i)],
                    [onnx.TensorProto.FLOAT],
                    [np.shape(bias)],
                    [bias],
                )

                MVN_name.extend(scale_param)
                MVN_name.extend(bias_param)
                MVN_node = nodes.create_InstanceNorm_op(Layers[i], node_name, MVN_name, output_name, input_shape)
                self.onnxNodeList.append(MVN_node)

            elif Layers[i].type == "Split":
                node_name = Layers[i].name
                assert len(Layers[i].bottom) == 1
                node_bottom = Layers[i].bottom[0]
                node_tops = Layers[i].top
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])

                # If a split branch is output of the model, relace it by a Identity Layer
                for top_name in node_tops:
                    is_output = True
                    for node in Layers:
                        if top_name in node.bottom:
                            is_output = False
                            break
                    if is_output:
                        id_node = nodes.createIdentity(Layers[i], top_name, input_name, [top_name], input_shape)
                        self.onnxNodeList.append(id_node)

                # remove split layer and repair the topology
                for j in range(i + 1, len(Layers)):
                    bottoms = []
                    for bottom in Layers[j].bottom:
                        if bottom in node_tops:
                            bottoms.append(node_bottom)
                        else:
                            bottoms.append(bottom)
                    del Layers[j].bottom[:]
                    Layers[j].bottom.extend(bottoms)

            elif Layers[i].type == "ArgMax":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                argmax_node = nodes.create_argmax_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(argmax_node)

            elif Layers[i].type == "Proposal":

                def reorder_input_meta(
                    input_names: list[str],
                    input_shapes: list[list[int]],
                    expected_input_names: list[str],
                ):
                    order = [input_names.index(i) for i in expected_input_names]
                    return [input_names[i] for i in order], [input_shapes[i] for i in order]

                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                input_name, input_shape = reorder_input_meta(input_name, input_shape, Layers[i].bottom)
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                proposal_node = nodes.create_proposal_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(proposal_node)

            elif Layers[i].type == "ROIAlign":

                def reorder_input_meta(input_names: list[str], input_shapes: list[list[int]]):
                    reorder_inputs = False
                    for elem in input_names:
                        if elem in self.model_input_name:
                            reorder_inputs = True
                    if len(input_shapes[0]) == 4 and len(input_shapes[1]) == 2 and input_shapes[1][-1] == 5:
                        reorder_inputs = False
                    if reorder_inputs:
                        order = [1, 0]
                        return [input_names[i] for i in order], [input_shapes[i] for i in order]
                    return input_names, input_shapes

                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                input_name, input_shape = reorder_input_meta(input_name, input_shape)
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                roi_align_node = nodes.create_roi_align_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(roi_align_node)

            elif Layers[i].type == "Reduction":

                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                reduction_node = nodes.create_reduction_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(reduction_node)

            elif Layers[i].type == "Tile":
                input_name, input_shape = self.GetLastLayerOutNameAndShape(Layers[i])
                # Convert Attributes to Inputs
                tile_attrs = nodes.get_tile_attr(Layers[i])
                repeats_datum = [1] * len(input_shape[0])
                repeats_datum[tile_attrs["axis"]] = tile_attrs["tiles"]
                repeats_data = [repeats_datum]
                repeats_shape = [np.shape(repeats_datum)]

                tile_name = self.AddInputsTVIMannul(
                    Layers[i],
                    op_layer_info.op_pname["Tile"],
                    op_layer_info.op_ptype["Tile"],
                    repeats_shape,
                    repeats_data,
                )

                input_name.extend(tile_name)

                output_name = self.GetCurrentLayerOutName(Layers[i])
                node_name = Layers[i].name
                tile_node = nodes.create_tile_node(Layers[i], node_name, input_name, output_name, input_shape)

                self.onnxNodeList.append(tile_node)

            else:
                print(f"Warning: Currently doesn't support ops: {Layers[i].type}, skipped.")
                # exit(-1)

    # 判断当前节点是否是输出节点
    def JudgeOutput(self, current_node, nodelist):
        for output_name in current_node.outputs_name:
            for node in nodelist:
                if output_name in node.inputs_name:
                    return False
        return True

    def DeleteUnusedInput(self):
        """
        删除没有被用到的输入
        """
        used_in_tvi = []
        for input in self.onnxmodel.in_tvi:
            used = False
            for node in self.onnxNodeList:
                if input.name in node.inputs_name:
                    used = True
            if used:
                used_in_tvi.append(input)
        self.onnxmodel.in_tvi = used_in_tvi
        return

    def FindPrevNode(self, cur_node, nodelist):
        cur_input_name = cur_node.inputs_name[0]
        for i, node in enumerate(nodelist):
            if cur_input_name in node.outputs_name:
                return i, node.node.output[0], node
        return None, None, None

    def RenameSameTopNodes(self, index, top_name, nodelist):
        """
        Rename the layers with same top name
            1. the last layer will be renamed to -> top
            2. other layers with same top will be renamed to -> top::layer_name
        """

        def rename_node(node, node_name, new_node_name, top_name, rename_input=False):
            node.node.name = new_node_name
            node.node.output[0] = new_node_name
            update_onnx_value_info(node_name, new_node_name)
            if rename_input:
                node.node.input[0] = top_name + "::" + node.node.input[0]
            print(
                "INFO: for keeping top name as output name, the node.name {} is changed to {}".format(
                    node_name, new_node_name
                )
            )
            return node

        def update_onnx_value_info(old_name, new_name):
            for i, value_info in enumerate(self.onnxmodel.hidden_out_tvi):
                if value_info.name == old_name:
                    self.onnxmodel.hidden_out_tvi[i].name = new_name

        cur_node = nodelist[index]
        rename_to_top = True
        if cur_node.top == cur_node.bottom:
            while cur_node.top == cur_node.bottom:
                if rename_to_top:
                    new_node_name = top_name
                    rename_to_top = False
                else:
                    new_node_name = top_name + "::" + cur_node.node.output[0]
                new_cur_node = rename_node(
                    cur_node, cur_node.node.output[0], new_node_name, top_name, rename_input=True
                )
                nodelist.pop(index)
                nodelist.insert(index, new_cur_node)
                index, prev_node_name, prev_node = self.FindPrevNode(cur_node, nodelist)
                assert prev_node is not None, "ERROR: failed to find node: {}".format(prev_node_name)
                cur_node = prev_node
            if cur_node.top != cur_node.bottom:
                new_node_name = top_name + "::" + cur_node.node.output[0]
                new_cur_node = rename_node(
                    cur_node, cur_node.node.output[0], new_node_name, top_name, rename_input=False
                )
                nodelist.pop(index)
                nodelist.insert(index, new_cur_node)
        else:
            new_node_name = top_name
            new_cur_node = rename_node(cur_node, cur_node.node.name, new_node_name, top_name, rename_input=False)
            nodelist.pop(index)
            nodelist.insert(index, new_cur_node)
        return nodelist

    # 添加模型输出信息和中间节点信息
    def AddOutputsTVIAndValueInfo(self):
        for i in range(len(self.onnxNodeList)):
            if self.JudgeOutput(self.onnxNodeList[i], self.onnxNodeList):  # 构建输出节点信息
                lastnode = self.onnxNodeList[i]
                for j in range(len(lastnode.outputs_shape)):
                    if lastnode.outputs_name[j] != lastnode.top[j] and lastnode.outputs_name[j] not in lastnode.top:
                        print(
                            f"INFO: Layer top and layer name are different, top={lastnode.top[j]} "
                            f"name={lastnode.outputs_name[j]}, the output name set to top: {lastnode.top[j]}"
                        )
                        output_name = lastnode.top[j]
                        self.onnxNodeList = self.RenameSameTopNodes(i, output_name, self.onnxNodeList)

                    else:
                        output_name = lastnode.outputs_name[j]

                    if lastnode.node.op_type == "ArgMax":  # Onnx ArgMin/ArgMax only supports int64 output
                        output_dtype = onnx.TensorProto.INT64
                    else:
                        output_dtype = onnx.TensorProto.FLOAT

                    output_tvi = helper.make_tensor_value_info(
                        output_name,
                        output_dtype,
                        lastnode.outputs_shape[j],
                    )
                    self.onnxmodel.addOutputsTVI(output_tvi)
            else:  # 构建中间节点信息
                innernode = self.onnxNodeList[i]
                for k in range(len(innernode.outputs_shape)):
                    hid_out_tvi = helper.make_tensor_value_info(
                        innernode.outputs_name[k],
                        onnx.TensorProto.FLOAT,
                        innernode.outputs_shape[k],
                    )
                    self.onnxmodel.addValueInfoTVI(hid_out_tvi)

    # 创建模型
    def polish_name(self, name):
        """
        polish_name.

        Nothing need to be done for now.
        """
        return name

    def createOnnxModel(self, opset_version: int = 16, ir_version: int = 8):
        node_def = [Node.node for Node in self.onnxNodeList]
        print("=============================ONNX Model=============================")
        for i in range(len(node_def)):
            node_def[i].name = self.polish_name(node_def[i].name)
            for j in range(len(node_def[i].input)):
                node_def[i].input[j] = self.polish_name(node_def[i].input[j])
            for j in range(len(node_def[i].output)):
                node_def[i].output[j] = self.polish_name(node_def[i].output[j])
            print("Node: ", node_def[i].name)
            print("OP Type: ", node_def[i].op_type)
            print("Input: ", node_def[i].input)
            print("Output: ", node_def[i].output)
            print("====================================================================")

        self.onnxmodel.name = self.polish_name(self.onnxmodel.name)
        for i in range(len(self.onnxmodel.in_tvi)):
            self.onnxmodel.in_tvi[i].name = self.polish_name(self.onnxmodel.in_tvi[i].name)
        for i in range(len(self.onnxmodel.out_tvi)):
            self.onnxmodel.out_tvi[i].name = self.polish_name(self.onnxmodel.out_tvi[i].name)
        for i in range(len(self.onnxmodel.init_t)):
            self.onnxmodel.init_t[i].name = self.polish_name(self.onnxmodel.init_t[i].name)
        for i in range(len(self.onnxmodel.hidden_out_tvi)):
            self.onnxmodel.hidden_out_tvi[i].name = self.polish_name(self.onnxmodel.hidden_out_tvi[i].name)

        graph_def = helper.make_graph(
            node_def,
            self.onnxmodel.name,
            self.onnxmodel.in_tvi,
            self.onnxmodel.out_tvi,
            self.onnxmodel.init_t,
            value_info=self.onnxmodel.hidden_out_tvi,
        )

        opset_imports = None
        if opset_version is not None:
            opset_imports = [
                helper.make_operatorsetid(domain=defs.ONNX_DOMAIN, version=opset_version),
                helper.make_operatorsetid(domain="ax.caffe2onnx", version=1),
            ]
        model_def = helper.make_model(graph_def, opset_imports=opset_imports, ir_version=ir_version)
        print("2. onnx model conversion done")
        return model_def
