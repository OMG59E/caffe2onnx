from caffe2onnx.graph import CaffeToOnnxNode


# 获取超参数
def getLSTMAttri(layer):
    hidden_size = layer.recurrent_param.num_output

    # TODO: Bi-LSTM
    direction = "forward"  # one of ["forward", "reverse", "bidirectional"]

    dict = {
        "hidden_size": hidden_size,
        "direction": direction,
    }

    return dict


# 计算输出维度
def inferLSTMNameAndShape(input_name, input_shape, output_name, layer, dict):
    hidden_size = layer.recurrent_param.num_output
    expose_hidden = layer.recurrent_param.expose_hidden

    is_bidirection = dict["direction"] == "bidirectional"
    if is_bidirection:
        raise Exception("Bidirectional LSTM not implemented!")
    num_direction = 2 if is_bidirection else 1

    # lstm输入应该按照onnx的输入接口排序，即[X, W, R, B, h, c]
    sorted_input_shape = []
    sorted_input_name = []

    # 寻找X
    sorted_input_shape.append(input_shape[0])
    sorted_input_name.append(input_name[0])
    seq_length = input_shape[0][0]
    if seq_length == 0:
        seq_length = 1
    batch_size = input_shape[0][1]

    # 寻找W,B,R
    for i in range(len(input_name)):
        if "_W" in input_name[i]:
            sorted_input_name.append(input_name[i])
            sorted_input_shape.append(input_shape[i])
    for i in range(len(input_name)):
        if "_R" in input_name[i]:
            sorted_input_name.append(input_name[i])
            sorted_input_shape.append(input_shape[i])
    for i in range(len(input_name)):
        if "_B" in input_name[i]:
            sorted_input_name.append(input_name[i])
            sorted_input_shape.append(input_shape[i])

    # 增加空的 sequence_lens
    if len(input_name) == 6:
        sorted_input_name.append("")
        sorted_input_shape.append([batch_size])
        # 增加 h, c
        for i in range(-2, 0):
            sorted_input_shape.append(input_shape[i])
            sorted_input_name.append(input_name[i])

    output_shape = [[seq_length, num_direction, batch_size, hidden_size]]
    suffixed_output_name = [
        output_name[0] + "_lstm"
    ]  # lstm outname shold be suffixed due to the following reshape layer

    if expose_hidden:
        output_shape.extend([[1, batch_size, hidden_size] for i in range(2)])
        suffixed_output_name = suffixed_output_name + output_name[1:]

    return sorted_input_name, sorted_input_shape, suffixed_output_name, output_shape


# 构建节点
def createLSTM(layer, node_name, input_name, output_name, input_shape):
    attributes = getLSTMAttri(layer)
    input_name, input_shape, output_name, output_shape = inferLSTMNameAndShape(
        input_name, input_shape, output_name, layer, attributes
    )
    # 构建node
    node = CaffeToOnnxNode(
        layer,
        node_name,
        "LSTM",
        input_name,
        output_name,
        input_shape,
        output_shape,
        attributes,
    )
    return node, output_name, output_shape
