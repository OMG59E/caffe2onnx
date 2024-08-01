from typing import Dict, List

import onnx

from caffe2onnx.graph import CaffeToOnnxNode


def get_attributes(layer, inputs_shape, proto_path) -> Dict:
    st_param = layer.st_param
    if st_param.output_H == 0 or st_param.output_W == 0:
        output_h = inputs_shape[0][-2]
        output_w = inputs_shape[0][-1]
    else:
        output_h = st_param.output_H
        output_w = st_param.output_W

    theta = {}
    theta_names = ["theta_1_1", "theta_1_2", "theta_1_3", "theta_2_1", "theta_2_2", "theta_2_3"]
    with open(proto_path, "r") as proto:
        for l in proto.readlines():
            l = l.strip()
            for theta_name in theta_names:
                if l.startswith(theta_name):
                    val = float(l.split(":")[1])
                    theta[theta_name] = val
                    break
    attributes = dict(
        domain="ax.caffe2onnx",
        output_h=output_h,
        output_w=output_w,
    )
    attributes.update(theta)
    return attributes


def get_output_shape(input_shape, attributes):
    output_shape = [
        input_shape[0][0],
        input_shape[0][1],
        attributes["output_h"],
        attributes["output_w"],
    ]
    return [output_shape]


def create_spatial_transformer(
    layer,
    node_name: str,
    inputs_name: List[str],
    outputs_name: List[str],
    inputs_shape: List,
    proto_path: str,
) -> onnx.NodeProto:

    attributes = get_attributes(layer, inputs_shape, proto_path)

    outputs_shape = get_output_shape(inputs_shape, attributes)

    node = CaffeToOnnxNode(
        layer,
        node_name,
        "SpatialTransformer",
        inputs_name,
        outputs_name,
        inputs_shape,
        outputs_shape,
        attributes,
    )
    return node
