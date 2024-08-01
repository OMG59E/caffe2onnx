from google.protobuf import text_format

from caffe2onnx.proto import caffe_pb2 as caffe

__all__ = ["load_caffe_model", "has_unregistered_op"]


def load_caffe_model(net_path, model_path):
    # read prototxt
    net = caffe.NetParameter()
    with open(net_path, "rb") as f:
        text_format.Merge(f.read(), net)
    # read caffemodel
    model = caffe.NetParameter()
    with open(model_path, "rb") as f:
        model.ParseFromString(f.read())
    return net, model


def has_unregistered_op(proto_path):
    proto_file = open(proto_path, "r")
    lines = proto_file.read()
    proto_file.close()
    flag = False
    unregistered_op = [
        "PriorBox",
        "DetectionOutput",
        "SpatialTransformer",
    ]
    for op in unregistered_op:
        if op in lines:
            flag = True
    return flag
