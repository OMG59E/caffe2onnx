from onnx import TensorProto

from caffe2onnx.proto import caffe_pb2 as caffe

Layer_CONCAT = caffe.V1LayerParameter.CONCAT  # 3
Layer_CONVOLUTION = caffe.V1LayerParameter.CONVOLUTION  # 4
Layer_DROPOUT = caffe.V1LayerParameter.DROPOUT  # 6
Layer_INNER_PRODUCT = caffe.V1LayerParameter.INNER_PRODUCT  # 14
Layer_LRN = caffe.V1LayerParameter.LRN  # 15
Layer_POOLING = caffe.V1LayerParameter.POOLING  # 17
Layer_RELU = caffe.V1LayerParameter.RELU  # 18
Layer_SOFTMAX = caffe.V1LayerParameter.SOFTMAX  # 20
Layer_ELTWISE = caffe.V1LayerParameter.ELTWISE  # 25
Layer_ARGMAX = caffe.V1LayerParameter.ARGMAX  # 30
Layer_UPSAMPLE = caffe.V1LayerParameter.UPSAMPLE  # 40

op_pname = {
    "Conv": ["_W", "_b"],
    "BatchNorm": ["_mean", "_var"],
    "Scale": ["_scale", "_b"],
    "Reshape": ["_shape"],
    "DouReshape": ["_Doureshape"],
    "InnerProduct": ["_W", "_B"],
    "Upsample": ["_roi_", "_Scale"],
    "PRelu": ["_slope"],
    "Tile": ["_repeats"],
    "Transpose": ["_trans"],
    "ConvTranspose": ["_W", "_b"],
    "Slice": ["_starts", "_ends", "_axes", "_steps"],
    "LSTM": ["_W", "_B", "_R"],
}

op_ptype = {
    "Conv": [TensorProto.FLOAT, TensorProto.FLOAT],
    "BatchNorm": [TensorProto.FLOAT, TensorProto.FLOAT],
    "Scale": [TensorProto.FLOAT, TensorProto.FLOAT],
    "Reshape": [TensorProto.INT64],
    "InnerProduct": [TensorProto.FLOAT, TensorProto.FLOAT],
    "Upsample": [TensorProto.FLOAT, TensorProto.FLOAT],
    "PRelu": [TensorProto.FLOAT],
    "Tile": [TensorProto.INT64],
    "Transpose": [TensorProto.INT64],
    "ConvTranspose": [TensorProto.FLOAT, TensorProto.FLOAT],
    "DouReshape": [TensorProto.INT64],
    "Slice": [
        TensorProto.INT64,
        TensorProto.INT64,
        TensorProto.INT64,
        TensorProto.INT64,
    ],
    "LSTM": [
        TensorProto.FLOAT,
        TensorProto.FLOAT,
        TensorProto.FLOAT,
    ],
}
