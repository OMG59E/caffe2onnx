from caffe2onnx.graph import CaffeToOnnxNode


def get_roi_align_attributes(layer):
    return {
        "pooled_h": layer.roi_align_param.pooled_h,
        "pooled_w": layer.roi_align_param.pooled_w,
        "spatial_scale": layer.roi_align_param.spatial_scale,
        "sampling_ratio": layer.roi_align_param.sampling_ratio,
        "domain": "ax.caffe2onnx",
    }


def get_roi_align_output_shape(input_shape, output_name, attrs):
    output_shape = []
    output_shape.append([input_shape[1][0], input_shape[0][1], attrs["pooled_h"], attrs["pooled_w"]])
    return output_shape


def create_roi_align_node(layer, node_name, input_name, output_name, input_shape):
    attrs = get_roi_align_attributes(layer)
    output_shape = get_roi_align_output_shape(input_shape, output_name, attrs)
    node = CaffeToOnnxNode(
        layer,
        node_name,
        "ROIAlign",
        input_name,
        output_name,
        input_shape,
        output_shape,
        attrs,
    )

    return node
