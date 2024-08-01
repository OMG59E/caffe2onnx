from caffe2onnx.graph import CaffeToOnnxNode


def get_proposal_attributes(layer):
    return {
        "feat_stride": layer.proposal_param.feat_stride,
        "base_size": layer.proposal_param.base_size,
        "min_size": layer.proposal_param.min_size,
        "ratio": layer.proposal_param.ratio,
        "scale": layer.proposal_param.scale,
        "pre_nms_topn": layer.proposal_param.pre_nms_topn,
        "post_nms_topn": layer.proposal_param.post_nms_topn,
        "nms_thresh": layer.proposal_param.nms_thresh,
        "domain": "ax.caffe2onnx",
    }


def get_proposal_output_shape(input_shape, output_name, attrs):
    output_shape = []
    post_nms_topn = attrs["post_nms_topn"]
    output_shape.append([post_nms_topn, 5])
    if len(output_name) == 2:
        output_shape.append([post_nms_topn])
    return output_shape


def create_proposal_node(layer, node_name, input_name, output_name, input_shape):
    attrs = get_proposal_attributes(layer)
    if len(output_name) > 2:
        raise AssertionError("Please make sure that your proposal layer has at most two outputs.")
    if len(output_name) == 1:
        output_name.append("scores")
    output_shape = get_proposal_output_shape(input_shape, output_name, attrs)
    node = CaffeToOnnxNode(
        layer,
        node_name,
        "Proposal",
        input_name,
        output_name,
        input_shape,
        output_shape,
        attrs,
    )

    return node
