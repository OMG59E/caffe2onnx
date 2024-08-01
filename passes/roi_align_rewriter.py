from collections import OrderedDict

import numpy as np

import caffe2onnx.graphsurgeon as gs

from .base_rewriter import BaseRewriter


class RoiAlignRewriter(BaseRewriter):
    def __init__(self, graph, verbose: bool = False):
        super().__init__(graph, verbose)

    def caffe_roi_align_detected(self, node):
        if (
            node.op == "ROIAlign"
            and node.domain == "ax.caffe2onnx"
            and len(node.inputs) == 2
            and len(node.outputs) == 1
        ):
            return True, node

        return False, None

    def rewrite_onnx_roi_align(self, node_caffe_roi_align, roi_align_idx):
        num_rois = node_caffe_roi_align.inputs[1].shape[0]
        dim_rois = len(node_caffe_roi_align.inputs[1].shape)
        slice_axes = dim_rois - 1
        slice_shape = node_caffe_roi_align.inputs[1].shape[:-1]

        rois_out = gs.Variable(f"rois{roi_align_idx}_out", dtype=np.float32, shape=[*slice_shape, 4])
        starts = gs.Constant(name=f"rois_starts{roi_align_idx}", values=np.array([1], dtype=np.int64))
        ends = gs.Constant(name=f"rois_ends{roi_align_idx}", values=np.array([5], dtype=np.int64))
        axes = gs.Constant(name=f"rois_axes{roi_align_idx}", values=np.array([slice_axes], dtype=np.int64))
        steps = gs.Constant(name=f"rois_steps{roi_align_idx}", values=np.array([1], dtype=np.int64))

        rois_slice_node = gs.Node(
            op="Slice",
            name=f"rois_slice{roi_align_idx}",
            inputs=[node_caffe_roi_align.inputs[1], starts, ends, axes, steps],
            outputs=[rois_out],
        )
        self.graph.nodes.append(rois_slice_node)
        rois_last_out = rois_out

        if slice_axes > 1:
            rois_squeeze_out = gs.Variable(f"rois_squeeze{roi_align_idx}_out", dtype=np.float32, shape=[num_rois, 4])
            squeeze_axes_values = np.array([1, 2], dtype=np.int64)
            squeeze_axes = gs.Constant(name=f"rois_squeeze_axes{roi_align_idx}", values=squeeze_axes_values)
            rois_squeeze_node = gs.Node(
                op="Squeeze",
                name=f"rois_squeeze{roi_align_idx}",
                inputs=[rois_out, squeeze_axes],
                outputs=[rois_squeeze_out],
            )
            self.graph.nodes.append(rois_squeeze_node)
            rois_last_out = rois_squeeze_out

        batch_indices_out = gs.Variable(f"batch_indices{roi_align_idx}_out", dtype=np.float32, shape=[*slice_shape, 1])
        starts = gs.Constant(name=f"batch_indices_starts{roi_align_idx}", values=np.array([0], dtype=np.int64))
        ends = gs.Constant(name=f"batch_indices_ends{roi_align_idx}", values=np.array([1], dtype=np.int64))
        axes = gs.Constant(name=f"batch_indices_axes{roi_align_idx}", values=np.array([slice_axes], dtype=np.int64))
        steps = gs.Constant(name=f"batch_indices_steps{roi_align_idx}", values=np.array([1], dtype=np.int64))

        batch_indices_slice_node = gs.Node(
            op="Slice",
            name=f"batch_indices_slice{roi_align_idx}",
            inputs=[node_caffe_roi_align.inputs[1], starts, ends, axes, steps],
            outputs=[batch_indices_out],
        )
        self.graph.nodes.append(batch_indices_slice_node)

        batch_indices_squeeze_out = gs.Variable(
            f"batch_indices_squeeze{roi_align_idx}_out",
            dtype=np.float32,
            shape=[num_rois],
        )
        squeeze_axes_values = np.array([1], dtype=np.int64) if slice_axes == 1 else np.array([1, 2, 3], dtype=np.int64)
        squeeze_axes = gs.Constant(name=f"batch_indices_squeeze_axes{roi_align_idx}", values=squeeze_axes_values)
        batch_indices_squeeze_node = gs.Node(
            op="Squeeze",
            name=f"batch_indices_squeeze{roi_align_idx}",
            inputs=[batch_indices_out, squeeze_axes],
            outputs=[batch_indices_squeeze_out],
        )
        self.graph.nodes.append(batch_indices_squeeze_node)

        cast_out = gs.Variable(f"batch_indices_cast{roi_align_idx}_out", dtype=np.int64, shape=[num_rois])
        batch_indices_cast_node = gs.Node(
            op="Cast",
            name=f"batch_indices_cast{roi_align_idx}",
            attrs={"to": 7},
            inputs=[batch_indices_squeeze_out],
            outputs=[cast_out],
        )
        self.graph.nodes.append(batch_indices_cast_node)

        node_onnx_roi_align_attrs = OrderedDict()
        node_onnx_roi_align_attrs["output_height"] = node_caffe_roi_align.attrs.get("pooled_h")
        node_onnx_roi_align_attrs["output_width"] = node_caffe_roi_align.attrs.get("pooled_w")
        node_onnx_roi_align_attrs["spatial_scale"] = node_caffe_roi_align.attrs.get("spatial_scale", 1.0)
        node_onnx_roi_align_attrs["sampling_ratio"] = node_caffe_roi_align.attrs.get("sampling_ratio", 2)
        node_onnx_roi_align_attrs["coordinate_transformation_mode"] = "output_half_pixel"

        roi_align_node = gs.Node(
            op="RoiAlign",
            name=f"roi_align{roi_align_idx}",
            attrs=node_onnx_roi_align_attrs,
            inputs=[node_caffe_roi_align.inputs[0], rois_last_out, cast_out],
            outputs=node_caffe_roi_align.outputs,
        )
        self.graph.nodes.append(roi_align_node)

        node_caffe_roi_align.outputs[0].inputs.pop(0)
        self.cleanup()

    def rewriting_caffe_roi_align(self, roi_align_idx):
        for node in self.graph.nodes:
            # Get Caffe ROIAlign node
            detected, node_caffe_roi_align = self.caffe_roi_align_detected(node)

            if detected:
                self.rewrite_onnx_roi_align(node_caffe_roi_align, roi_align_idx)
                return True
        return False

    def rewrite_roi_align(self):
        roi_align_idx = 0
        while self.rewriting_caffe_roi_align(roi_align_idx):
            roi_align_idx += 1
        return roi_align_idx
