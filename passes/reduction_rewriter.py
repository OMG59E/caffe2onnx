from collections import OrderedDict

import numpy as np

import caffe2onnx.graphsurgeon as gs

from .base_rewriter import BaseRewriter


class ReductionRewriter(BaseRewriter):
    def __init__(self, graph, verbose: bool = False):
        super().__init__(graph, verbose)

    def caffe_roi_align_detected(self, node):
        if node.op == "Reduction" and node.domain == "ax.caffe2onnx":
            return True, node

        return False, None

    def rewrite_onnx_reduction(self, node_caffe_reduction, reduction_idx):
        node_caffe_reduction.domain = ""

        caffe_attrs = node_caffe_reduction.attrs

        operation = caffe_attrs["operation"]
        if operation == 1:
            node_caffe_reduction.op = "ReduceSum"
        elif operation == 2:
            node_caffe_reduction.op = "ReduceL1"
        elif operation == 3:
            node_caffe_reduction.op = "ReduceSumSquare"
        elif operation == 4:
            node_caffe_reduction.op = "ReduceMean"
        else:
            raise NotImplementedError(f"Currently doesn't support operation: '{operation}' in Caffe Reduction.")

        onnx_attrs = OrderedDict()
        onnx_attrs["keepdims"] = 0
        node_caffe_reduction.attrs = onnx_attrs
        if operation != 1:
            onnx_attrs["axes"] = [caffe_attrs["axis"]]
        else:
            node_caffe_reduction.inputs.append(
                gs.Constant(
                    f"caffe_reduction{reduction_idx}_axes",
                    values=np.array([caffe_attrs["axis"]], dtype=np.int64),
                ),
            )

        scale = caffe_attrs["coeff"]
        if not np.isclose(scale, 1):
            # Mul Node
            mul_constant = gs.Constant(
                f"caffe_reduction{reduction_idx}_mul",
                values=np.array([scale], dtype=np.float32),
            )
            mul_out = gs.Variable(
                f"caffe_reduction{reduction_idx}_mul_out",
                dtype=np.float32,
                shape=node_caffe_reduction.outputs[0].shape,
            )
            mul_node = gs.Node(
                op="Mul",
                name=f"caffe_reduction{reduction_idx}_mul",
                attrs={},
                inputs=[node_caffe_reduction.outputs[0], mul_constant],
                outputs=[mul_out],
            )
            self.graph.nodes.append(mul_node)

            reduction_origin_output_name = node_caffe_reduction.o().inputs[0].name
            output_index = -1
            for i, out in enumerate(self.graph.outputs):
                if out.name == reduction_origin_output_name:
                    output_index = i
            if output_index < 0:
                node_caffe_reduction.o().inputs[0] = mul_out
            else:
                self.graph.outputs[output_index] = mul_out
                node_caffe_reduction.outputs[0].name = f"caffe_reduction{reduction_idx}_reduction_out"
                mul_out.name = reduction_origin_output_name

        self.cleanup()

    def rewriting_caffe_reduction(self, reduction_idx):
        for node in self.graph.nodes:
            # Get Caffe Reduction node
            detected, node_caffe_reduction = self.caffe_roi_align_detected(node)

            if detected:
                self.rewrite_onnx_reduction(node_caffe_reduction, reduction_idx)
                return True
        return False

    def rewrite_reduction(self):
        reduction_idx = 0
        while self.rewriting_caffe_reduction(reduction_idx):
            reduction_idx += 1
        return reduction_idx
