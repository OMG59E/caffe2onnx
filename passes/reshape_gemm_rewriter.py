import numpy as np

import caffe2onnx.graphsurgeon as gs

from .base_rewriter import BaseRewriter


class ReshapeGemmRewriter(BaseRewriter):
    def __init__(self, graph, verbose: bool = False):
        super().__init__(graph, verbose)

    def reshape_gemm_detected(self, node):
        if (
            node.op == "Reshape"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 1
            and node.o().op == "Gemm"
            and len(node.o().outputs[0].outputs) == 1
            and node.o().o().op != "Reshape"
            and node.inputs[1].values[0] != -1
            and node.inputs[1].values[1] != -1
        ):
            node_pre_reshape = node
            return True, node_pre_reshape

        return False, None

    def reshape_dynamic_batch_compatible(self, node_pre_reshape, inner_product_idx):
        if node_pre_reshape.inputs[1].values[0] != -1:
            shape_pre = node_pre_reshape.inputs[1].values
            node_pre_reshape_new_shape = gs.Constant(
                f"{node_pre_reshape.name}_dynamic{inner_product_idx}",
                values=np.array([-1, shape_pre[1]]),
            )
            node_pre_reshape.inputs[1] = node_pre_reshape_new_shape

        self.cleanup()

    def support_dynamic_batch(self, inner_product_idx):
        for node in self.graph.nodes:
            detected, node_pre_reshape = self.reshape_gemm_detected(node)

            if detected:
                self.reshape_dynamic_batch_compatible(node_pre_reshape, inner_product_idx)
                assert node_pre_reshape.inputs[1].values[0] == -1
                assert node_pre_reshape.inputs[1].values[1] != -1
                return True
        return False

    def rewrite_reshape_gemm(self):
        inner_product_idx = 0
        while self.support_dynamic_batch(inner_product_idx):
            inner_product_idx += 1
        return inner_product_idx
