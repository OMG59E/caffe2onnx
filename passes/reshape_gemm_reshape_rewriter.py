import math

import numpy as np

import caffe2onnx.graphsurgeon as gs

from .base_rewriter import BaseRewriter


class ReshapeGemmReshapeRewriter(BaseRewriter):
    def __init__(self, graph, verbose: bool = False):
        super().__init__(graph, verbose)

    def reshape_gemm_reshape_detected(self, node):
        if (
            node.op == "Reshape"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 1
            and node.o().op == "Gemm"
            and len(node.o().outputs[0].outputs) == 1
            and node.o().o().op == "Reshape"
            and len(node.inputs[0].shape) in [3, 4]
            and len(node.o().o().outputs[0].shape) == 3
            and node.o().attrs.get("transA", 0) == 0
            and node.o().attrs.get("transB", 0) == 1
        ):
            node_pre_reshape = node
            node_gemm = node.o()
            node_post_reshape = node.o().o()
            return True, node_pre_reshape, node_gemm, node_post_reshape

        return False, None, None, None

    def replace_gemm(self, node_pre_reshape, node_gemm, node_post_reshape, fused_matmul_idx):

        fused_matmul_in = node_pre_reshape.inputs[0]
        if len(fused_matmul_in.shape) == 4:
            reshape_shape = gs.Constant(
                f"fused_matmul_{fused_matmul_idx}_pre_reshape",
                values=np.array([fused_matmul_in.shape[0], -1, fused_matmul_in.shape[3]], dtype=np.int64),
            )

            reshape_out = gs.Variable(
                f"fused_matmul_{fused_matmul_idx}_reshape_out",
                dtype=np.float32,
                shape=(fused_matmul_in.shape[0], math.prod(fused_matmul_in.shape[1:-1]), fused_matmul_in.shape[-1]),
            )

            reshape_node = gs.Node(
                op="Reshape",
                name=f"fused_matmul_{fused_matmul_idx}_reshape",
                inputs=[fused_matmul_in, reshape_shape],
                outputs=[reshape_out],
            )
            self.graph.nodes.append(reshape_node)
            fused_matmul_in = reshape_out

        multiplier = gs.Constant(f"fused_matmul_{fused_matmul_idx}_W", values=node_gemm.inputs[1].values.T)

        matmul_out = gs.Variable(
            f"fused_matmul_{fused_matmul_idx}_matmul_out",
            dtype=np.float32,
            shape=(*fused_matmul_in.shape[:2], multiplier.shape[1]),
        )
        matmul_node = gs.Node(
            op="MatMul",
            name=f"fused_matmul_{fused_matmul_idx}",
            inputs=[fused_matmul_in, multiplier],
            outputs=[matmul_out],
        )
        self.graph.nodes.append(matmul_node)
        last_node_out = matmul_out

        if len(node_gemm.inputs) > 2:
            added_operand = gs.Constant(f"fused_matmul_{fused_matmul_idx}_B", values=node_gemm.inputs[2].values)

            add_out = gs.Variable(
                f"fused_matmul_{fused_matmul_idx}_add_out",
                dtype=np.float32,
                shape=matmul_out.shape,
            )
            add_node = gs.Node(
                op="Add",
                name=f"fused_matmul_{fused_matmul_idx}_add",
                attrs={},
                inputs=[matmul_out, added_operand],
                outputs=[add_out],
            )
            self.graph.nodes.append(add_node)
            last_node_out = add_out

        if len(node_post_reshape.outputs[0].outputs) > 0:
            output_name = None
            output_index = -1
            node_post_reshape.o().inputs[0] = last_node_out
        else:
            output_name = node_post_reshape.outputs[0].name
            output_index = -1
            for i, out in enumerate(self.graph.outputs):
                if out.name == output_name:
                    output_index = i
            self.graph.outputs[output_index] = last_node_out

        self.cleanup()

        # Post processing for Node names
        if output_name and output_index != -1:
            add_node.name = output_name
            self.graph.outputs[output_index].name = output_name

    def fuse_matmul(self, fused_matmul_idx):
        for node in self.graph.nodes:
            # Get Add node for eliminating
            detected, node_pre_reshape, node_gemm, node_post_reshape = self.reshape_gemm_reshape_detected(node)

            if detected:
                self.replace_gemm(node_pre_reshape, node_gemm, node_post_reshape, fused_matmul_idx)
                return True
        return False

    def rewrite_reshape_gemm_reshape(self):
        fused_matmul_idx = 0
        while self.fuse_matmul(fused_matmul_idx):
            fused_matmul_idx += 1
        return fused_matmul_idx
