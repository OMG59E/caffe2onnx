import numpy as np

import caffe2onnx.graphsurgeon as gs

from .base_rewriter import BaseRewriter


class ExpAddLogRewriter(BaseRewriter):
    def __init__(self, graph, verbose: bool = False):
        super().__init__(graph, verbose)

    def exp_add_log_detected(self, node):
        if (
            node.op == "Exp"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 1
            and node.o().op == "Add"
            and len(node.o().inputs) == 2
            and isinstance(node.o().inputs[1], gs.Constant)
            and node.o().inputs[1].values == 1
            and len(node.o().outputs) == 1
            and len(node.o().outputs[0].outputs) == 1
            and node.o().o().op == "Log"
        ):
            node_exp = node
            node_log = node.o().o()
            return True, node_exp, node_log

        return False, None, None

    def replace_softplus(self, node_exp, node_log, fused_softplus_idx):
        softplus_out = gs.Variable(
            f"fused_softplus{fused_softplus_idx}_out",
            dtype=np.float32,
            shape=node_exp.inputs[0].shape,
        )
        softplus_node = gs.Node(
            op="Softplus",
            name=f"fused_softplus_{fused_softplus_idx}",
            attrs={},
            inputs=node_exp.inputs,
            outputs=[softplus_out],
        )
        self.graph.nodes.append(softplus_node)

        if len(node_log.outputs[0].outputs) > 0:
            output_name = None
            output_index = -1
            node_log.o().inputs[0] = softplus_out
        else:
            output_name = node_log.name
            output_index = -1
            for i, out in enumerate(self.graph.outputs):
                if out.name == output_name:
                    output_index = i
            self.graph.outputs[output_index] = softplus_out
        self.cleanup()

        # Post processing for Node names
        if output_name and output_index != -1:
            softplus_node.name = output_name
            self.graph.outputs[output_index].name = output_name

    def fuse_softplus(self, fused_softplus_idx):
        for node in self.graph.nodes:
            # Get Exp-Add-Log node for fusion
            detected, node_exp, node_log = self.exp_add_log_detected(node)

            if detected:
                self.replace_softplus(node_exp, node_log, fused_softplus_idx)
                return True
        return False

    def rewrite_softplus(self):
        fused_softplus_idx = 0
        while self.fuse_softplus(fused_softplus_idx):
            fused_softplus_idx += 1
        return fused_softplus_idx
