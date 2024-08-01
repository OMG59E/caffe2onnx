import numpy as np

import caffe2onnx.graphsurgeon as gs

from .base_rewriter import BaseRewriter


class TransposeReshapeLSTMReshapeRewriter(BaseRewriter):
    def __init__(self, graph, verbose: bool = False):
        super().__init__(graph, verbose)

    def transpose_reshape_lstm_reshape_detected(self, node):
        if (
            node.op == "Transpose"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 1
            and node.o().op == "Reshape"
            and len(node.o().outputs[0].outputs) == 1
            and node.o().o().op == "LSTM"
            and node.o().o().outputs[0].shape[1] == 1  # num_directions
            and node.o().o().o().op == "Reshape"
            and isinstance(node.o().o().o().inputs[1], gs.Constant)
            and node.o().o().o().inputs[1].values.size == 3
        ):
            node_pre_reshape = node.o()
            node_post_reshape = node.o().o().o()
            if np.prod(node_pre_reshape.inputs[1].values) > 0 or node_post_reshape.inputs[1].values[1] != -1:
                return True, node_pre_reshape, node_post_reshape

        return False, None, None

    def reshape_dynamic_batch_compatible(self, node_pre_reshape, node_post_reshape, lstm_idx):
        if np.prod(node_pre_reshape.inputs[1].values) > 0:
            shape_pre = node_pre_reshape.inputs[1].values
            node_pre_reshape_new_shape = gs.Constant(
                f"{node_pre_reshape.name}_dynamic{lstm_idx}",
                values=np.array([shape_pre[0], 0, -1]),
            )
            node_pre_reshape.inputs[1] = node_pre_reshape_new_shape

        if node_post_reshape.inputs[1].values[1] != -1:
            shape_post = node_post_reshape.inputs[1].values
            node_post_reshape_new_shape = gs.Constant(
                f"{node_post_reshape.name}_dynamic{lstm_idx}",
                values=np.array([shape_post[0], -1, shape_post[-1]]),
            )
            node_post_reshape.inputs[1] = node_post_reshape_new_shape
        self.cleanup()

    def support_dynamic_batch(self, lstm_idx):
        for node in self.graph.nodes:
            # Get Add node for eliminating
            detected, node_pre_reshape, node_post_reshape = self.transpose_reshape_lstm_reshape_detected(node)

            if detected:
                self.reshape_dynamic_batch_compatible(node_pre_reshape, node_post_reshape, lstm_idx)
                assert np.prod(node_pre_reshape.inputs[1].values) <= 0
                assert node_post_reshape.inputs[1].values[1] == -1
                return True
        return False

    def rewrite_transpose_reshape_lstm_reshape(self):
        lstm_idx = 0
        while self.support_dynamic_batch(lstm_idx):
            lstm_idx += 1
        return lstm_idx
