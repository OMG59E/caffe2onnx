import math
from collections import OrderedDict

import numpy as np

import caffe2onnx.graphsurgeon as gs

from .base_rewriter import BaseRewriter


class BiLSTMRewriter(BaseRewriter):
    def __init__(self, graph, verbose: bool = False):
        super().__init__(graph, verbose)

    def bilstm_detected(self, node):
        if (
            node.op == "Reshape"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 2
            and node.o().op == "Slice"
            and node.o(1).op == "Concat"
            and node.o(1).o().op == "LSTM"
            and node.o(1).o().o().op == "Reshape"
            and node.o(1).o().o().o().op == "Slice"
            and len(node.o(1).o().o().outputs[0].outputs) == 2
            and node.o(1).o().o().o(1).op == "Slice"
        ):
            node_reshape = node
            node_lstm = node.o(1).o()
            # Two Slice on the left
            if node.o(1).o().o().o().o().op == "Slice" and node.o(1).o().o().o().o().o().op == "Concat":
                node_concat = node.o(1).o().o().o().o().o()
                return True, node_reshape, node_lstm, node_concat
            # Two Slice on the right
            if node.o(1).o().o().o().o().op == "Concat" and node.o(1).o().o().o(1).o().op == "Slice":
                node_concat = node.o(1).o().o().o().o()
                return True, node_reshape, node_lstm, node_concat

        return False, None, None, None

    def replace_bilstm(self, node_start, node_lstm, node_end, fused_bilstm_prefix, fused_bilstm_idx):

        bilstm_attrs = node_lstm.attrs.copy()
        bilstm_attrs["direction"] = "bidirectional"

        input_shape = node_lstm.inputs[0].shape
        input_shape[1] = 1
        X = gs.Variable(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_X",
            dtype=np.float32,
            shape=input_shape,
        )
        W = gs.Constant(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_W",
            values=np.repeat(node_lstm.inputs[1].values, 2, axis=0),
        )
        R = gs.Constant(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_R",
            values=np.repeat(node_lstm.inputs[2].values, 2, axis=0),
        )
        B = gs.Constant(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_B",
            values=np.repeat(node_lstm.inputs[3].values, 2, axis=0),
        )
        output_shape = node_lstm.outputs[0].shape
        output_shape[1], output_shape[2] = output_shape[2], output_shape[1]
        Y = gs.Variable(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_Y",
            dtype=np.float32,
            shape=output_shape,
        )

        # Created a fused node
        fused_bilstm_node = gs.Node(
            op="LSTM",
            name=f"{fused_bilstm_prefix}_{fused_bilstm_idx}",
            attrs=bilstm_attrs,
            inputs=[X, W, R, B],
            outputs=[Y],
        )

        self.append_bilstm_transpose_reshape_cleanup(
            node_start,
            node_end,
            fused_bilstm_node,
            fused_bilstm_prefix,
            fused_bilstm_idx,
        )

    def fuse_bilstm(self, fused_bilstm_idx):
        fused_bilstm_prefix = "fused_bilstm"
        for node in self.graph.nodes:
            # Get nodes for fusion and bidirectional LSTM insertion if the forward LSTM is detected
            detected, node_start, node_lstm, node_end = self.bilstm_detected(node)

            if detected:
                self.replace_bilstm(
                    node_start,
                    node_lstm,
                    node_end,
                    fused_bilstm_prefix,
                    fused_bilstm_idx,
                )
                return True
        return False

    def bilstm2_detected(self, node):
        if (
            node.op == "Transpose"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 2
            and node.o(1).op == "LSTM"
            and node.o(1).o().op == "Reshape"
            and node.o(1).o().o().op == "Concat"
            and node.o().op == "Slice"
            and node.o().o().op == "LSTM"
            and node.o().o().o().op == "Reshape"
            and node.o().o().o().o().op == "Slice"
            and node.o().o().o().o().o().op == "Concat"
        ):
            node_start = node
            node_lstm1 = node.o(1)
            node_lstm2 = node.o().o()
            node_end = node.o().o().o().o().o()
            return True, node_start, node_lstm1, node_lstm2, node_end

        if (
            node.op == "Transpose"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 2
            and node.o(0).op == "LSTM"
            and node.o(0).o().op == "Reshape"
            and node.o(0).o().o().op == "Concat"
            and node.o(1).op == "Slice"
            and node.o(1).o().op == "LSTM"
            and node.o(1).o().o().op == "Reshape"
            and node.o(1).o().o().o().op == "Slice"
            and node.o(1).o().o().o().o().op == "Concat"
        ):
            node_start = node
            node_lstm1 = node.o(0)
            node_lstm2 = node.o(1).o()
            node_end = node.o(1).o().o().o().o()
            return True, node_start, node_lstm1, node_lstm2, node_end

        if (
            node.op == "Reshape"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 2
            and node.o().op == "LSTM"
            and node.o().o().op == "Reshape"
            and node.o().o().o().op == "Concat"
            and node.o(1).op == "Slice"
            and node.o(1).o().op == "LSTM"
            and node.o(1).o().o().op == "Reshape"
            and node.o(1).o().o().o().op == "Slice"
            and node.o(1).o().o().o().o().op == "Concat"
        ):
            node_start = node
            node_lstm1 = node.o()
            node_lstm2 = node.o(1).o()
            node_end = node.o().o().o()
            return True, node_start, node_lstm1, node_lstm2, node_end

        return False, None, None, None, None

    def replace_bilstm2(self, node_start, node_lstm1, node_lstm2, node_end, fused_bilstm_prefix, fused_bilstm_idx):

        bilstm_attrs = node_lstm1.attrs.copy()
        bilstm_attrs["direction"] = "bidirectional"

        X = gs.Variable(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_X",
            dtype=np.float32,
            shape=node_lstm1.inputs[0].shape,
        )
        W = gs.Constant(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_W",
            values=np.concatenate((node_lstm1.inputs[1].values, node_lstm2.inputs[1].values), axis=0),
        )
        R = gs.Constant(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_R",
            values=np.concatenate((node_lstm1.inputs[2].values, node_lstm2.inputs[2].values), axis=0),
        )
        B = gs.Constant(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_B",
            values=np.concatenate((node_lstm1.inputs[3].values, node_lstm2.inputs[3].values), axis=0),
        )
        Y = gs.Variable(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_Y",
            dtype=np.float32,
            shape=(node_lstm1.outputs[0].shape[0], 2, *(node_lstm1.outputs[0].shape[2:])),
        )

        # Created a fused node
        fused_bilstm_node = gs.Node(
            op="LSTM",
            name=f"{fused_bilstm_prefix}_{fused_bilstm_idx}",
            attrs=bilstm_attrs,
            inputs=[X, W, R, B],
            outputs=[Y],
        )

        self.append_bilstm_transpose_reshape_cleanup(
            node_start,
            node_end,
            fused_bilstm_node,
            fused_bilstm_prefix,
            fused_bilstm_idx,
        )

    def fuse_bilstm2(self, fused_bilstm_idx):
        fused_bilstm_prefix = "fused_bilstm2"
        for node in self.graph.nodes:
            # Get nodes for fusion and bidirectional LSTM insertion if the forward LSTM is detected
            detected, node_start, node_lstm1, node_lstm2, node_end = self.bilstm2_detected(node)

            if detected:
                self.replace_bilstm2(
                    node_start,
                    node_lstm1,
                    node_lstm2,
                    node_end,
                    fused_bilstm_prefix,
                    fused_bilstm_idx,
                )
                return True
        return False

    def append_bilstm_transpose_reshape_cleanup(
        self,
        node_start,
        node_end,
        fused_bilstm_node,
        fused_bilstm_prefix,
        fused_bilstm_idx,
    ):
        self.graph.nodes.append(fused_bilstm_node)

        trans_attrs = OrderedDict()
        trans_attrs["perm"] = [0, 2, 1, 3]

        transpose_out = gs.Variable(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_transpose_out",
            dtype=np.float32,
            shape=([fused_bilstm_node.outputs[0].shape[i] for i in trans_attrs["perm"]]),
        )

        transpose_node = gs.Node(
            op="Transpose",
            name=f"{fused_bilstm_prefix}_{fused_bilstm_idx}_transpose",
            attrs=trans_attrs,
            inputs=fused_bilstm_node.outputs,
            outputs=[transpose_out],
        )
        self.graph.nodes.append(transpose_node)

        reshape_shape = gs.Constant(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_shape",
            values=np.array([transpose_out.shape[0], 0, -1], dtype=np.int64),
        )

        reshape_out = gs.Variable(
            f"{fused_bilstm_prefix}_{fused_bilstm_idx}_reshape_out",
            dtype=np.float32,
            shape=(*transpose_out.shape[:2], math.prod(transpose_out.shape[2:])),
        )

        reshape_node = gs.Node(
            op="Reshape",
            name=f"{fused_bilstm_prefix}_{fused_bilstm_idx}_reshape",
            inputs=[transpose_out, reshape_shape],
            outputs=[reshape_out],
        )
        self.graph.nodes.append(reshape_node)

        if (
            node_start.op == "Reshape"
            and isinstance(node_start.inputs[1], gs.Constant)
            and math.prod(node_start.inputs[1].values[1:]) != 0
        ):
            node_reshape_new_shape = gs.Constant(
                f"{fused_bilstm_prefix}_{fused_bilstm_idx}_shape_dynamic",
                values=np.array([node_start.inputs[1].values[0], 0, -1], dtype=np.int64),
            )
            node_start.inputs[1] = node_reshape_new_shape

        node_start.outputs[0] = fused_bilstm_node.inputs[0]
        for _ in range(len(node_end.outputs[0].outputs)):
            node_end.o().inputs[0] = reshape_out

        self.cleanup()

    def bilstm_with_add_detected(self, node):
        if (
            node.op == "Reshape"
            and len(node.outputs) == 1
            and len(node.outputs[0].outputs) == 2
            and node.o().op == "Slice"
            and len(node.inputs[0].inputs) > 0
            and node.i().op == "Transpose"
            and node.o(1).op == "Concat"
            and node.o(1).o().op == "Add"
            and node.o(1).o().o().op == "LSTM"
            and node.o(1).o().o().o().op == "Reshape"
            and node.o(1).o().o().o().o().op == "Slice"
            and len(node.o(1).o().o().o().outputs[0].outputs) == 2
            and node.o(1).o().o().o().o(1).op == "Slice"
            and node.o(1).o().o().o().o(1).o().op == "Slice"
            and node.o(1).o().o().o().o().o().op == "Concat"
            and len(node.o(1).o().inputs) == 2
            and node.o(1).o().i(1).op == "Reshape"
            and node.o(1).o().i(1).i().op == "Mul"
            and node.o(1).o().i(1).i().i().op == "Concat"
            and node.o(1).o().i(1).i().i().i().op == "Transpose"
        ):
            tensor_clip = node.o(1).o().i(1).i().i().i().inputs[0]
            node_transpose = node.i()
            node_pre_add = node.o(1)
            node_lstm = node.o(1).o().o()
            node_add = node.o(1).o()
            return True, node_lstm, node_pre_add, node_transpose, node_add, tensor_clip

        if (
            node.op == "Transpose"
            and len(node.outputs[0].outputs) == 2
            and node.o(0).op == "Add"
            and node.o(0).o().op == "LSTM"
            and node.o(0).o().o().op == "Reshape"
            and node.o(0).o().o().o().op == "Concat"
            and len(node.o(0).inputs) == 2
            and node.o(0).i(1).op == "Reshape"
            and node.o(0).i(1).i().op == "Mul"
            and node.o(0).i(1).i().i().op == "Transpose"
            and node.o(1).op == "Slice"
            and node.o(1).o().op == "LSTM"
            and node.o(1).o().o().op == "Reshape"
            and node.o(1).o().o().o().op == "Slice"
            and node.o(1).o().o().o().o().op == "Concat"
        ):
            node_transpose = node
            node_pre_add = node
            node_add = node.o(0)
            node_lstm = node.o(0).o()
            tensor_clip = node.o(0).i(1).i().i().inputs[0]
            return True, node_lstm, node_pre_add, node_transpose, node_add, tensor_clip

        if (
            node.op == "Transpose"
            and len(node.outputs[0].outputs) == 2
            and node.o(1).op == "Add"
            and node.o(1).o().op == "LSTM"
            and node.o(1).o().o().op == "Reshape"
            and node.o(1).o().o().o().op == "Concat"
            and len(node.o(1).inputs) == 2
            and node.o(1).i(1).op == "Reshape"
            and node.o(1).i(1).i().op == "Mul"
            and node.o(1).i(1).i().i().op == "Transpose"
            and node.o(0).op == "Slice"
            and node.o(0).o().op == "LSTM"
            and node.o(0).o().o().op == "Reshape"
            and node.o(0).o().o().o().op == "Slice"
            and node.o(0).o().o().o().o().op == "Concat"
        ):
            node_transpose = node
            node_pre_add = node
            node_add = node.o(1)
            node_lstm = node.o(1).o()
            tensor_clip = node.o(1).i(1).i().i().inputs[0]
            return True, node_lstm, node_pre_add, node_transpose, node_add, tensor_clip

        return False, None, None, None, None, None

    def eliminate_add_in_lstm(
        self,
        node_lstm,
        node_pre_add,
        node_transpose,
        node_add,
        tensor_clip,
        eliminate_add_idx,
    ):
        # Reshape Node
        reshape_expected_length = len(node_transpose.inputs[0].shape)
        if reshape_expected_length == 3:
            reshape_expected_values = [0, tensor_clip.shape[1], 1]
            reshape_expected_shape = (*tensor_clip.shape, 1)
        elif reshape_expected_length == 4:
            reshape_expected_values = [-1, 1, *tensor_clip.shape]
            reshape_expected_shape = (1, 1, *tensor_clip.shape)
        else:
            raise NotImplementedError(f"Currently doesn't support shape: {node_transpose.inputs[0].shape}")

        reshape_shape = gs.Constant(
            f"pre_reshape_{eliminate_add_idx}_shape",
            values=np.array(reshape_expected_values, dtype=np.int64),
        )
        reshape_out = gs.Variable(
            f"pre_reshape_{eliminate_add_idx}_reshape_out",
            dtype=np.float32,
            shape=reshape_expected_shape,
        )
        reshape_node = gs.Node(
            op="Reshape",
            name=f"pre_reshape_{eliminate_add_idx}_reshape",
            inputs=[tensor_clip, reshape_shape],
            outputs=[reshape_out],
        )
        self.graph.nodes.append(reshape_node)

        # Mul Node
        mul_constant = gs.Constant(
            f"fused_mul_{eliminate_add_idx}_mul",
            values=np.array([0.0], dtype=np.float32),
        )
        mul_out = gs.Variable(
            f"fused_mul_{eliminate_add_idx}_mul_out",
            dtype=np.float32,
            shape=reshape_expected_shape,
        )
        mul_node = gs.Node(
            op="Mul",
            name=f"fused_mul_{eliminate_add_idx}_mul",
            attrs={},
            inputs=[reshape_out, mul_constant],
            outputs=[mul_out],
        )
        self.graph.nodes.append(mul_node)

        # Add Node
        add_out = gs.Variable(
            f"fused_mul_{eliminate_add_idx}_add_out",
            dtype=np.float32,
            shape=node_transpose.inputs[0].shape,
        )
        add_node = gs.Node(
            op="Add",
            name=f"fused_mul_{eliminate_add_idx}_add",
            attrs={},
            inputs=[node_transpose.inputs[0], mul_out],
            outputs=[add_out],
        )
        self.graph.nodes.append(add_node)

        node_transpose.inputs[0] = add_out
        node_add.inputs.clear()
        node_add.outputs.clear()
        node_lstm.inputs[0] = node_pre_add.outputs[0]

        self.cleanup()

    def eliminate_add(self, eliminate_add_idx):
        for node in self.graph.nodes:
            # Get Add node for eliminating
            (
                detected,
                node_lstm,
                node_pre_add,
                node_transpose,
                node_add,
                tensor_clip,
            ) = self.bilstm_with_add_detected(node)

            if detected:
                self.eliminate_add_in_lstm(
                    node_lstm,
                    node_pre_add,
                    node_transpose,
                    node_add,
                    tensor_clip,
                    eliminate_add_idx,
                )
                return True
        return False

    def rewrite_bilstm_pre_clip(self):
        eliminate_add_idx = 0
        while self.eliminate_add(eliminate_add_idx):
            eliminate_add_idx += 1
        return eliminate_add_idx

    def rewrite_bilstm(self):
        fused_bilstm_idx = 0
        while self.fuse_bilstm(fused_bilstm_idx):
            fused_bilstm_idx += 1

        fused_bilstm2_idx = 0
        while self.fuse_bilstm2(fused_bilstm2_idx):
            fused_bilstm2_idx += 1
        return fused_bilstm_idx, fused_bilstm2_idx
