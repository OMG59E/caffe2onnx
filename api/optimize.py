# Copyright (c) 2023, Axera Tech. All rights reserved.

import caffe2onnx.graphsurgeon as gs

from caffe2onnx.passes import (
    BiLSTMRewriter,
    ExpAddLogRewriter,
    ProposalRewriter,
    ReductionRewriter,
    ReshapeGemmReshapeRewriter,
    ReshapeGemmRewriter,
    RoiAlignRewriter,
    TransposeReshapeLSTMReshapeRewriter,
)


def optimize_onnx(onnx_model, verbose: bool = True):
    graph = gs.import_onnx(onnx_model)

    # Optimize BiLSTM
    opt = BiLSTMRewriter(graph, verbose=verbose)
    num_add_eliminated = opt.rewrite_bilstm_pre_clip()
    if num_add_eliminated > 0:
        opt.info(f"Rewriter: Eliminated {num_add_eliminated} Add before BiLSTM")
    num_bilstm_fused, num_bilstm2_fused = opt.rewrite_bilstm()
    if num_bilstm_fused > 0:
        opt.info(f"Rewriter: Fused {num_bilstm_fused} bidirectional LSTM")
    if num_bilstm2_fused > 0:
        opt.info(f"Rewriter: Fused {num_bilstm2_fused} bidirectional LSTM v2")

    # Optimize Reshape-Gemm-Reshape
    opt = ReshapeGemmReshapeRewriter(opt.graph, verbose=verbose)
    num_matmul_fused = opt.rewrite_reshape_gemm_reshape()
    if num_matmul_fused > 0:
        opt.info(f"Rewriter: Fused {num_matmul_fused} Reshape-Gemm-Reshape")

    # Make Reshape support dynamic batch
    opt = TransposeReshapeLSTMReshapeRewriter(opt.graph, verbose=verbose)
    num_reshape_modified = opt.rewrite_transpose_reshape_lstm_reshape()
    if num_reshape_modified > 0:
        opt.info(f"Rewriter: Modified {num_reshape_modified} Transpose-Reshape-LSTM-Reshape")

    # Make Reshape support dynamic batch
    opt = ReshapeGemmRewriter(opt.graph, verbose=verbose)
    num_inner_product_modified = opt.rewrite_reshape_gemm()
    if num_inner_product_modified > 0:
        opt.info(f"Rewriter: Modified {num_inner_product_modified} Reshape-Gemm")

    # Fuse Softplus
    opt = ExpAddLogRewriter(opt.graph, verbose=verbose)
    num_softplus_fused = opt.rewrite_softplus()
    if num_softplus_fused > 0:
        opt.info(f"Rewriter: Fused {num_softplus_fused} Exp-Add-Log")

    # Modify Caffe ROIAlign to Onnx RoiAlign
    opt = RoiAlignRewriter(opt.graph, verbose=verbose)
    num_roi_align_modified = opt.rewrite_roi_align()
    if num_roi_align_modified > 0:
        opt.info(f"Rewriter: Modified {num_roi_align_modified} Caffe ROIAlign")

    # Insert 1 to Caffe Proposal im_info
    opt = ProposalRewriter(opt.graph, verbose=verbose)
    num_im_info_modified = opt.insert_one_to_im_info()
    if num_im_info_modified > 0:
        opt.info(f"Rewriter: Modified {num_im_info_modified} im_info of Proposal")

    # Rewriter Caffe Reduction to Onnx ReduceSum and ReduceMean
    opt = ReductionRewriter(opt.graph, verbose=verbose)
    num_reduction_modified = opt.rewrite_reduction()
    if num_reduction_modified > 0:
        opt.info(f"Rewriter: Modified {num_reduction_modified} Caffe Reduction")

    model_optimized = gs.export_onnx(opt.graph)
    return model_optimized
