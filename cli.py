#!/usr/bin/env python3

import argparse
import os

from pathlib import Path


def convert_caffe_to_onnx_cli(args):
    import onnx

    from caffe2onnx.api import convert_caffe_to_onnx, optimize_onnx
    from caffe2onnx.api.helper import glob_caffe_graph_and_params

    graph_paths, params_paths = glob_caffe_graph_and_params(
        args.checkpoint_path,
        args.prototxt_path,
        args.caffemodel_path,
    )
    opset_version = args.opset_version
    skip_onnxsim = args.skip_onnxsim

    for caffe_graph_path, caffe_params_path in zip(graph_paths, params_paths):
        print(f"Converter: Converting caffe model: {caffe_params_path}")
        onnx_name = caffe_graph_path.stem
        onnx_save_path = caffe_graph_path.with_suffix(".onnx")

        if args.onnx_path is not None:
            onnx_save_path = args.onnx_path

        onnx_model = convert_caffe_to_onnx(
            caffe_graph_path,
            caffe_params_path,
            onnx_name=onnx_name,
            opset_version=opset_version,
            skip_onnxsim=skip_onnxsim,
            batch_size=args.batch_size
        )

        if not args.skip_rewriter:
            print("Rewriter: Passing layers...")
            onnx_model = optimize_onnx(onnx_model)

        onnx.save_model(onnx_model, onnx_save_path)
        print(f"Converter: Converted ONNX model saved at: {onnx_save_path}")


def validate_caffe_onnx_cli(args):
    import caffe

    from caffe2onnx.api import validate_recursively
    from caffe2onnx.api.validate import ValidateFlag

    caffe.set_mode_cpu()

    validate_info = validate_recursively(
        args.checkpoint_path,
        coef_threshold=args.coef,
        std_dev_threshold=args.std_dev,
        cos_sim_threshold=args.cos_sim,
        norm_rel_err_threshold=args.rel_err,
        diff_max_threshold=args.diff_max,
        diff_mean_threshold=args.diff_mean,
        num_epoch=args.num_epoch,
    )

    print("The following are the detailed reports:")
    log_info = validate_info["log_info"]

    for flag in ValidateFlag:
        if len(log_info[flag.value]) > 0:
            print(log_info[flag.value])

    print(
        f"Validation complete, validated {validate_info['num_case']} in total, "
        f"{validate_info['num_case'] - validate_info['num_failure'] - validate_info['num_unresolvable']} successed, "
        f"{validate_info['num_failure']} failed, {validate_info['num_unresolvable']} unresolvable."
    )


def optimize_onnx_cli(args):
    import onnx

    from caffe2onnx.api import optimize_onnx, toposort_and_simlify
    from caffe2onnx.api.helper import glob_onnx_models

    onnx_model_paths = glob_onnx_models(args.checkpoint_path, check_caffe_exists=False)
    for onnx_path in onnx_model_paths:
        onnx_model = onnx.load(onnx_path)
        print(f"Simplifier: Simplifying {onnx_path}...")
        onnx_optimized = toposort_and_simlify(onnx_model)

        if not args.skip_rewriter:
            print("Rewriter: Passing layers...")
            onnx_optimized = optimize_onnx(onnx_optimized)

        onnx_optimized_path = Path(args.onnx_path) / onnx_path.name
        print(f"Simplifier: Save optimized model to {onnx_optimized_path}.")
        onnx.save(onnx_optimized, onnx_optimized_path)


def generate_dummy_caffemodel_cli(args):
    """
    Create a new Caffe model file with all layers initialized with random weights.
    """
    import caffe

    caffe.set_mode_cpu()

    prototxt_paths = sorted(Path(args.checkpoint_path).rglob("**/*.prototxt"))

    for prototxt_path in prototxt_paths:
        print(f"Generator: Load the prototxt file {prototxt_path}.")
        net = caffe.Net(str(prototxt_path), caffe.TEST)

        print("Generator: Save the dummy caffemodel file.")
        net.save(str(prototxt_path.with_suffix(".caffemodel")))


def main():
    parser = argparse.ArgumentParser(description="Caffe model to ONNX converter and validation")
    parser.add_argument("--convert", action="store_true", help="Convert Caffe to ONNX.")
    parser.add_argument("--validate", action="store_true", help="Validate Caffe vs ONNX.")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX models.")
    parser.add_argument("--generate", action="store_true", help="Generate dummy caffemodel from prototxt.")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Directory path for prototxt and caffemodel files.",
    )
    parser.add_argument(
        "--prototxt_path",
        type=str,
        default=None,
        help="Path to the .prototxt file with text description of the network architecture.",
    )
    parser.add_argument(
        "--caffemodel_path",
        type=str,
        default=None,
        help="Path to the .caffemodel file with learned network.",
    )
    parser.add_argument("--onnx_path", type=str, default=None, help="Path to the .onnx file exported by caffe2onnx.")
    parser.add_argument("--opset_version", type=int, default=16, help="Opset version used to export ONNX models.")
    parser.add_argument("--skip_onnxsim", action="store_true", help="Skipping onnx-simplifier.")

    parser.add_argument("--coef", type=float, default=0.999, help="Threshold of correlation coefficients")
    parser.add_argument("--std_dev", type=float, default=1e-4, help="Threshold of standard deviation")
    parser.add_argument("--cos_sim", type=float, default=0.999, help="Threshold of cosine similarity")
    parser.add_argument("--rel_err", type=float, default=1e-4, help="Threshold of normalized relative error")
    parser.add_argument("--diff_max", type=float, default=2e-4, help="Threshold of max difference")
    parser.add_argument("--diff_mean", type=float, default=1e-5, help="Threshold of mean difference")
    parser.add_argument("--num_epoch", type=int, default=10, help="Number of epoch for validation")
    parser.add_argument("--skip_rewriter", action="store_true", help="Skipping onnx-rewriter.")
    parser.add_argument("--batch_size", type=int, default=1, help="Setting batch_size.")

    args = parser.parse_args()
    if args.convert:
        convert_caffe_to_onnx_cli(args)
    if args.validate:
        os.environ["GLOG_minloglevel"] = "3"
        validate_caffe_onnx_cli(args)
    if args.optimize:
        optimize_onnx_cli(args)
    if args.generate:
        os.environ["GLOG_minloglevel"] = "3"
        generate_dummy_caffemodel_cli(args)
