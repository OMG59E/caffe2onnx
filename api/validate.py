#!/usr/bin/env python3
"""
Validate the consistency between caffemodel and onnxmodel in the given path recursively.
"""

from collections import defaultdict
from enum import auto, Flag
from pathlib import Path

import numpy as np
import onnxruntime as ort

from rich.console import Console
from rich.progress import track
from rich.table import Table

from caffe2onnx.utils import compute_corrcoef_std, cosine_similarity, diff_max_mean, normalized_relative_error

from .helper import glob_onnx_models


class ValidateFlag(Flag):
    PASS = 0
    FAIL = auto()
    UNRESOLVABLE = auto()


def validate_recursively(
    checkpoint_path: str,
    coef_threshold: float = 0,
    std_dev_threshold: float = 0,
    cos_sim_threshold: float = 0,
    norm_rel_err_threshold: float = 0,
    diff_max_threshold: float = 0,
    diff_mean_threshold: float = 0,
    num_epoch: int = 10,
):

    validation_info = {"num_failure": 0, "num_case": 0, "num_unresolvable": 0, "log_info": ["", "", ""]}

    onnx_paths = glob_onnx_models(checkpoint_path)

    for onnx_path in onnx_paths:
        caffemodel_path = str(onnx_path.with_suffix(".caffemodel"))
        prototxt_path = str(onnx_path.with_suffix(".prototxt"))

        validation_info["num_case"] += 1
        try:
            perfs = consistency_between_caffe_onnxruntime(
                prototxt_path,
                caffemodel_path,
                str(onnx_path),
                num_epoch=num_epoch,
            )

            # check if it failed
            flag = ValidateFlag.PASS
            for perf in perfs.values():
                if (
                    perf["corr"] < coef_threshold
                    or perf["std_dev"] > std_dev_threshold
                    or perf["cos_sim"] < cos_sim_threshold
                    or perf["rel_err"] > norm_rel_err_threshold
                    or perf["diff_max"] > diff_max_threshold
                    or perf["diff_mean"] > diff_mean_threshold
                ):
                    flag = ValidateFlag.FAIL
            validation_info["num_failure"] += flag.value

            # log info
            validation_info["log_info"][flag.value] += f"{flag.name}: {onnx_path}\n"
            console = Console()
            table = generate_table(perfs, onnx_path)
            console.print(table)

        except Exception as e:
            flag = ValidateFlag.UNRESOLVABLE
            validation_info["num_unresolvable"] += 1
            validation_info["log_info"][flag.value] += f"{flag.name}: {onnx_path}\n"
            validation_info["log_info"][flag.value] += f"{e}\n"

    return validation_info


def generate_table(performance, model_name: str):
    info_table = Table(
        "Layer",
        "Correlation",
        "standard deviation",
        "cosine similarity",
        "2-norm RE",
        "max difference",
        "mean difference",
        title=f"performance on: {model_name}",
    )
    for name, perf in performance.items():
        info_table.add_row(
            f"{name}",
            f"{perf['corr']:.10f}",
            f"{perf['std_dev']:.10f}",
            f"{perf['cos_sim']:.10f}",
            f"{perf['rel_err']:.10f}",
            f"{perf['diff_max']:.10f}",
            f"{perf['diff_mean']:.10f}",
        )
    return info_table


def consistency_between_caffe_onnxruntime(proto_path, caffemodel_path, onnx_path, num_epoch: int = 10):
    import caffe

    caffe_model = caffe.Net(proto_path, caffemodel_path, caffe.TEST)

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    onnx_model = ort.InferenceSession(onnx_path, sess_options=sess_options)

    perfs = defaultdict(
        lambda: {
            "corr": 0.0,
            "std_dev": 0.0,
            "cos_sim": 0.0,
            "rel_err": 0.0,
            "diff_max": 0.0,
            "diff_mean": 0.0,
        }
    )

    # onnx inputs and outputs info
    onnx_inputs = onnx_model.get_inputs()
    onnx_outputs_name = onnx_model.get_outputs()

    # caffe inputs and output info
    caffe_inputs_name = caffe_model.inputs  # list
    caffe_outputs_name = caffe_model.outputs  # list
    assert len(caffe_inputs_name) == len(onnx_inputs), "Caffe inputs and ONNX inputs are not identical!"

    for i in track(range(num_epoch), description=f"Validating {Path(onnx_path).name}"):
        # generate random data
        input_data = {}
        for inp in onnx_inputs:
            if inp.name == "clip":
                input_data[inp.name] = np.ones(inp.shape, dtype=np.float32)
                input_data[inp.name][0][0] = 0.0
            else:
                input_data[inp.name] = np.random.rand(*inp.shape).astype(np.float32)

        # onnx inference
        onnx_output_data = onnx_model.run([], input_data)
        onnx_output = {}
        for j in range(len(onnx_outputs_name)):
            onnx_output[onnx_outputs_name[j].name] = onnx_output_data[j]

        # caffe inference
        for j, (name, data) in enumerate(input_data.items()):
            caffe_model.blobs[caffe_inputs_name[j]].data[...] = data

        caffe_model.forward()
        caffe_output = {}
        for out_name in caffe_outputs_name:
            caffe_output[out_name] = np.asarray(caffe_model.blobs[out_name].data)

        # calculate loss
        assert len(caffe_output) == len(onnx_output), "The number of Caffe outputs and ONNX outputs are not equal!"
        for out_name in caffe_output.keys():
            coef, std_dev = compute_corrcoef_std(caffe_output[out_name], onnx_output[out_name])
            cos_sim = cosine_similarity(caffe_output[out_name], onnx_output[out_name])
            rel_err = normalized_relative_error(caffe_output[out_name], onnx_output[out_name])
            diff_max, diff_mean = diff_max_mean(caffe_output[out_name], onnx_output[out_name])
            perfs[out_name]["corr"] = (perfs[out_name]["corr"] * i + coef) / (i + 1)

            perfs[out_name]["std_dev"] = (perfs[out_name]["std_dev"] * i + std_dev) / (i + 1)
            perfs[out_name]["cos_sim"] = (perfs[out_name]["cos_sim"] * i + cos_sim) / (i + 1)
            perfs[out_name]["rel_err"] = (perfs[out_name]["rel_err"] * i + rel_err) / (i + 1)
            perfs[out_name]["diff_max"] = max(perfs[out_name]["diff_max"], diff_max)
            perfs[out_name]["diff_mean"] = (perfs[out_name]["diff_mean"] * i + diff_mean) / (i + 1)

    return perfs
