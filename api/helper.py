from pathlib import Path
from typing import List, Optional, Tuple


def glob_caffe_graph_and_params(
    checkpoint_path: Optional[str],
    prototxt_path: Optional[str],
    caffemodel_path: Optional[str],
) -> Tuple[List[Path], ...]:
    """
    Finding the list of caffemodel and prototxt files to be converted.
    """
    if checkpoint_path is None and prototxt_path is None and caffemodel_path is None:
        raise AssertionError("Either checkpoint_path or (prototxt_path, caffemodel_path) must be set.")

    if checkpoint_path is not None:
        caffemodel_files = sorted(Path(checkpoint_path).rglob("**/*.caffemodel"))
        graph_paths = []
        params_paths = []
        for caffemodel_file in caffemodel_files:
            prototxt_file = caffemodel_file.with_suffix(".prototxt")
            if not prototxt_file.exists():
                raise FileNotFoundError(f"Can't find the corresponding caffe params file: '{prototxt_file}'.")
            graph_paths.append(prototxt_file)
            params_paths.append(caffemodel_file)
        return graph_paths, params_paths

    if not Path(prototxt_path).exists():
        raise FileNotFoundError(f"Can't find prototxt file: '{prototxt_path}'.")
    if not Path(caffemodel_path).exists():
        raise FileNotFoundError(f"Can't find caffemodel file: '{caffemodel_path}'.")
    return [Path(prototxt_path)], [Path(caffemodel_path)]


def glob_onnx_models(checkpoint_path: str, check_caffe_exists: bool = True) -> List[Path]:
    """
    Finding the list of converted ONNX models.
    """

    onnx_files = sorted(Path(checkpoint_path).rglob("**/*.onnx"))

    if not check_caffe_exists:
        return onnx_files

    onnx_paths = []
    for onnx_file in onnx_files:
        caffemodel_file = onnx_file.with_suffix(".caffemodel")
        if not caffemodel_file.exists():
            raise FileNotFoundError(f"Can't find the corresponding caffemodel file: '{caffemodel_file}'.")
        prototxt_file = onnx_file.with_suffix(".prototxt")
        if not prototxt_file.exists():
            raise FileNotFoundError(f"Can't find the corresponding prototxt file: '{prototxt_file}'.")
        onnx_paths.append(onnx_file)
    return onnx_paths
