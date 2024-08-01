import copy

from caffe2onnx.graph import CaffeToOnnxNode


def get_tile_attr(layer):
    return {
        "axis": layer.tile_param.axis,
        "tiles": layer.tile_param.tiles,
    }


def get_tile_output_shape(input_shape: list[list[int]], attrs: dict[str, int]) -> list[list[int]]:
    output_shape = copy.deepcopy(input_shape[0])
    output_shape[attrs["axis"]] *= attrs["tiles"]
    return [output_shape]


def create_tile_node(layer, node_name, input_name, output_name, input_shape):
    attrs = get_tile_attr(layer)
    output_shape = get_tile_output_shape(input_shape, attrs)

    node = CaffeToOnnxNode(layer, node_name, "Tile", input_name, output_name, input_shape, output_shape)

    return node
