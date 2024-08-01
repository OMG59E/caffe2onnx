import caffe2onnx.graphsurgeon as gs


class BaseRewriter:
    def __init__(self, graph: gs.Graph, verbose: bool = False):
        self.graph = graph
        self.verbose = verbose

    def info(self, prefix=""):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, "
                f"{len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs."
            )

    def cleanup(self, return_onnx: bool = False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name
