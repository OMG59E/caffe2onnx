from .base_rewriter import BaseRewriter


class ProposalRewriter(BaseRewriter):
    def __init__(self, graph, verbose: bool = False):
        super().__init__(graph, verbose)

    def node_proposal_detected(self, node):
        if (
            node.op == "Proposal"
            and node.domain == "ax.caffe2onnx"
            and len(node.inputs) == 3
            and len(node.inputs[2].shape) != 4
        ):
            return True, node

        return False, None

    def rewrite_im_info_shape(self, node_proposal):
        im_info_shape: list = node_proposal.inputs[2].shape
        if len(im_info_shape) > 4:
            raise AttributeError("Length of input im_info of Proposal must be smaller than 4.")
        # Make length of im_info to 4
        while len(im_info_shape) < 4:
            im_info_shape.insert(0 if len(im_info_shape) == 1 else 1, 1)

        self.cleanup()

    def rewriting_im_info_shape(self):
        for node in self.graph.nodes:
            # Get Caffe Proposal node
            detected, node_proposal = self.node_proposal_detected(node)

            if detected:
                self.rewrite_im_info_shape(node_proposal)
                return True
        return False

    def insert_one_to_im_info(self):
        im_info_index = 0
        while self.rewriting_im_info_shape():
            im_info_index += 1
        return im_info_index
