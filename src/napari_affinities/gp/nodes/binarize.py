import gunpowder as gp
import numpy as np


class Binarize(gp.BatchFilter):
    def __init__(self, input_key: gp.ArrayKey, output_key: gp.ArrayKey):
        self.input_key = input_key
        self.output_key = output_key

    def setup(self):
        self.provides(self.output_key, self.spec[self.input_key].copy())

    def prepare(self, request: gp.BatchRequest) -> gp.BatchRequest:
        deps = gp.BatchRequest()
        deps[self.input_key] = request[self.output_key].copy()
        return deps

    def process(self, batch: gp.Batch, request: gp.BatchRequest) -> gp.Batch:
        outputs = gp.Batch()
        output_spec = batch[self.input_key].spec.copy()
        output_spec.dtype = bool
        outputs[self.output_key] = gp.Array(
            (batch[self.input_key].data > 0).astype(bool), output_spec
        )
        return outputs
