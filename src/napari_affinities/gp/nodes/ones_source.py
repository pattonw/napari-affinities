import gunpowder as gp
from gunpowder.profiling import Timing
from gunpowder.array_spec import ArraySpec

import numpy as np


class OnesSource(gp.BatchProvider):
    """
    A gunpowder source that provides ones in the given spec
    Args:
        spec (ArraySpec):
            The spec to populate with ones (virtually)
        key (``gp.ArrayKey``):
            The key to provide data into
    """

    def __init__(self, spec: ArraySpec, key: gp.ArrayKey):
        self.array_spec = spec
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()

        timing_provide = Timing(self, "provide")
        timing_provide.start()

        spec = self.array_spec.copy()
        spec.roi = request[self.key].roi
        assert spec.voxel_size is not None, "Please provide a voxel size!"

        output[self.key] = gp.Array(
            np.ones(tuple(spec.roi.get_shape() / spec.voxel_size), dtype=spec.dtype),
            spec=spec,
        )

        timing_provide.stop()
        output.profiling_stats.add(timing_provide)

        return output
