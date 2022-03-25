from napari.layers import Image

import gunpowder as gp
from gunpowder.profiling import Timing
from gunpowder.array_spec import ArraySpec

import numpy as np


class NapariImageSource(gp.BatchProvider):
    """
    A gunpowder interface to a napari Image
    Args:
        image (Image):
            The napari Image to pull data from
        key (``gp.ArrayKey``):
            The key to provide data into
    """

    def __init__(self, image: Image, key: gp.ArrayKey):
        self.array_spec = self._read_metadata(image)
        self.image = gp.Array(image.data, self.array_spec)
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()

        timing_provide = Timing(self, "provide")
        timing_provide.start()

        output[self.key] = self.image.crop(request[self.key].roi)

        timing_provide.stop()

        output.profiling_stats.add(timing_provide)

        return output

    def _read_metadata(self, image):
        # offset assumed to be in world coordinates
        offset = gp.Coordinate(image.metadata.get("offset", (1, 1)))
        voxel_size = gp.Coordinate(image.metadata.get("resolution", (1, 1)))
        shape = gp.Coordinate(image.data.shape[-offset.dims() :])

        return gp.ArraySpec(
            roi=gp.Roi(offset, voxel_size * shape),
            dtype=image.dtype,
            interpolatable=True,
            voxel_size=voxel_size,
        )
