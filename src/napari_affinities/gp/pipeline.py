from .nodes import NapariImageSource, NapariLabelsSource, OnesSource

import gunpowder as gp
from lsd.gp import AddLocalShapeDescriptor

import numpy as np
from bioimageio.core.resource_io.nodes import Model

from contextlib import contextmanager
from typing import List, Tuple

LayerName = str
LayerType = str


class PipelineDataGenerator:
    """
    Simple wrapper around a gunpowder pipeline.
    """

    def __init__(
        self,
        pipeline: gp.Pipeline,
        request: gp.BatchRequest,
        snapshot_request: gp.BatchRequest,
        keys: List[Tuple[gp.ArrayKey, str]],
    ):
        self.pipeline = pipeline
        self.request = request
        self.snapshot_request = snapshot_request
        self.keys = keys

    def next(self, snapshot: bool) -> List[Tuple[np.ndarray, LayerName, LayerType]]:
        request = gp.BatchRequest()
        request_template = self.snapshot_request if snapshot else self.request
        for k, v in request_template.items():
            request[k] = v
        batch = self.pipeline.request_batch(request)

        arrays = []
        snapshot_arrays = []
        for key, layer_type in self.keys:
            if key in request_template:
                layer = (batch[key].data, {"name": f"sample_{key}".lower()}, layer_type)
                if key in self.request:
                    arrays.append(layer)
                else:
                    snapshot_arrays.append(layer)

        return arrays, snapshot_arrays


@contextmanager
def build_pipeline(
    raw,
    gt,
    mask,
    lsds: bool,
    model: Model,
    num_cpu_processes: int = 1,
    batch_size: int = 1,
):
    # read metadata from model
    offsets = model.config["mws"]["offsets"]
    dims = len(offsets[0])
    input_shape = gp.Coordinate(model.inputs[0].shape.min[-dims:])
    output_shape = gp.Coordinate(input_shape)

    # get voxel sizes TODO: read from metadata?
    voxel_size = gp.Coordinate((1,) * input_shape.dims())

    # witch to world units:
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    # padding of groundtruth/mask
    # without padding you random sampling won't be uniform over the whole volume
    padding = output_size  # TODO: add sampling for extra lsd/affinities context

    # define keys:
    raw_key = gp.ArrayKey("RAW")
    gt_key = gp.ArrayKey("GT")
    mask_key = gp.ArrayKey("MASK")
    affinity_key = gp.ArrayKey("AFFINITY")
    affinity_mask_key = gp.ArrayKey("AFFINITY_MASK")
    lsd_key = gp.ArrayKey("LSD")
    lsd_mask_key = gp.ArrayKey("LSD_MASK")

    # Get source nodes:
    raw_source = NapariImageSource(raw, raw_key)
    # Pad raw infinitely with 0s. This is to avoid failing to train on any of
    # the ground truth because there wasn't enough raw context.
    raw_source += gp.Pad(raw_key, None, 0)

    gt_source = NapariLabelsSource(gt, gt_key)

    if mask is not None:
        mask_source = NapariLabelsSource(mask, mask_key)
    else:
        with gp.build(gt_source):
            mask_source = OnesSource(gt_source.spec[gt_key], mask_key)

    # Pad gt/mask with just enough to make sure random sampling is uniform
    gt_source += gp.Pad(gt_key, padding, 0)
    mask_source += gp.Pad(mask_key, padding, 0)

    pipeline = (
        (raw_source, gt_source, mask_source) + gp.MergeProvider() + gp.RandomLocation()
    )

    # TODO: add augments

    # Generate Affinities
    pipeline += gp.AddAffinities(
        offsets,
        gt_key,
        affinity_key,
        labels_mask=mask_key,
        affinities_mask=affinity_mask_key,
    )
    if lsds:
        pipeline += AddLocalShapeDescriptor(
            gt_key,
            lsd_key,
            mask=lsd_mask_key,
            sigma=3 * voxel_size,
        )

    # Trainer attributes:
    if num_cpu_processes > 1:
        pipeline += gp.PreCache(num_workers=num_cpu_processes)

    # stack to create a batch dimension
    pipeline += gp.Stack(batch_size)

    request = gp.BatchRequest()
    request.add(raw_key, input_size)
    request.add(affinity_key, output_size)
    request.add(affinity_mask_key, output_size)
    if lsds:
        request.add(lsd_key, output_size)
        request.add(lsd_mask_key, output_size)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw_key, input_size)
    snapshot_request.add(affinity_key, output_size)
    snapshot_request.add(affinity_mask_key, output_size)
    if lsds:
        snapshot_request.add(lsd_key, output_size)
        snapshot_request.add(lsd_mask_key, output_size)
    snapshot_request.add(gt_key, output_size)
    snapshot_request.add(mask_key, output_size)

    assert not lsds

    keys = [
        (raw_key, "image"),
        (affinity_key, "labels"),
        (affinity_mask_key, "labels"),
        (lsd_key, "image"),
        (lsd_mask_key, "labels"),
        (gt_key, "labels"),
        (mask_key, "labels"),
    ]

    with gp.build(pipeline):
        yield PipelineDataGenerator(pipeline, request, snapshot_request, keys)
