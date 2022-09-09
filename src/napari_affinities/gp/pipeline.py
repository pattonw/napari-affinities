from .nodes import (
    NapariImageSource,
    NapariLabelsSource,
    OnesSource,
    Binarize,
    NpArraySource,
    AddLocalShapeDescriptor
)

import gunpowder as gp

import numpy as np
from bioimageio.core.resource_io.nodes import Model

from contextlib import contextmanager
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math

LayerName = str
LayerType = str


@dataclass
class GunpowderParameters:
    lsd_sigma: int = 5
    intensity_scale_min: float = 0.8
    intensity_scale_max: float = 1.2
    intensity_shift_min: float = -0.2
    intensity_shift_max: float = 0.2
    gausian_noise_mean: float = 0.0
    gausian_noise_var: float = 0.02
    elastic_control_point_spacing: int = 50
    elastic_control_point_sigma: int = 0
    zoom_min: float = 0.8
    zoom_max: float = 1.2
    rotation: bool = True
    mirror: bool = True
    transpose: bool = True
    num_cpu_processes: int = 1
    batch_size: int = 1


class PipelineDataGenerator:
    """
    Simple wrapper around a gunpowder pipeline.
    """

    def __init__(
        self,
        pipeline: gp.Pipeline,
        val_pipeline: gp.Pipeline,
        request: gp.BatchRequest,
        val_request: gp.BatchRequest,
        snapshot_request: gp.BatchRequest,
        keys: List[Tuple[gp.ArrayKey, str]],
        axes: List[str],
    ):
        self.pipeline = pipeline
        self.val_pipeline = val_pipeline
        self.request = request
        self.val_request = val_request
        self.snapshot_request = snapshot_request
        self.keys = keys
        self.spatial_axes = axes

    def next(
        self, snapshot: bool
    ) -> Tuple[
        List[Tuple[np.ndarray, Dict[str, Any], LayerType]],
        List[Tuple[np.ndarray, Dict[str, Any], LayerType]],
    ]:
        request = gp.BatchRequest()
        request_template = self.snapshot_request if snapshot else self.request
        for k, v in request_template.items():
            request[k] = v
        batch = self.pipeline.request_batch(request)

        arrays = []
        snapshot_arrays = []
        for key, layer_type in self.keys:
            if key in request_template:
                layer = (
                    batch[key].data,
                    {
                        "name": f"sample_{key}".lower(),
                        "axes": ("batch", "channel", *self.spatial_axes),
                    },
                    layer_type,
                )
                if key in self.request:
                    arrays.append(layer)
                else:
                    snapshot_arrays.append(layer)

        return arrays, snapshot_arrays

    def next_validation(
        self,
    ) -> List[Tuple[np.ndarray, Dict[str, Any], LayerType]]:
        request = gp.BatchRequest()
        request_template = self.val_request
        for k, v in request_template.items():
            request[k] = v
        batch = self.val_pipeline.request_batch(request)

        arrays = []
        for key, layer_type in self.keys:
            if key in request_template:
                layer = (
                    batch[key].data,
                    {
                        "name": f"sample_{key}".lower(),
                        "axes": ("batch", "channel", *self.spatial_axes),
                    },
                    layer_type,
                )
                if key in self.request:
                    arrays.append(layer)

        return arrays


@contextmanager
def build_pipeline(
    raw,
    gt,
    mask,
    model: Model,
    parameters: GunpowderParameters,
    affs_high_inter_object: bool = False,
):

    outputs = model.outputs
    metadata_output_names = [output.name.lower() for output in outputs]
    output_names = [x for x in metadata_output_names]
    if "affinities" not in output_names:
        output_names[0] = "affinities"
        if len(output_names) > 1:
            output_names[1] = "fgbg"
        if len(output_names) > 2:
            output_names[2] = "lsds"
        if len(output_names) > 3:
            raise ValueError(
                f"Don't know how to handle outputs: {metadata_output_names}"
            )
    try:
        affs_index = output_names.index("affinities")
    except ValueError as e:
        raise ValueError(
            'This model does not provide an output with name "affinities"! '
            f"{model.name} only provides: {output_names}"
        )
    try:
        lsd_index = output_names.index("lsds")
        lsds = True
    except ValueError:
        lsds = False
    try:
        fgbg_index = output_names.index("fgbg")
        fgbg = True
    except ValueError:
        fgbg = False

    # read metadata from model
    offsets = model.config["mws"]["offsets"]
    dims = len(offsets[0])
    spatial_axes = ["time", "z", "y", "x"][-dims:]

    input_shape = gp.Coordinate(model.inputs[0].shape.min[-dims:])
    output_shape = gp.Coordinate(input_shape)

    # get voxel sizes TODO: read from metadata?
    voxel_size = gp.Coordinate((1,) * input_shape.dims())

    # witch to world units:
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape
    context = (input_size - output_size) / 2

    # padding of groundtruth/mask
    # without padding you random sampling won't be uniform over the whole volume
    padding = (
        output_size  # TODO: add sampling for extra lsd/affinities context
    )

    # define keys:
    raw_key = gp.ArrayKey("RAW")
    gt_key = gp.ArrayKey("GT")
    mask_key = gp.ArrayKey("MASK")
    affinity_key = gp.ArrayKey("AFFINITY")
    affinity_mask_key = gp.ArrayKey("AFFINITY_MASK")
    lsd_key = gp.ArrayKey("LSD")
    lsd_mask_key = gp.ArrayKey("LSD_MASK")
    fgbg_key = gp.ArrayKey("FGBG")
    fgbg_mask_key = gp.ArrayKey("FGBG_MASK")
    training_mask_key = gp.ArrayKey("TRAINING_MASK_KEY")

    # Get source nodes:
    raw_source = NapariImageSource(raw, raw_key)
    val_raw_source = NapariImageSource(raw, raw_key)
    # Pad raw infinitely with 0s. This is to avoid failing to train on any of
    # the ground truth because there wasn't enough raw context.
    raw_source += gp.Pad(raw_key, None, 0)
    val_raw_source += gp.Pad(raw_key, None, 0)

    gt_source = NapariLabelsSource(gt, gt_key)
    val_gt_source = NapariLabelsSource(gt, gt_key)
    with gp.build(val_gt_source):
        val_roi = gp.Roi(
            val_gt_source.spec[gt_key].roi.get_offset(), output_size
        )
        total_roi = val_gt_source.spec[gt_key].roi.copy()
        training_mask_spec = val_gt_source.spec[gt_key].copy()

    shape = total_roi.get_shape() / voxel_size
    training_mask = np.ones(shape, dtype=training_mask_spec.dtype)
    val_slices = [
        slice(0, val_shape) for val_shape in val_roi.get_shape() / voxel_size
    ]
    training_mask[tuple(val_slices)] = 0

    training_mask_source = NpArraySource(
        training_mask, training_mask_spec, training_mask_key
    )

    if mask is not None:
        mask_source = NapariLabelsSource(mask, mask_key)
        val_mask_source = NapariLabelsSource(mask, mask_key)
    else:
        with gp.build(gt_source):
            mask_spec = gt_source.spec[gt_key]
            mask_spec.dtype = bool
            mask_source = OnesSource(gt_source.spec[gt_key], mask_key)
            val_mask_source = OnesSource(
                gt_source.spec[gt_key].copy(), mask_key
            )

    # Pad gt/mask with just enough to make sure random sampling is uniform
    gt_source += gp.Pad(gt_key, padding, 0)
    val_gt_source += gp.Pad(gt_key, padding, 0)
    mask_source += gp.Pad(mask_key, padding, 0)
    val_mask_source += gp.Pad(mask_key, padding, 0)

    val_pipeline = (
        val_raw_source,
        val_gt_source,
        val_mask_source,
    ) + gp.MergeProvider()
    pipeline = (
        (raw_source, gt_source, mask_source, training_mask_source)
        + gp.MergeProvider()
        + gp.RandomLocation(min_masked=1, mask=training_mask_key)
    )

    if parameters.mirror or parameters.transpose:
        pipeline += gp.SimpleAugment(
            mirror_only=[1 if parameters.mirror else 0 for _ in range(dims)],
            transpose_only=[
                1 if parameters.transpose else 0 for _ in range(dims)
            ],
        )
    pipeline += gp.ElasticAugment(
        control_point_spacing=[
            parameters.elastic_control_point_spacing for _ in range(dims)
        ],
        jitter_sigma=[
            parameters.elastic_control_point_sigma for _ in range(dims)
        ],
        rotation_interval=(0, 2 * math.pi if parameters.rotation else 0),
        scale_interval=(parameters.zoom_min, parameters.zoom_max),
    )
    pipeline += gp.NoiseAugment(
        raw_key,
        mean=parameters.gausian_noise_mean,
        var=parameters.gausian_noise_var,
    )

    # Generate Affinities
    pipeline += gp.AddAffinities(
        offsets,
        gt_key,
        affinity_key,
        labels_mask=mask_key,
        affinities_mask=affinity_mask_key,
        dtype=np.int16,
    )
    val_pipeline += gp.AddAffinities(
        offsets,
        gt_key,
        affinity_key,
        labels_mask=mask_key,
        affinities_mask=affinity_mask_key,
        dtype=np.int16,
    )
    if affs_high_inter_object:
        pipeline += gp.IntensityScaleShift(affinity_key, -1, 1)
        val_pipeline += gp.IntensityScaleShift(affinity_key, -1, 1)

    if lsds:
        pipeline += AddLocalShapeDescriptor(
            gt_key,
            lsd_key,
            lsds_mask=lsd_mask_key,
            sigma=parameters.lsd_sigma,
        )
        val_pipeline += AddLocalShapeDescriptor(
            gt_key,
            lsd_key,
            lsds_mask=lsd_mask_key,
            sigma=parameters.lsd_sigma,
        )
    if fgbg:
        pipeline += Binarize(
            gt_key,
            fgbg_key,
        )
        val_pipeline += Binarize(
            gt_key,
            fgbg_key,
        )

    # Trainer attributes:
    if parameters.num_cpu_processes > 1:
        pipeline += gp.PreCache(num_workers=parameters.num_cpu_processes)

    # add channel dimensions
    pipeline += gp.Unsqueeze([raw_key, gt_key, mask_key])
    val_pipeline += gp.Unsqueeze([raw_key, gt_key, mask_key])
    if fgbg:
        pipeline += gp.Unsqueeze([fgbg_key])
        val_pipeline += gp.Unsqueeze([fgbg_key])

    # stack to create a batch dimension
    pipeline += gp.Stack(parameters.batch_size)
    val_pipeline += gp.Stack(1)

    request = gp.BatchRequest()
    request.add(raw_key, input_size)
    request.add(affinity_key, output_size)
    request.add(affinity_mask_key, output_size)
    request.add(training_mask_key, output_size)
    if lsds:
        request.add(lsd_key, output_size)
        request.add(lsd_mask_key, output_size)
    if fgbg:
        request.add(fgbg_key, output_size)
        request.add(mask_key, output_size)

    val_request = gp.BatchRequest()
    val_request[raw_key] = gp.ArraySpec(roi=val_roi.grow(context, context))
    val_request[affinity_key] = gp.ArraySpec(roi=val_roi)
    val_request[affinity_mask_key] = gp.ArraySpec(roi=val_roi)
    if lsds:
        val_request[lsd_key] = gp.ArraySpec(roi=val_roi)
        val_request[lsd_mask_key] = gp.ArraySpec(roi=val_roi)
    if fgbg:
        val_request[fgbg_key] = gp.ArraySpec(roi=val_roi)
        val_request[mask_key] = gp.ArraySpec(roi=val_roi)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw_key, input_size)
    snapshot_request.add(affinity_key, output_size)
    snapshot_request.add(affinity_mask_key, output_size)
    snapshot_request.add(training_mask_key, output_size)
    if lsds:
        snapshot_request.add(lsd_key, output_size)
        snapshot_request.add(lsd_mask_key, output_size)
    if fgbg:
        snapshot_request.add(fgbg_key, output_size)
    snapshot_request.add(mask_key, output_size)
    snapshot_request.add(gt_key, output_size)

    keys = [
        (raw_key, "image"),
        (affinity_key, "labels"),
        (affinity_mask_key, "labels"),
        (lsd_key, "image"),
        (lsd_mask_key, "labels"),
        (fgbg_key, "labels"),
        (mask_key, "labels"),
        (gt_key, "labels"),
    ]

    with gp.build(pipeline):
        with gp.build(val_pipeline):
            yield PipelineDataGenerator(
                pipeline,
                val_pipeline,
                request,
                val_request,
                snapshot_request,
                keys,
                axes=spatial_axes,
            )
