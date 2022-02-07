from magicgui import magic_factory
import napari

from affogato.segmentation.mws import compute_mws_segmentation_from_affinities

import bioimageio.core
from bioimageio.core.prediction_pipeline import create_prediction_pipeline

from xarray import DataArray

from pathlib import Path


@magic_factory
def train_affinities_widget(
    model: str,
    raw: napari.layers.Image,
    gt: napari.layers.Labels,
    lsds: bool = False,
):
    pass


@magic_factory
def predict_affinities_widget(
    model: str,
    raw: napari.layers.Image,
) -> napari.layers.Image:
    model = Path("sample_data/models/EpitheliaAffinityModel.zip")
    model = bioimageio.core.load_resource_description(model)

    offsets = model.config["mws"]["offsets"]
    ndim = len(offsets[0])

    # Assuming raw data comes in with a channel dim
    # This doesn't have to be the case, in which case
    # plugin will fail.
    # TODO: How to determine axes of raw data. metadata?
    # guess? simply make it fit what the model expects?
    raw = raw.data
    raw = raw.reshape((1, *raw.shape))

    with create_prediction_pipeline(bioimageio_model=model) as pp:
        # [0] to access first input array/output array
        raw = DataArray(raw, dims=tuple(pp.input_specs[0].axes))
        affs = pp(raw)[0].values

    # remove batch dimensions
    raw = raw.squeeze()
    affs = affs.squeeze()

    # assert result is as expected
    assert raw.ndim == ndim, f"Raw has dims: {raw.ndim}, but expected: {ndim}"
    assert affs.ndim == ndim + 1, f"Affs have dims: {affs.ndim}, but expected: {ndim+1}"
    assert affs.shape[0] == len(offsets), (
        f"Number of affinity channels ({affs.shape[0]}) "
        f"does not match number of offsets ({len(offsets)})"
    )

    # Generate affinities and keep the offsets as metadata
    return napari.layers.Image(affs, name="Affinities", metadata={"offsets": offsets})


@magic_factory
def mutex_watershed_widget(
    affinities: napari.layers.Image,
    # seeds: napari.layers.Labels,
) -> napari.layers.Labels:
    # Affinities must come with "offsets" in its metadata.

    # TODO: why must the affinity data be inverted? Constantine model wierdness

    segmentation = compute_mws_segmentation_from_affinities(
        1 - affinities.data,
        affinities.metadata["offsets"],
        beta_parameter=0.5,
        foreground_mask=None,
        edge_mask=None,
        return_valid_edge_mask=False,
    )
    return napari.layers.Labels(segmentation, name="Segmentation")
