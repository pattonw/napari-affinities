from magicgui import magic_factory
import napari

from affogato.segmentation import MWSGridGraph, compute_mws_clustering

import bioimageio.core
from bioimageio.core.prediction_pipeline import create_prediction_pipeline

import numpy as np
from xarray import DataArray

from pathlib import Path
from typing import Optional

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QLabel,
)
from napari_plugin_engine import napari_hook_implementation


class ModelWidget(QWidget):
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()

        # No model loaded by default.
        self.__model = None

        # initialize layout
        layout = QVBoxLayout()

        # Loaded model name
        self.model_label = QLabel("None")
        layout.addWidget(self.model_label)

        # Load model from file
        model_file_loader = QPushButton("Load a model from File!", self)
        model_file_loader.clicked.connect(self.model_from_file)
        layout.addWidget(model_file_loader)

        # Load model from url
        self.model_url_loader = QLineEdit("Load a model from Url!", self)
        self.model_url_loader.returnPressed.connect(self.model_from_url)
        layout.addWidget(self.model_url_loader)

        # activate layout
        self.setLayout(layout)

    @property
    def model(self) -> bioimageio.core.resource_io.io_.ResourceDescription:
        return self.__model

    @model.setter
    def model(self, new_model: bioimageio.core.resource_io.io_.ResourceDescription):
        self.__model = new_model
        self.model_label.setText(new_model.name)

    def model_from_file(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilters(["zip files (*.zip)"])
        dlg.selectNameFilter("zip files (*.zip)")

        if dlg.exec_():
            filenames = dlg.selectedFiles()
            model_file = Path(filenames[0])

            self.model = bioimageio.core.load_resource_description(model_file)

    def model_from_url(self):
        self.model = bioimageio.core.load_resource_description(
            self.model_url_loader.text()
        )


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
    raw: napari.types.ImageData,
    affinities: Optional[napari.layers.Image],
) -> napari.types.LayerDataTuple:
    model = Path("sample_data/models/EpitheliaAffinityModel.zip")
    model = bioimageio.core.load_resource_description(model)

    offsets = model.config["mws"]["offsets"]
    ndim = len(offsets[0])

    # Assuming raw data comes in with a channel dim
    # This doesn't have to be the case, in which case
    # plugin will fail.
    # TODO: How to determine axes of raw data. metadata?
    # guess? simply make it fit what the model expects?

    # add batch dimension
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
    return (
        affs,
        {
            "name": "Affinities" if affinities is None else affinities.name,
            "metadata": {"offsets": offsets},
        },
        "image",
    )


@magic_factory
def mutex_watershed_widget(
    affinities: napari.layers.Image,
    seeds: Optional[napari.layers.Labels],
    mask: Optional[napari.layers.Labels],
    previous_segmentation: Optional[napari.layers.Labels],
) -> napari.types.LayerDataTuple:

    # TODO:
    # beta slider
    # live update checkbox
    # invert affinities checkbox

    # Assumptions:
    # Affinities must come with "offsets" in its metadata.
    assert "offsets" in affinities.metadata, f"{affinities.metadata}"
    # seeds and mask must be same size as affinities if provided
    shape = affinities.data.shape[1:]
    if seeds is not None:
        assert (
            seeds.data.shape[0] == 1
        ), "Seeds should only have 1 channel but has multiple!"
        assert (
            shape == seeds.data.shape[1:]
        ), f"Got shape {seeds.data.shape[1:]} for seeds but expected {shape}"
        seeds = seeds.data
    if mask is not None:
        assert (
            mask.data.shape[0] == 1
        ), "Mask should only have 1 channel but has multiple!"
        assert (
            shape == mask.data.shape[1:]
        ), f"Got shape {seeds.data.shape} for mask but expected {shape}"

    # if a previous segmentation is provided, it must have a "grid graph"
    # in its metadata.

    grid_graph = None
    if previous_segmentation is not None:
        grid_graph = previous_segmentation.metadata["grid_graph"]
    if grid_graph is None:
        grid_graph = MWSGridGraph(shape)
        if mask is not None:
            grid_graph.set_mask(mask.data)
        if seeds is not None:
            grid_graph.update_seeds(seeds.data)

    offsets = affinities.metadata["offsets"]
    ndim = len(offsets[0])

    grid_graph.add_attractive_seed_edges = True
    uvs, weights = grid_graph.compute_nh_and_weights(
        1.0 - np.require(affinities.data[:ndim], requirements="C"), offsets[:ndim]
    )

    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
        np.require(affinities.data[ndim:], requirements="C"),
        offsets[ndim:],
        [1] * ndim,
        randomize_strides=False,
    )

    # compute the segmentation
    n_nodes = grid_graph.n_nodes
    segmentation = compute_mws_clustering(
        n_nodes, uvs, mutex_uvs, weights, mutex_weights
    )
    grid_graph.relabel_to_seeds(segmentation)
    segmentation = segmentation.reshape(shape)
    if mask is not None:
        segmentation[np.logical_not(mask)] = 0

    return (
        segmentation,
        {
            "name": "Segmentation"
            if previous_segmentation is None
            else previous_segmentation.name
        },
    )
