from .gp import NapariImageSource, NapariLabelsSource, OnesSource

from magicgui import magic_factory, magicgui
import napari

import gunpowder as gp
from affogato.segmentation import MWSGridGraph, compute_mws_clustering
from lsd.gp import AddLocalShapeDescriptor
import bioimageio.core
from bioimageio.core.resource_io.nodes import Model, ImportedSource
from bioimageio.core.prediction_pipeline import create_prediction_pipeline

from marshmallow import missing
import torch
import numpy as np
from xarray import DataArray
from superqt import QCollapsible
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QInputDialog,
    QLabel,
)


from pathlib import Path
from typing import Optional


class ModelWidget(QWidget):
    def __init__(self, napari_viewer):
        # basic initialization
        self.viewer = napari_viewer
        super().__init__()

        # Widget layout
        layout = QVBoxLayout()

        # Model name Label Widget
        self.model_label = QLabel()
        layout.addWidget(self.model_label)

        # Load model from file Widget
        model_file_loader = QPushButton("Load a model from File!", self)
        model_file_loader.clicked.connect(self.model_from_file)
        layout.addWidget(model_file_loader)

        # Load model from url Widget
        model_url_loader = QPushButton("Load a model from Url!", self)
        model_url_loader.clicked.connect(self.model_from_url)
        layout.addWidget(model_url_loader)

        # Train widget(Collapsable), use magicgui to avoid having to make layer dropdowns myself
        # Magicgui can't update layer dropdowns automatically anymore so add event listener
        collapsable_train_widget = QCollapsible("Training: expand for options:", self)
        train_widget = magicgui(
            train_affinities_widget,
            call_button="Train Affinities",
            parent={"bind": self},
        )
        train_widget.raw.reset_choices()
        napari_viewer.layers.events.inserted.connect(train_widget.raw.reset_choices)
        train_widget.gt.reset_choices()
        napari_viewer.layers.events.inserted.connect(train_widget.gt.reset_choices)
        train_widget.mask.reset_choices()
        napari_viewer.layers.events.inserted.connect(train_widget.mask.reset_choices)
        collapsable_train_widget.addWidget(
            train_widget.native
        )  # FunctionGui -> QWidget via .native
        layout.addWidget(collapsable_train_widget)

        # Predict widget(Collapsable), use magicgui to avoid having to make layer dropdowns myself
        # Magicgui can't update layer dropdowns automatically anymore so add event listener
        collapsable_predict_widget = QCollapsible(
            "Prediction: expand for options:", self
        )
        predict_widget = magicgui(
            predict_affinities_widget,
            call_button="Predict Affinities",
            parent={"bind": self},
        )
        predict_widget.raw.reset_choices()
        napari_viewer.layers.events.inserted.connect(predict_widget.raw.reset_choices)
        collapsable_predict_widget.addWidget(
            predict_widget.native
        )  # FunctionGui -> QWidget via .native
        layout.addWidget(collapsable_predict_widget)

        # activate layout
        self.setLayout(layout)

        # Widget state
        self.model = None

    @property
    def model(self) -> Optional[Model]:
        return self.__model

    @model.setter
    def model(self, new_model: Optional[Model]):
        self.__model = new_model
        if new_model is not None:
            self.model_label.setText(new_model.name)
        else:
            self.model_label.setText("None")

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
        # TODO: async
        url, ok = QInputDialog.getText(
            self, "Url Input Dialog", "Url for model rdf.yaml folder:"
        )
        if ok:
            self.model = bioimageio.core.load_resource_description(url)


def get_nn_instance(model_node: Model, **kwargs):
    """
    Get python torch model from a bioimage.io `Model` class
    copied from:
    https://github.com/bioimage-io/core-bioimage-io-python/blob/3364875eec581b5cd5950441915aa00219bbaf18/
    bioimageio/core/prediction_pipeline/_model_adapters/_pytorch_model_adapter.py#L54
    """
    # TODO: This is torch specific. Bioimage-io models support many more
    # model frameworks. How to handle non-torch models still needs to be
    # handled
    # Most notebooks/code I could find related to loading a bioimage-io model
    # worked under the assumption that you knew what model, and thus what
    # framework you would be using

    weight_spec = model_node.weights.get("pytorch_state_dict")
    assert weight_spec is not None
    assert isinstance(weight_spec.architecture, ImportedSource)
    model_kwargs = weight_spec.kwargs
    joined_kwargs = {} if model_kwargs is missing else dict(model_kwargs)
    joined_kwargs.update(kwargs)
    return weight_spec.architecture(**joined_kwargs)


def train_affinities_widget(
    parent: QWidget,
    raw: napari.layers.Image,
    gt: napari.layers.Labels,
    mask: Optional[napari.layers.Labels] = None,
    lsds: bool = False,
    num_iterations: int = 1000,
) -> napari.types.LayerDataTuple:
    # get currently loaded model
    model = parent.model
    if model is None:
        raise ValueError("Please load a model either from your filesystem or a url")

    # Get necessary metadata:
    offsets = model.config["mws"]["offsets"]
    snapshot_interval = None
    num_cpu_processes = 1
    batch_size = 1
    input_shape = (100, 100)  # TODO: read from metadata
    output_shape = (100, 100)  # TODO: read from metadata

    # extract torch model from bioimageio format
    torch_model = get_nn_instance(model)

    # define Loss function and Optimizer TODO: make options available as choices?
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=torch_model.parameters())

    ##############################################################################
    ########################## Create Training Pipeline ##########################
    ##############################################################################

    # define shapes:
    input_shape = gp.Coordinate(input_shape)
    output_shape = gp.Coordinate(output_shape)

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
    affinity_weight_key = gp.ArrayKey("WEIGHT")
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
            sigma=3.0 * voxel_size,
        )

    # balance loss weight between positive and negative affinities
    pipeline += gp.BalanceLabels(affinity_key, affinity_weight_key)

    # Trainer attributes:
    if num_cpu_processes > 1:
        pipeline += gp.PreCache(num_workers=num_cpu_processes)

    # stack to create a batch dimension
    pipeline += gp.Stack(batch_size)

    # TODO: How to display profiling stats

    # Train loop:
    with gp.build(pipeline):
        for iteration in range(num_iterations):
            # create request
            request = gp.BatchRequest()
            request.add(raw_key, input_size)
            request.add(affinity_key, output_size)
            request.add(affinity_mask_key, output_size)
            request.add(affinity_weight_key, output_size)
            if lsds:
                request.add(lsd_key, output_size)
                request.add(lsd_mask_key, output_size)
            # request additional keys for snapshots
            if snapshot_interval is not None and iteration % snapshot_interval == 0:
                request.add(gt_key, output_size)
                request.add(mask_key, output_size)

            # fetch data:
            batch = pipeline.request_batch(request)
            device = torch.device("cuda")
            raw = torch.as_tensor(batch[raw_key].data, device=device)
            aff_target = torch.as_tensor(batch[affinity_key].data, device=device)
            aff_mask = torch.as_tensor(batch[affinity_mask_key].data, device=device)
            aff_weight = torch.as_tensor(batch[affinity_weight_key].data, device=device)
            torch_model = torch_model.to(device)

            optimizer.zero_grad()
            if lsds:
                lsd_target = torch.as_tensor(batch[lsd_key], device=device)
                lsd_mask = torch.as_tensor(batch[lsd_mask_key], device=device)

                aff_pred, lsd_pred = torch_model(raw)

                aff_loss = loss_func(
                    aff_pred * aff_mask * aff_weight, aff_target * aff_mask * aff_weight
                )
                lsd_loss = loss_func(lsd_pred * lsd_mask, lsd_target * lsd_mask)
                loss = aff_loss * lsd_loss
            else:
                aff_pred = torch_model(raw)
                loss = loss_func(
                    aff_pred * aff_mask * aff_weight, aff_target * aff_mask * aff_weight
                )

            if snapshot_interval is not None and iteration % snapshot_interval == 0:
                # TODO: return layers live?
                pass

            loss.backward()
            optimizer.step()


def predict_affinities_widget(
    parent: QWidget,
    raw: napari.types.ImageData,
) -> napari.types.LayerDataTuple:

    model = parent.model

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
            "name": "Affinities",
            "metadata": {"offsets": offsets},
        },
        "image",
    )


@magic_factory
def mutex_watershed_widget(
    affinities: napari.layers.Image,
    seeds: Optional[napari.layers.Labels],
    mask: Optional[napari.layers.Labels],
    invert_affinities: bool,
) -> napari.types.LayerDataTuple:
    # TODO: Add Keybindings

    # TODO:
    # beta slider
    # live update checkbox (simple layer onchange eventlistener similar to
    # the layer dropdown choice reloader on the model widget)

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
    grid_graph = MWSGridGraph(shape)
    if mask is not None:
        grid_graph.set_mask(mask.data)
    if seeds is not None:
        grid_graph.update_seeds(seeds.data)

    offsets = affinities.metadata["offsets"]
    ndim = len(offsets[0])

    grid_graph.add_attractive_seed_edges = True
    affs = np.require(affinities.data[:ndim], requirements="C")
    affs = 1 - affs if invert_affinities else affs
    uvs, weights = grid_graph.compute_nh_and_weights(affs, offsets[:ndim])

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
        {"name": "Segmentation"},
    )
