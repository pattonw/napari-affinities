# local package imports
from copy import deepcopy
from ..gp.pipeline import build_pipeline
from .gui_helpers import layer_choice_widget
from ..bioimageio.helpers import get_torch_module

# github repo libraries
import gunpowder as gp

# pip installed libraries
import napari
from napari.qt.threading import thread_worker
from magicgui.widgets import create_widget, Container
import bioimageio.core
from bioimageio.core.build_spec import build_model
from bioimageio.core.resource_io.nodes import Model
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from marshmallow import missing
import torch
import numpy as np
from xarray import DataArray
from superqt import QCollapsible
from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QInputDialog,
    QLabel,
    QFrame,
)

# python built in libraries
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import contextmanager
import dataclasses


class ModelWidget(QWidget):
    def __init__(self, napari_viewer):
        # basic initialization
        self.viewer = napari_viewer
        super().__init__()

        # initialize state variables
        self.__training_generator = None

        # supported axes
        self.__axes = ["batch", "channel", "time", "z", "y", "x"]

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

        # Save model to file
        self.save_button = QPushButton("Save model!", self)
        self.save_button.clicked.connect(self.save)
        layout.addWidget(self.save_button)

        # Train widget(Collapsable)
        collapsable_train_widget = QCollapsible("Training: expand for options:", self)
        self.train_widget = self.create_train_widget(napari_viewer)
        collapsable_train_widget.addWidget(
            self.train_widget.native
        )  # FunctionGui -> QWidget via .native

        # add loss/iterations widget
        iterations_frame = QFrame(collapsable_train_widget)
        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("iterations:")
        self.iterations_widget = QLabel("None")
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations_widget)
        iterations_frame.setLayout(iterations_layout)
        collapsable_train_widget.addWidget(iterations_frame)

        loss_frame = QFrame(collapsable_train_widget)
        loss_layout = QHBoxLayout()
        loss_label = QLabel("loss:")
        self.loss_widget = QLabel("nan")
        loss_layout.addWidget(loss_label)
        loss_layout.addWidget(self.loss_widget)
        loss_frame.setLayout(loss_layout)
        collapsable_train_widget.addWidget(loss_frame)

        # add buttons
        self.reset_training_state()
        self.train_button = QPushButton("Train!", self)
        self.train_button.clicked.connect(self.train)
        self.pause_button = QPushButton("Pause!", self)
        self.pause_button.clicked.connect(self.pause_training)
        self.snapshot_button = QPushButton("Snapshot!", self)
        self.snapshot_button.clicked.connect(self.snapshot)
        self.async_predict_button = QPushButton("Predict!", self)
        self.async_predict_button.clicked.connect(self.async_predict)
        self.update_button = QPushButton("Update Model!", self)
        self.update_button.clicked.connect(self.update)
        collapsable_train_widget.addWidget(self.train_button)
        collapsable_train_widget.addWidget(self.pause_button)
        collapsable_train_widget.addWidget(self.snapshot_button)
        collapsable_train_widget.addWidget(self.async_predict_button)
        collapsable_train_widget.addWidget(self.update_button)

        layout.addWidget(collapsable_train_widget)

        # Predict widget(Collapsable)
        collapsable_predict_widget = QCollapsible(
            "Prediction: expand for options:", self
        )
        self.predict_widget = self.create_predict_widget(napari_viewer)
        collapsable_predict_widget.addWidget(self.predict_widget.native)

        # add buttons
        self.predict_button = QPushButton("Predict!", self)
        self.predict_button.clicked.connect(self.predict)
        collapsable_predict_widget.addWidget(self.predict_button)

        layout.addWidget(collapsable_predict_widget)

        # activate layout
        self.setLayout(layout)

        # Widget state
        self.model = None

        # No buttons should be enabled
        self.disable_buttons(
            train=True,
            pause=True,
            snapshot=True,
            async_predict=True,
            update=True,
            predict=True,
            save=True,
        )

    @property
    def model(self) -> Optional[Model]:
        return self.__model

    @contextmanager
    def build_pipeline(self, raw, gt, mask, lsds):
        with build_pipeline(raw, gt, mask, lsds, self.model) as pipeline:
            yield pipeline

    def reset_training_state(self):
        if self.__training_generator is not None:
            self.__training_generator.quit()
        self.__training_generator = None
        self.iterations_widget.setText("None")
        self.loss_widget.setText("nan")

    def disable_buttons(
        self,
        train: bool = False,
        pause: bool = False,
        snapshot: bool = False,
        async_predict: bool = False,
        update: bool = False,
        predict: bool = False,
        save: bool = False,
    ):
        self.train_button.setEnabled(not train)
        self.pause_button.setEnabled(not pause)
        self.snapshot_button.setEnabled(not snapshot)
        self.async_predict_button.setEnabled(not async_predict)
        self.update_button.setEnabled(not update)
        self.predict_button.setEnabled(not predict)
        self.save_button.setEnabled(not save)

    @model.setter
    def model(self, new_model: Optional[Model]):
        self.reset_training_state()
        self.__model = new_model
        if new_model is not None:
            self.model_label.setText(new_model.name)
            self.disable_buttons(
                pause=True,
                snapshot=True,
                async_predict=True,
                update=True,
                save=True,
            )
        else:
            self.model_label.setText("None")

    def start_training_loop(self):
        self.__training_generator = self.train_affinities(
            self.train_widget.raw.value,
            self.train_widget.gt.value,
            self.train_widget.mask.value,
            self.train_widget.lsds.value,
        )
        self.__training_generator.yielded.connect(self.on_yield)
        self.__training_generator.returned.connect(self.on_return)
        self.__training_generator.start()

    def train(self):
        self.disable_buttons(
            train=True,
        )
        if self.__training_generator is None:
            self.start_training_loop()
        else:
            self.__training_generator.resume()

    def pause_training(self):
        self.disable_buttons(pause=True)
        self.__training_generator.pause()

    def snapshot(self):
        self.disable_buttons(train=True)
        if self.__training_generator is None:
            self.train()
        self.__training_generator.send("snapshot")
        self.__training_generator.resume()

    def async_predict(self):
        self.disable_buttons(train=True)
        if self.__training_generator is None:
            self.train()
        self.__training_generator.send(
            "predict",
        )
        self.__training_generator.resume()

    def spatial_dims(self, ndims):
        return ["time", "z", "y", "x"][-ndims:]

    def predict(self):
        """
        Predict on data provided through the predict widget. Not necessarily the
        same as the training data.
        """
        offsets = self.model.config["mws"]["offsets"]
        ndim = len(offsets[0])
        spatial_axes = self.spatial_dims(ndim)
        affs = self._predict(self.model, self.predict_widget.raw.value.data[:], offsets)
        self.add_layers(
            [
                (
                    affs,
                    {
                        "name": "Affinities",
                        "metadata": {"offsets": offsets},
                        "axes": (
                            "channel",
                            *spatial_axes,
                        ),
                    },
                    "image",
                ),
            ]
        )

    def _predict(self, model, raw_data, offsets):
        ndim = len(offsets[0])

        # add batch and potentially channel dimensions
        assert len(raw_data.shape) >= ndim, (
            f"raw data has {len(raw_data.shape)} dimensions but "
            f"should have {ndim} spatial dimensions"
        )
        while len(raw_data.shape) < ndim + 2:
            raw_data = raw_data.reshape((1, *raw_data.shape))

        with create_prediction_pipeline(bioimageio_model=model) as pp:
            # [0] to access first input array/output array
            pred_data = DataArray(raw_data, dims=tuple(pp.input_specs[0].axes))
            affs = pp(pred_data)[0].values

        # remove batch dimensions
        pred_data = pred_data.squeeze()
        affs = affs.squeeze()

        # assert result is as expected
        assert (
            pred_data.ndim == ndim
        ), f"Raw has dims: {pred_data.ndim}, but expected: {ndim}"
        assert (
            affs.ndim == ndim + 1
        ), f"Affs have dims: {affs.ndim}, but expected: {ndim+1}"
        assert affs.shape[0] == len(offsets), (
            f"Number of affinity channels ({affs.shape[0]}) "
            f"does not match number of offsets ({len(offsets)})"
        )
        return affs

    def update(self):
        """
        End training loop, update loaded model with new weights,
        reset iterations/loss
        """
        if self.__training_generator is None:
            self.train()
        self.__training_generator.resume()
        self.__training_generator.send("stop")
        self.disable_buttons(
            pause=True,
            snapshot=True,
            async_predict=True,
            update=True,
            save=True,
        )
        self.reset_training_state()

    def save(self):
        """
        Save model to file
        """

        # get architecture source
        def get_architecture_source():
            raw_resource = bioimageio.core.load_raw_resource_description(self.__rdf)
            model_source = raw_resource.weights["pytorch_state_dict"].architecture
            # download the source file if necessary
            source_file = bioimageio.core.resource_io.utils.resolve_source(
                model_source.source_file
            )
            # if the source file path does not exist, try combining it with the root path of the model
            if not Path(source_file).exists():
                source_file = Path(
                    raw_resource.root_path,
                    Path(source_file).relative_to(Path(".").absolute()),
                )
            assert Path(source_file).exists(), source_file
            class_name = model_source.callable_name
            return f"{source_file}:{class_name}"

        # the path to save the new model with torchscript weights
        zip_path = f"saved_model.zip"

        preprocessing = [
            [{"name": prep.name, "kwargs": prep.kwargs} for prep in inp.preprocessing]
            for inp in self.model.inputs
            if inp.preprocessing != missing
        ]
        postprocessing = [
            [{"name": post.name, "kwargs": post.kwargs} for post in outp.postprocessing]
            if outp.postprocessing != missing
            else None
            for outp in self.model.outputs
        ]
        citations = [
            {k: v for k, v in dataclasses.asdict(citation).items() if v != missing}
            for citation in self.model.cite
        ]

        kwargs = {
            "weight_uri": self.model.weights["pytorch_state_dict"].source,
            "test_inputs": self.model.test_inputs,
            "test_outputs": self.model.test_outputs,
            "input_axes": ["".join(inp.axes) for inp in self.model.inputs],
            "input_min_shape": [inp.shape.min for inp in self.model.inputs],
            "input_step": [inp.shape.step for inp in self.model.inputs],
            "output_axes": ["".join(outp.axes) for outp in self.model.outputs],
            "output_path": zip_path,
            "name": self.model.name,
            "description": f"{self.model.description}\nFinetuned with the napari-affinities plugin!",
            "authors": [dataclasses.asdict(author) for author in self.model.authors],
            "license": self.model.license,
            "documentation": self.model.documentation,
            "covers": self.model.covers,
            "tags": self.model.tags,
            "cite": citations,
            "parent": self.model.parent,
            "architecture": get_architecture_source(),
            "model_kwargs": self.model.weights["pytorch_state_dict"].kwargs,
            "preprocessing": preprocessing,
            "postprocessing": postprocessing,
            "training_data": self.model.training_data
            if self.model.training_data != missing
            else None,
            "config": self.model.config,
        }

        # build the model! it will be saved to 'zip_path'
        new_model_raw = build_model(**kwargs)

    def create_train_widget(self, viewer):
        # inputs:
        raw = layer_choice_widget(
            viewer,
            annotation=napari.layers.Image,
            name="raw",
        )
        gt = layer_choice_widget(viewer, annotation=napari.layers.Labels, name="gt")
        mask = layer_choice_widget(
            viewer, annotation=Optional[napari.layers.Labels], name="mask"
        )
        lsds = create_widget(
            annotation=bool,
            name="lsds",
            label='<a href="https://localshapedescriptors.github.io"><font color=white>LSDs</font></a>',
            value=False,
        )

        train_widget = Container(widgets=[raw, gt, mask, lsds])

        return train_widget

    def create_predict_widget(self, viewer):
        # inputs:
        raw = layer_choice_widget(
            viewer,
            annotation=napari.layers.Image,
            name="raw",
        )

        predict_widget = Container(widgets=[raw])

        return predict_widget

    def model_from_file(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilters(["zip files (*.zip)"])
        dlg.selectNameFilter("zip files (*.zip)")

        if dlg.exec_():
            filenames = dlg.selectedFiles()
            model_file = Path(filenames[0])

            self.__rdf = model_file
            self.model = bioimageio.core.load_resource_description(model_file)

    def model_from_url(self):
        # TODO: async
        url, ok = QInputDialog.getText(
            self, "Url Input Dialog", "Url for model rdf.yaml folder:"
        )
        if ok:
            self.__rdf = url
            self.model = bioimageio.core.load_resource_description(url)

    def on_yield(self, step_data):
        iteration, loss, *layers = step_data
        if len(layers) > 0:
            self.add_layers(layers)
        if iteration is not None and loss is not None:
            self.iterations_widget.setText(f"{iteration}")
            self.loss_widget.setText(f"{loss}")

    def on_return(self, weights_path: Path):
        """
        Update model to use provided returned weights
        """
        self.model.weights["pytorch_state_dict"].source = weights_path
        self.reset_training_state()

    def add_layers(self, layers):
        viewer_axis_labels = self.viewer.dims.axis_labels

        for data, metadata, layer_type in layers:
            # then try to update the viewer layer with that name.
            name = metadata.pop("name")
            axes = metadata.pop("axes")

            # handle viewer axes if still default numerics
            # TODO: Support using xarray axis labels as soon as napari does
            if len(set(viewer_axis_labels).intersection(set(axes))) == 0:
                spatial_axes = [
                    axis for axis in axes if axis not in ["batch", "channel"]
                ]
                assert (
                    len(viewer_axis_labels) - len(spatial_axes) <= 1
                ), f"Viewer has axes: {viewer_axis_labels}, but we expect ((channels), {spatial_axes})"
                viewer_axis_labels = (
                    ("channels", *spatial_axes)
                    if len(viewer_axis_labels) > len(spatial_axes)
                    else spatial_axes
                )
                self.viewer.dims.axis_labels = viewer_axis_labels

            batch_dim = axes.index("batch") if "batch" in axes else -1
            sample_shape = data.shape[batch_dim + 1 :]

            try:
                # add to existing layer
                layer = self.viewer.layers[name]

                # concatenate along batch dimension
                layer.data = np.concatenate(
                    [
                        layer.data.reshape(*(-1, *sample_shape)),
                        data.reshape(-1, *sample_shape),
                    ],
                    axis=0,
                )

                # if make first dimension "batch" if it isn't
                if viewer_axis_labels[0] != "batch":
                    viewer_axis_labels = ("batch", *viewer_axis_labels)
                    self.viewer.dims.axis_labels = viewer_axis_labels

            except KeyError:  # layer not in the viewer
                # TODO: Support defining layer axes as soon as napari does
                if layer_type == "image":
                    self.viewer.add_image(data, name=name, **metadata)
                elif layer_type == "labels":
                    self.viewer.add_labels(data, name=name, **metadata)

    @thread_worker
    def train_affinities(self, raw, gt, mask, lsds) -> int:

        iteration = 0
        if self.model is None:
            raise ValueError("Please load a model either from your filesystem or a url")

        model = deepcopy(self.model)

        # constants
        offsets = model.config["mws"]["offsets"]
        ndim = len(offsets[0])
        spatial_axes = self.spatial_dims(ndim)

        # extract torch model from bioimageio format
        torch_module = get_torch_module(model)

        # define Loss function and Optimizer TODO: make options available as choices?
        lsd_loss_func = torch.nn.MSELoss()
        # aff_loss_func = torch.nn.BCEWithLogitsLoss()
        aff_loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=torch_module.parameters())

        # TODO: How to display profiling stats
        device = torch.device("cuda")
        torch_module = torch_module.to(device)
        torch_module.train()

        # prepare data for full volume prediction
        raw_data = raw.data
        # add batch dimension
        raw_data = raw_data.reshape((1, *raw_data.shape))

        # Train loop:
        with self.build_pipeline(raw, gt, mask, lsds) as pipeline:
            iteration = 0
            loss = float("nan")
            mode = yield (iteration, loss)
            while True:

                if mode == "predict":
                    checkpoint = Path(f"/tmp/checkpoints/{iteration}.pt")
                    if not checkpoint.parent.exists():
                        checkpoint.parent.mkdir(parents=True)
                    torch.save(torch_module.state_dict(), checkpoint)
                    model.weights["pytorch_state_dict"].source = checkpoint

                    # Assuming raw data comes in with a channel dim
                    # This doesn't have to be the case, in which case
                    # plugin will fail.
                    # TODO: How to determine axes of raw data. metadata?
                    # guess? simply make it fit what the model expects?

                    affs = self._predict(model, raw_data, offsets)

                    # Generate affinities and keep the offsets as metadata
                    mode = yield (
                        None,
                        None,
                        (
                            affs,
                            {
                                "name": "Affinities",
                                "metadata": {"offsets": offsets},
                                "axes": (
                                    "channel",
                                    *spatial_axes,
                                ),
                            },
                            "image",
                        ),
                    )
                elif mode is None or mode == "snapshot":
                    snapshot_iteration = mode == "snapshot"

                    # fetch data:
                    arrays, snapshot_arrays = pipeline.next(snapshot_iteration)
                    tensors = [
                        torch.as_tensor(array[0], device=device).float()
                        for array in arrays
                    ]
                    raw, aff_target, aff_mask, *lsd_arrays = tensors

                    optimizer.zero_grad()
                    if lsds:
                        lsd_target, lsd_mask = lsd_arrays

                        aff_pred, lsd_pred = torch_module(raw)

                        aff_loss = aff_loss_func(
                            aff_pred * aff_mask,
                            aff_target * aff_mask,
                        )
                        lsd_loss = lsd_loss_func(
                            lsd_pred * lsd_mask, lsd_target * lsd_mask
                        )
                        loss = aff_loss * lsd_loss
                    else:
                        aff_pred = torch_module(raw)
                        loss = aff_loss_func(
                            aff_pred * aff_mask,
                            aff_target * aff_mask,
                        )

                    loss.backward()
                    optimizer.step()
                    iteration += 1

                    if snapshot_iteration:
                        pred_arrays = []
                        pred_arrays.append(
                            (
                                aff_pred.detach().cpu().numpy(),
                                {
                                    "name": "sample_aff_pred",
                                    "axes": (
                                        "batch",
                                        "channel",
                                        *spatial_axes,
                                    ),
                                },
                                "image",
                            )
                        )
                        if lsds:
                            pred_arrays.append(
                                (
                                    lsd_pred.detach().cpu().numpy(),
                                    {
                                        "name": "sample_lsd_pred",
                                        "axes": (
                                            "batch",
                                            "channel",
                                            *spatial_axes,
                                        ),
                                    },
                                    "image",
                                )
                            )
                        mode = yield (
                            iteration,
                            loss,
                            *arrays,
                            *snapshot_arrays,
                            *pred_arrays,
                        )
                    else:
                        mode = yield (iteration, loss)
                elif mode == "stop":
                    checkpoint = Path(f"/tmp/checkpoints/{iteration}.pt")
                    if not checkpoint.parent.exists():
                        checkpoint.parent.mkdir(parents=True)
                    torch.save(torch_module.state_dict(), checkpoint)
                    return checkpoint
                else:
                    raise ValueError(
                        f"Unknown message passed to train worker: ({mode})"
                    )
