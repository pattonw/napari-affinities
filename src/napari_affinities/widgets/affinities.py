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
from magicgui.widgets import create_widget, Container, Label
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
        self.train_button = QPushButton("Train!", self)
        self.train_button.clicked.connect(self.train)
        self.snapshot_button = QPushButton("Snapshot!", self)
        self.snapshot_button.clicked.connect(self.snapshot)
        self.async_predict_button = QPushButton("Predict(online)!", self)
        self.async_predict_button.clicked.connect(self.async_predict)
        collapsable_train_widget.addWidget(self.train_button)
        collapsable_train_widget.addWidget(self.snapshot_button)
        collapsable_train_widget.addWidget(self.async_predict_button)

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

        # Save widget(Collapsable)
        collapsable_save_widget = QCollapsible("Save Model: expand for options:", self)
        self.save_widget = self.create_save_widget(napari_viewer)
        collapsable_save_widget.addWidget(self.save_widget.native)

        # add buttons
        self.save_button = QPushButton("save!", self)
        self.save_button.clicked.connect(self.save)
        collapsable_save_widget.addWidget(self.save_button)

        layout.addWidget(collapsable_save_widget)

        # activate layout
        self.setLayout(layout)

        # Widget state
        self.model = None

        # No buttons should be enabled
        self.disable_buttons(
            train=True,
            snapshot=True,
            async_predict=True,
            predict=True,
            save=True,
        )

    @property
    def model(self) -> Optional[Model]:
        return self.__model

    @model.setter
    def model(self, new_model: Optional[Model]):
        self.reset_training_state()
        self.__model = new_model
        if new_model is not None:
            self.model_label.setText(new_model.name)
            self.disable_buttons(
                snapshot=True,
                async_predict=True,
            )
        else:
            self.model_label.setText("None")

    @property
    def training(self) -> bool:
        try:
            return self.__training
        except AttributeError:
            return False

    @training.setter
    def training(self, training: bool):
        self.__training = training
        if training:
            if self.__training_generator is None:
                self.start_training_loop()
            assert self.__training_generator is not None
            self.__training_generator.resume()
            self.train_button.setText("Pause!")
            self.disable_buttons()
        else:
            if self.__training_generator is not None:
                self.__training_generator.send("stop")
            self.train_button.setText("Train!")
            self.disable_buttons(snapshot=True, async_predict=True)

    @contextmanager
    def build_pipeline(self, raw, gt, mask, lsds):
        with build_pipeline(raw, gt, mask, lsds, self.model) as pipeline:
            yield pipeline

    def reset_training_state(self, keep_stats=False):
        if self.__training_generator is not None:
            self.__training_generator.quit()
        self.__training_generator = None
        if not keep_stats:
            self.iteration = 0
            self.iterations_widget.setText("None")
            self.loss_widget.setText("nan")

    def disable_buttons(
        self,
        train: bool = False,
        snapshot: bool = False,
        async_predict: bool = False,
        predict: bool = False,
        save: bool = False,
    ):
        self.train_button.setEnabled(not train)
        self.snapshot_button.setEnabled(not snapshot)
        self.async_predict_button.setEnabled(not async_predict)
        self.predict_button.setEnabled(not predict)
        self.save_button.setEnabled(not save)

    def start_training_loop(self):
        self.__training_generator = self.train_affinities(
            self.train_widget.raw.value,
            self.train_widget.gt.value,
            self.train_widget.mask.value,
            # self.train_widget.lsds.value,
            iteration=self.iteration,
        )
        self.__training_generator.yielded.connect(self.on_yield)
        self.__training_generator.returned.connect(self.on_return)
        self.__training_generator.start()

        # all buttons are enabled while the training loop is running
        self.disable_buttons()

    def train(self):
        self.training = not self.training

    def snapshot(self):
        self.__training_generator.send("snapshot")
        self.training = True

    def async_predict(self):
        self.__training_generator.send(
            "predict",
        )
        self.training = True

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
                        "metadata": {"offsets": offsets, "overwrite": True},
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

        print("Predicting with weights: ", model.weights["pytorch_state_dict"].source)
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
        zip_path = self.save_widget.filename.value
        assert zip_path.name.endswith(".zip"), "Must save model in a zip"

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
        authors = [dataclasses.asdict(author) for author in self.model.authors]
        if (
            self.save_widget.author.value is not None
            and len(self.save_widget.author.value) > 0
        ):
            authors += [{"name": self.save_widget.author.value}]
        name = (
            self.save_widget.model_name.value
            if self.save_widget.model_name.value is not None
            and len(self.save_widget.model_name.value) > 0
            else self.model.name
        )

        kwargs = {
            "weight_uri": self.model.weights["pytorch_state_dict"].source,
            "test_inputs": self.model.test_inputs,
            "test_outputs": self.model.test_outputs,
            "input_axes": ["".join(inp.axes) for inp in self.model.inputs],
            "input_min_shape": [inp.shape.min for inp in self.model.inputs],
            "input_step": [inp.shape.step for inp in self.model.inputs],
            "output_axes": ["".join(outp.axes) for outp in self.model.outputs],
            "output_path": zip_path,
            "name": name,
            "description": f"{self.model.description}\nFinetuned with the napari-affinities plugin!",
            "authors": authors,
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
        lsd_label = Label(
            name="lsd_label",
            label='<a href="https://localshapedescriptors.github.io"><font color=white>LSDs</font></a>',
        )
        use_lsds = create_widget(
            annotation=bool,
            name="lsds",
            label="use LSDs",
            value=False,
        )
        sigma = create_widget(annotation=float, name="sigma", label="sigma", value=0)
        lsds = Container(widgets=[lsd_label, use_lsds, sigma], name="lsds")

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

    def create_save_widget(self, viewer):
        # inputs:
        filename = create_widget(
            annotation=Path, name="filename", options={"mode": "w"}
        )
        author = create_widget(annotation=Optional[str], name="author")
        model_name = create_widget(annotation=Optional[str], name="model_name")
        save_widget = Container(widgets=[filename, author, model_name])

        return save_widget

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
            self.iteration = iteration
            self.iterations_widget.setText(f"{iteration}")
            self.loss_widget.setText(f"{loss}")

    def on_return(self, weights_path: Path):
        """
        Update model to use provided returned weights
        """
        assert self.model is not None
        self.model.weights["pytorch_state_dict"].source = weights_path
        self.reset_training_state(keep_stats=True)

    def add_layers(self, layers):
        viewer_axis_labels = self.viewer.dims.axis_labels

        for data, metadata, layer_type in layers:
            # then try to update the viewer layer with that name.
            name = metadata.pop("name")
            axes = metadata.pop("axes")
            overwrite = metadata.pop("overwrite", False)

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
                if overwrite:
                    layer.data = data.reshape(*sample_shape)
                else:
                    layer.data = np.concatenate(
                        [
                            layer.data.reshape(*(-1, *sample_shape)),
                            data.reshape(-1, *sample_shape),
                        ],
                        axis=0,
                    )

                # make first dimension "batch" if it isn't
                if not overwrite and viewer_axis_labels[0] != "batch":
                    viewer_axis_labels = ("batch", *viewer_axis_labels)
                    self.viewer.dims.axis_labels = viewer_axis_labels

            except KeyError:  # layer not in the viewer
                # TODO: Support defining layer axes as soon as napari does
                if layer_type == "image":
                    self.viewer.add_image(data, name=name, **metadata)
                elif layer_type == "labels":
                    self.viewer.add_labels(data, name=name, **metadata)

    @thread_worker
    def train_affinities(self, raw, gt, mask, lsds=False, iteration=0):

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
                                "name": "Affinities(online)",
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
