# local package imports
from copy import deepcopy
from ..gp.pipeline import GunpowderParameters, build_pipeline
from .gui_helpers import layer_choice_widget, MplCanvas
from ..bioimageio.helpers import get_torch_module

# github repo libraries
import gunpowder as gp

# pip installed libraries
import napari
from napari.qt.threading import thread_worker
import bioimageio.core
from bioimageio.core.build_spec import build_model
from bioimageio.core.resource_io.nodes import Model
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.prediction import predict_with_tiling
from marshmallow import missing
import torch
import numpy as np
from xarray import DataArray

# widget stuff
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from magicgui.widgets import create_widget, Container, Label
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
        self._validation_interval = 100
        self.__model = None

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
        collapsable_train_widget = QCollapsible(
            "Training: expand for options:", self
        )
        self.train_widget = self.create_train_widget(napari_viewer)
        collapsable_train_widget.addWidget(
            self.train_widget.native
        )  # FunctionGui -> QWidget via .native

        # add loss/iterations widget
        self.progress_plot = MplCanvas(self, width=5, height=3, dpi=100)
        toolbar = NavigationToolbar(self.progress_plot, self)
        progress_plot_layout = QVBoxLayout()
        progress_plot_layout.addWidget(toolbar)
        progress_plot_layout.addWidget(self.progress_plot)
        self.loss_plot = None
        self.val_plot = None
        plot_container_widget = QWidget()
        plot_container_widget.setLayout(progress_plot_layout)
        collapsable_train_widget.addWidget(plot_container_widget)

        # add buttons
        self.train_button = QPushButton("Train!", self)
        self.train_button.clicked.connect(self.train)
        self.async_predict_button = QPushButton("Predict(online)!", self)
        self.async_predict_button.clicked.connect(self.async_predict)
        collapsable_train_widget.addWidget(self.train_button)
        collapsable_train_widget.addWidget(self.async_predict_button)

        layout.addWidget(collapsable_train_widget)

        # add advanced dropdown
        advanced_options = QCollapsible(
            "Training(Advanced): expand for options:", self
        )
        self.advanced_widget = self.create_advanced_widget(napari_viewer)
        advanced_options.addWidget(
            self.advanced_widget.native
        )  # FunctionGui -> QWidget via .native

        # add buttons
        self.snapshot_button = QPushButton("Snapshot!", self)
        self.snapshot_button.clicked.connect(self.snapshot)
        advanced_options.addWidget(self.snapshot_button)

        layout.addWidget(advanced_options)

        # Predict widget(Collapsable)
        collapsable_predict_widget = QCollapsible(
            "Prediction: expand for options:", self
        )
        self.predict_widget = self.create_predict_widget(napari_viewer)
        collapsable_predict_widget.addWidget(self.predict_widget.native)

        # add buttons
        self.predict_button = QPushButton("Predict!", self)
        self.predict_button.clicked.connect(self.predict_worker)
        collapsable_predict_widget.addWidget(self.predict_button)

        layout.addWidget(collapsable_predict_widget)

        # Save widget(Collapsable)
        collapsable_save_widget = QCollapsible(
            "Save Model: expand for options:", self
        )
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
    def training_parameters(self) -> GunpowderParameters:
        parameters = GunpowderParameters(
            self.advanced_widget.lsds.lsd_sigma.value,
            self.advanced_widget.intensities.scale_min.value,
            self.advanced_widget.intensities.scale_max.value,
            self.advanced_widget.intensities.shift_min.value,
            self.advanced_widget.intensities.shift_max.value,
            self.advanced_widget.intensities.noise_mean.value,
            self.advanced_widget.intensities.noise_var.value,
            self.advanced_widget.spatial.elastic_control_spacing.value,
            self.advanced_widget.spatial.elastic_control_sigma.value,
            self.advanced_widget.spatial.zoom_min.value,
            self.advanced_widget.spatial.zoom_max.value,
            self.advanced_widget.spatial.rotation.value,
            self.advanced_widget.spatial.mirror.value,
            self.advanced_widget.spatial.transpose.value,
            self.advanced_widget.training.num_cpus.value,
            self.advanced_widget.training.batch_size.value,
        )
        return parameters

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
            self.disable_buttons(predict=True)
        else:
            if self.__training_generator is not None:
                self.__training_generator.send("stop")
            self.train_button.setText("Train!")
            self.disable_buttons(snapshot=True, async_predict=True)

    @contextmanager
    def build_pipeline(
        self,
        raw,
        gt,
        mask,
        parameters: GunpowderParameters,
        affs_high_inter_object: bool = False,
    ):
        with build_pipeline(
            raw, gt, mask, self.model, parameters, affs_high_inter_object
        ) as pipeline:
            yield pipeline

    def reset_training_state(self, keep_stats=False):
        if self.__training_generator is not None:
            self.__training_generator.quit()
        self.__training_generator = None
        if not keep_stats:
            self.iteration = 0
            self.__iterations = []
            self.__losses = []
            self.__val_iterations = []
            self.__val_losses = []
            if self.loss_plot is None:
                self.loss_plot = self.progress_plot.axes.plot(
                    self.__iterations, self.__losses, label="Training Loss"
                )[0]
                self.val_plot = self.progress_plot.axes.plot(
                    self.__val_iterations,
                    self.__val_losses,
                    label="Validation Loss",
                )[0]
                self.progress_plot.axes.legend()
                if self.model is not None:
                    self.progress_plot.axes.set_title(
                        f"{self.model.name} Training Progress"
                    )
                else:
                    self.progress_plot.axes.set_title(f"Training Progress")
                self.progress_plot.axes.set_xlabel("Iterations")
                self.progress_plot.axes.set_ylabel("Loss")
            self.update_progress_plot()

    def update_progress_plot(self):
        self.loss_plot.set_xdata(self.__iterations)
        self.loss_plot.set_ydata(self.__losses)
        self.val_plot.set_xdata(self.__val_iterations)
        self.val_plot.set_ydata(self.__val_losses)
        self.progress_plot.axes.relim()
        self.progress_plot.axes.autoscale_view()
        try:
            self.progress_plot.draw()
        except np.linalg.LinAlgError as e:
            # matplotlib seems to throw a LinAlgError on draw sometimes. Not sure
            # why yet. Seems to only happen when initializing models without any
            # layers loaded. No idea whats going wrong.
            # For now just avoid drawing. Seems to work as soon as there is data to plot
            pass

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
            self.training_parameters,
            iteration=self.iteration,
            affs_high_inter_object=self.model.config.get(
                "affs_high_inter_label", False
            ),
        )
        self.__training_generator.yielded.connect(self.on_yield)
        self.__training_generator.returned.connect(self.on_return)
        self.__training_generator.start()

        # all buttons are enabled while the training loop is running
        self.disable_buttons(predict=True)

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

    def predict_worker(self):
        predict_worker = self.predict()
        predict_worker.start()
        predict_worker.returned.connect(self.add_layers)

    @thread_worker
    def predict(self):
        """
        Predict on data provided through the predict widget. Not necessarily the
        same as the training data.
        """

        outputs = self.model.outputs
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
                f"{self.model.name} only provides: {output_names}"
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

        offsets = self.model.config["mws"]["offsets"]
        ndim = len(offsets[0])
        spatial_axes = self.spatial_dims(ndim)
        predictions = self._predict(
            self.model, self.predict_widget.raw.value.data[:], offsets
        )
        # Generate affinities and keep the offsets as metadata
        prediction_layers = [
            (
                predictions[affs_index],
                {
                    "name": "Affinities",
                    "metadata": {
                        "offsets": offsets,
                        "high_inter_label": self.model.config.get(
                            "affs_high_inter_label", False
                        ),
                    },
                    "axes": (
                        "channel",
                        *spatial_axes,
                    ),
                    "overwrite": True,
                },
                "image",
            ),
        ]
        if lsds:
            prediction_layers.append(
                (
                    predictions[lsd_index],
                    {
                        "name": "LSDs",
                        "metadata": {},
                        "axes": (
                            "channel",
                            *spatial_axes,
                        ),
                        "overwrite": True,
                    },
                    "image",
                ),
            )
        if fgbg:
            prediction_layers.append(
                (
                    predictions[fgbg_index],
                    {
                        "name": "fgbg",
                        "metadata": {},
                        "axes": (
                            "channel",
                            *spatial_axes,
                        ),
                        "overwrite": True,
                    },
                    "image",
                ),
            )
        return prediction_layers

    def _predict(self, model, raw_data, offsets):
        ndim = len(offsets[0])

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
                f"{self.model.name} only provides: {output_names}"
            )
        try:
            lsd_index = output_names.index("lsds")
            lsds = True
        except ValueError:
            lsds = False
            lsd_index = None
        try:
            fgbg_index = output_names.index("fgbg")
            fgbg = True
        except ValueError:
            fgbg = False
            fgbg_index = None

        # add batch and potentially channel dimensions
        assert len(raw_data.shape) >= ndim, (
            f"raw data has {len(raw_data.shape)} dimensions but "
            f"should have {ndim} spatial dimensions"
        )
        while len(raw_data.shape) < ndim + 2:
            raw_data = raw_data.reshape((1, *raw_data.shape))

        with create_prediction_pipeline(
            bioimageio_model=model, devices=["cuda"]
        ) as pp:
            # [0] to access first input array/output array
            pred_data = DataArray(raw_data, dims=tuple(pp.input_specs[0].axes))
            outputs = list(
                predict_with_tiling(pp, pred_data, True, verbose=True)
            )

        affs = outputs[affs_index].values

        result = [None for _ in output_names]

        # remove batch dimensions
        pred_data = pred_data.squeeze()
        affs = affs.squeeze()
        result[affs_index] = affs
        if lsds:
            lsds = outputs[lsd_index].values
            lsds = lsds.squeeze()
            result[lsd_index] = lsds
        if fgbg:
            fgbg = outputs[fgbg_index].values
            fgbg = fgbg.squeeze(0)
            result[fgbg_index] = fgbg

        # assert result is as expected
        assert (
            pred_data.ndim == ndim
        ), f"Raw has dims: {pred_data.ndim}, but expected: {ndim}"
        assert (
            affs.ndim == ndim + 1
        ), f"Affs have dims: {affs.ndim}, but expected: {ndim+1}"
        assert (
            affs.shape[0] == len(offsets) or affs.shape[0] == len(offsets) + 1
        ), (
            f"Number of affinity channels ({affs.shape[0]}) "
            f"does not match number of offsets ({len(offsets)})"
        )

        return tuple(result)

    def save(self):
        """
        Save model to file
        """

        # get architecture source
        def get_architecture_source():
            raw_resource = bioimageio.core.load_raw_resource_description(
                self.__rdf
            )
            model_source = raw_resource.weights[
                "pytorch_state_dict"
            ].architecture
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
            [
                {"name": prep.name, "kwargs": prep.kwargs}
                for prep in inp.preprocessing
            ]
            for inp in self.model.inputs
            if inp.preprocessing != missing
        ]
        postprocessing = [
            [
                {"name": post.name, "kwargs": post.kwargs}
                for post in outp.postprocessing
            ]
            if outp.postprocessing != missing
            else None
            for outp in self.model.outputs
        ]
        citations = [
            {
                k: v
                for k, v in dataclasses.asdict(citation).items()
                if v != missing
            }
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
        gt = layer_choice_widget(
            viewer, annotation=napari.layers.Labels, name="gt"
        )
        mask = layer_choice_widget(
            viewer, annotation=Optional[napari.layers.Labels], name="mask"
        )

        train_widget = Container(widgets=[raw, gt, mask])

        return train_widget

    def create_advanced_widget(self, viewer):
        # inputs:
        default_parameters = GunpowderParameters()
        lsd_sigma = create_widget(
            annotation=int,
            name="lsd_sigma",
            label="sigma",
            value=default_parameters.lsd_sigma,
        )
        scale_min = create_widget(
            annotation=float,
            name="scale_min",
            label="Scale min",
            value=default_parameters.intensity_scale_min,
            options={"min": 0},
        )
        scale_max = create_widget(
            annotation=float,
            name="scale_max",
            label="Scale max",
            value=default_parameters.intensity_scale_max,
            options={"min": 0},
        )
        shift_min = create_widget(
            annotation=float,
            name="shift_min",
            label="Shift min",
            value=default_parameters.intensity_shift_min,
            options={
                "min": -1,
                "max": 1,
            },
        )
        shift_max = create_widget(
            annotation=float,
            name="shift_max",
            label="Shift max",
            value=default_parameters.intensity_shift_max,
            options={
                "min": -1,
                "max": 1,
            },
        )
        noise_mean = create_widget(
            annotation=float,
            name="noise_mean",
            label="Noise mean",
            value=default_parameters.gausian_noise_mean,
        )
        noise_var = create_widget(
            annotation=float,
            name="noise_var",
            label="Noise var",
            value=default_parameters.gausian_noise_var,
        )
        elastic_control_spacing = create_widget(
            annotation=int,
            name="elastic_control_spacing",
            label="Elastic control spacing",
            value=default_parameters.elastic_control_point_spacing,
        )
        elastic_control_sigma = create_widget(
            annotation=int,
            name="elastic_control_sigma",
            label="Elastic control sigma",
            value=default_parameters.elastic_control_point_sigma,
        )
        zoom_min = create_widget(
            annotation=float,
            name="zoom_min",
            label="Zoom min",
            value=default_parameters.zoom_min,
            options={"min": 0},
        )
        zoom_max = create_widget(
            annotation=float,
            name="zoom_max",
            label="Zoom max",
            value=default_parameters.zoom_max,
            options={"min": 0},
        )
        rotation = create_widget(
            annotation=bool,
            name="rotation",
            label="Rotation",
            value=default_parameters.rotation,
        )
        mirror = create_widget(
            annotation=bool,
            name="mirror",
            label="Mirror",
            value=default_parameters.mirror,
        )
        transpose = create_widget(
            annotation=bool,
            name="transpose",
            label="Transpose",
            value=default_parameters.transpose,
        )
        num_cpus = create_widget(
            annotation=int,
            name="num_cpus",
            label="Num CPUs",
            value=default_parameters.num_cpu_processes,
        )
        batch_size = create_widget(
            annotation=int,
            name="batch_size",
            label="Batch Size",
            value=default_parameters.batch_size,
        )
        containers = [
            Container(
                name="lsds",
                label='<a href="https://localshapedescriptors.github.io"><font color=white>LSDs</font></a>',
                widgets=[
                    lsd_sigma,
                ],
            ),
            Container(
                name="intensities",
                label="Intensity Augmentations",
                widgets=[
                    scale_min,
                    scale_max,
                    shift_min,
                    shift_max,
                    noise_mean,
                    noise_var,
                ],
            ),
            Container(
                name="spatial",
                label="Spatial Augmentations",
                widgets=[
                    elastic_control_spacing,
                    elastic_control_sigma,
                    zoom_min,
                    zoom_max,
                    rotation,
                    mirror,
                    transpose,
                ],
            ),
            Container(
                name="training",
                label="Training Parameters",
                widgets=[
                    num_cpus,
                    batch_size,
                ],
            ),
        ]

        advanced_widget = Container(widgets=containers)
        return advanced_widget

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
        iteration, loss, val_loss, *layers = step_data
        if len(layers) > 0:
            self.add_layers(layers)
        if iteration is not None and loss is not None:
            self.iteration = iteration
            self.__iterations.append(iteration)
            self.__losses.append(loss)
            if val_loss is not None:
                self.__val_iterations.append(iteration)
                self.__val_losses.append(val_loss)
            self.update_progress_plot()

    def on_return(self, weights_path: Path):
        """
        Update model to use provided returned weights
        """
        assert self.model is not None
        self.model.weights["pytorch_state_dict"].source = weights_path
        self.reset_training_state(keep_stats=True)
        self.disable_buttons(snapshot=True, async_predict=True)

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
            assert batch_dim in [
                -1,
                0,
            ], f"Batch dim must be first"
            if batch_dim == 0:
                data = data[0]

            try:
                # add to existing layer
                layer = self.viewer.layers[name]

                if overwrite:
                    layer.data = data.reshape(*data.shape)
                else:
                    # concatenate along batch dimension
                    layer.data = np.concatenate(
                        [
                            layer.data.reshape(-1, *data.shape),
                            data.reshape(-1, *data.shape).astype(
                                layer.data.dtype
                            ),
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
                    self.viewer.add_labels(
                        data.astype(int), name=name, **metadata
                    )

    @thread_worker
    def train_affinities(
        self,
        raw,
        gt,
        mask,
        parameters,
        iteration=0,
        affs_high_inter_object: bool = False,
    ):

        if self.model is None:
            raise ValueError(
                "Please load a model either from your filesystem or a url"
            )

        model = deepcopy(self.model)

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
                f"{self.model.name} only provides: {output_names}"
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

        # constants
        offsets = model.config["mws"]["offsets"]
        ndim = len(offsets[0])
        spatial_axes = self.spatial_dims(ndim)

        # extract torch model from bioimageio format
        torch_module = get_torch_module(model)

        # define Loss function and Optimizer TODO: make options available as choices?
        lsd_loss_func = torch.nn.MSELoss()
        # aff_loss_func = torch.nn.BCEWithLogitsLoss()
        aff_loss_func = torch.nn.BCELoss()
        fgbg_loss_func = torch.nn.MSELoss()  # TODO: add support for DiceLoss
        optimizer = torch.optim.Adam(params=torch_module.parameters())

        # TODO: How to display profiling stats
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        torch_module = torch_module.to(device)
        torch_module.train()

        # prepare data for full volume prediction
        raw_data = raw.data
        # add batch dimension
        raw_data = raw_data.reshape((1, *raw_data.shape))

        # Train loop:
        with self.build_pipeline(
            raw, gt, mask, parameters, affs_high_inter_object
        ) as pipeline:
            mode = yield (None, None, None)
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

                    predictions = tuple(
                        self._predict(model, raw_data, offsets)
                    )

                    # Generate affinities and keep the offsets as metadata
                    prediction_layers = [
                        (
                            predictions[affs_index],
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
                    ]
                    if lsds:
                        prediction_layers.append(
                            (
                                predictions[lsd_index],
                                {
                                    "name": "LSDs(online)",
                                    "metadata": {},
                                    "axes": (
                                        "channel",
                                        *spatial_axes,
                                    ),
                                },
                                "image",
                            ),
                        )
                    if fgbg:
                        prediction_layers.append(
                            (
                                predictions[fgbg_index],
                                {
                                    "name": "fgbg(online)",
                                    "metadata": {},
                                    "axes": (
                                        "channel",
                                        *spatial_axes,
                                    ),
                                },
                                "image",
                            ),
                        )
                    mode = yield (None, None, None, *prediction_layers)
                elif mode is None or mode == "snapshot":
                    snapshot_iteration = mode == "snapshot"
                    val_loss = None

                    if iteration % self._validation_interval == 0:
                        val_arrays = pipeline.next_validation()
                        val_tensors = [
                            torch.as_tensor(array, device=device).float()
                            for array, _, _ in val_arrays
                        ]
                        (
                            val_raw,
                            val_aff_target,
                            val_aff_mask,
                            *val_optional_arrays,
                        ) = val_tensors
                        if lsds:
                            (
                                val_lsd_target,
                                val_lsd_mask,
                                *val_optional_arrays,
                            ) = val_optional_arrays
                        if fgbg:
                            (
                                val_fgbg_target,
                                val_fgbg_mask,
                            ) = val_optional_arrays
                        val_outputs = tuple(torch_module(val_raw))

                        val_affs_loss = aff_loss_func(
                            val_outputs[affs_index][-len(offsets) :]
                            * val_aff_mask,
                            val_aff_target * val_aff_mask,
                        )

                        val_losses = [val_affs_loss]
                        if lsds:
                            val_lsd_loss = lsd_loss_func(
                                val_outputs[lsd_index] * val_lsd_mask,
                                val_lsd_target * val_lsd_mask,
                            )
                            val_losses.append(val_lsd_loss)
                        if fgbg:
                            val_fgbg_loss = fgbg_loss_func(
                                val_outputs[fgbg_index] * val_fgbg_mask,
                                val_fgbg_target * val_fgbg_mask,
                            )
                            val_losses.append(val_fgbg_loss)

                        val_loss = sum(val_losses)

                    # fetch data:
                    arrays, snapshot_arrays = pipeline.next(snapshot_iteration)

                    tensors = [
                        torch.as_tensor(array, device=device).float()
                        for array, _, _ in arrays
                    ]
                    raw, aff_target, aff_mask, *optional_arrays = tensors
                    if lsds:
                        (
                            lsd_target,
                            lsd_mask,
                            *optional_arrays,
                        ) = optional_arrays
                    if fgbg:
                        fgbg_target, fgbg_mask = optional_arrays
                    outputs = tuple(torch_module(raw))

                    affs_loss = aff_loss_func(
                        outputs[affs_index][-len(offsets) :] * aff_mask,
                        aff_target * aff_mask,
                    )

                    losses = [affs_loss]
                    if lsds:
                        lsd_loss = lsd_loss_func(
                            outputs[lsd_index] * lsd_mask,
                            lsd_target * lsd_mask,
                        )
                        losses.append(lsd_loss)
                    if fgbg:
                        fgbg_loss = fgbg_loss_func(
                            outputs[fgbg_index] * fgbg_mask,
                            fgbg_target * fgbg_mask,
                        )
                        losses.append(fgbg_loss)

                    loss = sum(losses)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    iteration += 1

                    if snapshot_iteration:
                        pred_arrays = []
                        pred_arrays.append(
                            (
                                outputs[affs_index].detach().cpu().numpy(),
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
                                    outputs[lsd_index].detach().cpu().numpy(),
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
                        if fgbg:
                            pred_arrays.append(
                                (
                                    outputs[fgbg_index].detach().cpu().numpy(),
                                    {
                                        "name": "sample_fgbg_pred",
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
                            loss.detach().cpu().item(),
                            val_loss.detach().cpu().item()
                            if val_loss is not None
                            else None,
                            *arrays,
                            *snapshot_arrays,
                            *pred_arrays,
                        )
                    else:
                        mode = yield (
                            iteration,
                            loss.detach().cpu().item(),
                            val_loss.detach().cpu().item()
                            if val_loss is not None
                            else None,
                        )
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
