from napari_affinities.widgets.affinities import ModelWidget
import numpy as np


def test_affinities_widget(make_napari_viewer, capsys, models):
    viewer = make_napari_viewer()
    model_widget = ModelWidget(viewer)
    
    model_widget.load_model(models)

    # TODO: test train step

    # TODO: test predict on sample data