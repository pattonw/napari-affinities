from napari_affinities import (
    train_affinities_widget,
    predict_affinities_widget,
    mutex_watershed_widget,
)
import numpy as np


def test_train_affinities_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    my_widget = train_affinities_widget()

    my_widget(viewer.layers[0])


def test_predict_affinities_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    my_widget = predict_affinities_widget()

    my_widget(viewer.layers[0])


def test_mutex_watershed_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    my_widget = mutex_watershed_widget()

    my_widget(viewer.layers[0])
