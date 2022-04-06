from napari_affinities.widgets.affinities import (
    ModelWidget
)
import numpy as np


def test_train_affinities_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    raw = viewer.add_image(np.random.random((100, 100)))
    gt = viewer.add_labels(np.random.random((100, 100), dtype=np.int32))
    mask = viewer.add_labels(np.random.random((100, 100), dtype=np.int32))
    lsds = viewer.add_image(np.random.random((100, 100)))
    model_widget = ModelWidget(viewer)
    my_widget = model_widget.create_train_widget(viewer)
    my_widget(raw, gt, mask, lsds)


# def test_predict_affinities_widget(make_napari_viewer, capsys):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))

#     my_widget = predict_affinities_widget()

#     my_widget(viewer.layers[0])


# def test_mutex_watershed_widget(make_napari_viewer, capsys):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))

#     my_widget = mutex_watershed_widget()

#     my_widget(viewer.layers[0])
