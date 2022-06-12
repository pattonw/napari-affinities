from napari_affinities.widgets.affinities import ModelWidget

import napari

import logging

# logging.basicConfig(level=logging.DEBUG)


viewer = napari.Viewer()
widget = ModelWidget(viewer)
viewer.window.add_dock_widget(widget)
napari.run()

