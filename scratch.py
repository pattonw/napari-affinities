from src.napari_affinities.widgets.affinities import ModelWidget

import napari


viewer = napari.Viewer()
widget = ModelWidget(viewer)
viewer.window.add_dock_widget(widget)
napari.run()
