from magicgui import magic_factory
import napari


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
    raw: napari.layers.Image,
):
    pass


@magic_factory
def mutex_watershed_widget(
    affinities: napari.layers.Image,
    constraints: napari.layers.Image,
):
    pass
