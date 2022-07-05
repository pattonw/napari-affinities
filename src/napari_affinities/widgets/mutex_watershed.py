from matplotlib import interactive
import napari
from napari.types import LabelsData, LayerDataTuple
from napari.layers import Labels, Image

from affogato.segmentation import MWSGridGraph, compute_mws_clustering

import numpy as np
from typing import Optional

from magicgui import magic_factory, widgets

import napari
from napari.qt.threading import FunctionWorker, thread_worker

def toggle_interactivity_callback(widget, interactive_callbacks):
    def callback(live):
        # remove old callbacks
        seed_layer = widget.seeds.value
        while len(interactive_callbacks) > 0:
            old_cb = interactive_callbacks.pop()
            seed_layer.events.set_data.disconnect(old_cb)
        if live:
            # add callback for seed_layer change
            if seed_layer is not None:
                def change(*args, **kwargs):
                    # this sets off the auto run
                    widget.toggle.value = 1 - widget.toggle.value

                cb = seed_layer.events.set_data.connect(change)
                interactive_callbacks.append(cb)
    return callback

def add_interactive_callback(widget, interactive_callbacks):
    def callback(seed_layer):
        if seed_layer is not None:

            def change(*args, **kwargs):
                # this sets off the auto run
                widget.toggle.value = 1 - widget.toggle.value

            # remove old callbacks
            while len(interactive_callbacks) > 0:
                old_cb = interactive_callbacks.pop()
                seed_layer.events.set_data.disconnect(old_cb)
            if widget.live.value:
                # add new callback
                cb = seed_layer.events.set_data.connect(change)
                interactive_callbacks.append(cb)
    return callback


def init(widget):
    interactive_callbacks = []
    widget.seeds.changed.connect(
        add_interactive_callback(widget, interactive_callbacks)
    )
    widget.live.changed.connect(
        toggle_interactivity_callback(widget, interactive_callbacks)
    )


@magic_factory(
    toggle={"visible": False},
    auto_call=True,
    widget_init=init,
    call_button=True,
)
def mutex_watershed_widget(
    affinities: Image,
    seeds: Optional[Labels],
    mask: Optional[Labels],
    live: bool = False,
    toggle: int = 1,
) -> FunctionWorker[LayerDataTuple]:
    if affinities is None or "offsets" not in affinities.metadata:
        raise ValueError("Please provide affinities with offset metadata!")
    assert "offsets" in affinities.metadata, f"{affinities.metadata}"
    offsets = affinities.metadata["offsets"]
    affs = affinities.data
    invert_affinities = affinities.metadata.get("high_inter_label", False)

    @thread_worker(connect={"returned": lambda: None})
    def async_mutex_watershed(seeds: LabelsData) -> LayerDataTuple:
        shape = affs.shape[1:]
        if seeds is not None:
            assert (
                shape == seeds.data.shape[-len(shape) :]
            ), f"Got shape {seeds.data.shape[-len(shape):]} for seeds but expected {shape}"
            assert len(seeds.shape) <= len(shape) + 1
            seeds = seeds[0] if len(seeds.shape) == len(shape) + 1 else seeds
        if mask is not None:
            assert (
                mask.data.shape[0] == 1
            ), "Mask should only have 1 channel but has multiple!"
            assert (
                shape == mask.data.shape[1:]
            ), f"Got shape {seeds.data.shape} for mask but expected {shape}"

        grid_graph = MWSGridGraph(shape)
        if seeds is not None:
            grid_graph.update_seeds(seeds.data)

        ndim = len(offsets[0])

        grid_graph.add_attractive_seed_edges = True
        neighbor_affs, lr_affs = (
            np.require(affs[:ndim], requirements="C"),
            np.require(affs[ndim:], requirements="C"),
        )

        # assuming affinities are 1 between voxels that belong together and
        # 0 if they are not part of the same object. Invert if the other way
        # around.
        # neighbors_affs should be high for objects that belong together
        # lr_affs is the oposite
        if invert_affinities:
            neighbor_affs = 1 - neighbor_affs
        else:
            lr_affs = 1 - lr_affs

        uvs, weights = grid_graph.compute_nh_and_weights(
            neighbor_affs, offsets[:ndim]
        )

        grid_graph.add_attractive_seed_edges = False
        mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
            lr_affs,
            offsets[ndim:],
            [4] * ndim,
            randomize_strides=True,
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

    return async_mutex_watershed(seeds.data if seeds is not None else None)
