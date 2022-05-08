from magicgui import magic_factory
import napari

from affogato.segmentation import MWSGridGraph, compute_mws_clustering

import numpy as np
from typing import Optional


@magic_factory
def mutex_watershed_widget(
    affinities: napari.layers.Image,
    seeds: Optional[napari.layers.Labels],
    mask: Optional[napari.layers.Labels],
    invert_affinities: bool,
) -> napari.types.LayerDataTuple:
    # TODO: Add Keybindings

    # TODO:
    # beta slider
    # live update checkbox (simple layer onchange eventlistener similar to
    # the layer dropdown choice reloader on the model widget)

    # Assumptions:
    # Affinities must come with "offsets" in its metadata.
    assert "offsets" in affinities.metadata, f"{affinities.metadata}"
    # seeds and mask must be same size as affinities if provided
    shape = affinities.data.shape[1:]
    if seeds is not None:
        assert (
            seeds.data.shape[0] == 1
        ), "Seeds should only have 1 channel but has multiple!"
        assert (
            shape == seeds.data.shape[1:]
        ), f"Got shape {seeds.data.shape[1:]} for seeds but expected {shape}"
        seeds = seeds.data
    if mask is not None:
        assert (
            mask.data.shape[0] == 1
        ), "Mask should only have 1 channel but has multiple!"
        assert (
            shape == mask.data.shape[1:]
        ), f"Got shape {seeds.data.shape} for mask but expected {shape}"

    grid_graph = MWSGridGraph(shape)
    if mask is not None:
        grid_graph.set_mask(mask.data)
    if seeds is not None:
        grid_graph.update_seeds(seeds.data)

    offsets = affinities.metadata["offsets"]
    ndim = len(offsets[0])

    grid_graph.add_attractive_seed_edges = True
    neighbor_affs, lr_affs = (
        np.require(affinities.data[:ndim], requirements="C"),
        np.require(affinities.data[ndim:], requirements="C"),
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
    uvs, weights = grid_graph.compute_nh_and_weights(neighbor_affs, offsets[:ndim])

    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
        lr_affs,
        offsets[ndim:],
        [1] * ndim,
        randomize_strides=False,
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
