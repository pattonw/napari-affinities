def test_open(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.open_sample(plugin="napari-affinities", sample="epithelial_sample.0")
    viewer.open_sample(plugin="napari-affinities", sample="lightsheet_sample.0")
