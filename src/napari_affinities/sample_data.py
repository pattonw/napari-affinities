import zarr
import h5py


EPITHELIAL_ZARR = "sample_data/per01_100.zarr"
LIGHTSHEET_H5 = "sample_data/lightsheet_nuclei_test_data"


def sample_epithelial():
    container = zarr.open(EPITHELIAL_ZARR, "r")
    return [
        (
            container["volumes/raw"][:],
            {
                "name": "Raw",
                "metadata": {"axes": ["y", "x"]},
            },
            "image",
        ),
        (
            container["volumes/gt_labels"][:],
            {
                "name": "Labels",
                "metadata": {"axes": ["y", "x"]},
            },
            "labels",
        ),
        (
            container["volumes/gt_tanh"][:],
            {
                "name": "TanH",
                "metadata": {"axes": ["y", "x"]},
            },
            "image",
        ),
        (
            container["volumes/gt_fgbg"][:],
            {
                "name": "FG/BG",
                "metadata": {"axes": ["y", "x"]},
            },
            "labels",
        ),
    ]


def sample_lightsheet():
    container = h5py.File(LIGHTSHEET_H5)
    return [
        (
            container["raw"][100:260, 200:360, 400:560] / 255,
            {
                "name": "Raw",
                "metadata": {"axes": ["z", "y", "x"]},
            },
            "image",
        ),
        (
            container["seg"],
            {
                "name": "Segmentation",
                "metadata": {"axes": ["z", "y", "x"]},
            },
            "labels",
        ),
    ]
