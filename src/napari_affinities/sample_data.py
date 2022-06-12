import zarr
import h5py

from pathlib import Path

EPITHELIAL_ZARR = Path(__file__).parent / "sample_data/per01_100.zarr"
LIGHTSHEET_H5 = Path(__file__).parent / "sample_data/lightsheet_nuclei_test_data"


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
    ]


def sample_lightsheet():
    container = h5py.File(LIGHTSHEET_H5)
    return [
        (
            container["raw"][:] / 255,
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
