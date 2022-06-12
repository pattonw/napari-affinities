import zarr

EPITHELIAL_ZARR = "sample_data/per01_100.zarr"
LIGHTSHEET_ZARR = "sample_data/lightsheet_nuclei.zarr"


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
    container = zarr.open(LIGHTSHEET_ZARR, "r")
    return [
        (
            container["volumes/raw"][100:260, 200:360, 400:560] / 255,
            {
                "name": "Raw",
                "metadata": {"axes": ["z", "y", "x"]},
            },
            "image",
        ),
        (
            container["volumes/seg"][:],
            {
                "name": "Segmentation",
                "metadata": {"axes": ["z", "y", "x"]},
            },
            "labels",
        ),
    ]
