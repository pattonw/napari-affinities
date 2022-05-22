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
            },
            "image",
        ),
        (
            container["volumes/gt_labels"][:],
            {
                "name": "Labels",
            },
            "labels",
        ),
        (
            container["volumes/gt_tanh"][:],
            {
                "name": "TanH",
            },
            "image",
        ),
        (
            container["volumes/gt_fgbg"][:],
            {
                "name": "FG/BG",
            },
            "labels",
        ),
    ]


def sample_lightsheet():
    container = zarr.open(LIGHTSHEET_ZARR, "r")
    return [
        (
            container["volumes/raw"][:] / 255,
            {
                "name": "Raw",
            },
            "image",
        ),
    ]
