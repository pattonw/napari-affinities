import zarr

EPITHELIAL_ZARR = "sample_data/per01_100.zarr"

def sample_epithelial_raw():
    container = zarr.open(EPITHELIAL_ZARR, "r")
    return [(container["volumes/raw"][:],)]

def sample_epithelial_affinities():
    container = zarr.open(EPITHELIAL_ZARR, "r")
    return [(container["volumes/gt_affs"][:],)]

def sample_epithelial_labels():
    container = zarr.open(EPITHELIAL_ZARR, "r")
    return [(container["volumes/gt_labels"][:],)]

def sample_epithelial_tanh():
    container = zarr.open(EPITHELIAL_ZARR, "r")
    return [(container["volumes/gt_tanh"][:],)]

def sample_epithelial_fgbg():
    container = zarr.open(EPITHELIAL_ZARR, "r")
    return [(container["volumes/gt_fgbg"][:],)]
    