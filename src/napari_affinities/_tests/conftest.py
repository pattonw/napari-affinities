import logging
import warnings

import pytest

from bioimageio.core import export_resource_package
from bioimageio.spec import __version__ as bioimageio_spec_version

logger = logging.getLogger(__name__)
warnings.warn(f"testing with bioimageio.spec {bioimageio_spec_version}")

# test models for various frameworks
affinity_models = [
    "epithelial_affs",
    "epithelial_lsds"
]


model_sources = {
    "epithelial_affs": (
        "https://raw.githubusercontent.com/"
        "pattonw/model-specs/main/epithelial_affinities/rdf.yaml"
    ),
    "epithelial_lsds": (
        "https://raw.githubusercontent.com/"
        "pattonw/model-specs/main/epithelial_lsds/rdf.yaml"
    ),
}

try:
    import torch

    torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
    logger.warning(f"detected torch version {torch_version}.x")
except ImportError:
    torch = None
    torch_version = None
skip_torch = torch is None

# load all model packages we need for testing
load_model_packages = set()
if not skip_torch:
    load_model_packages |= set(affinity_models)


def pytest_configure():

    # load all model packages used in tests
    pytest.model_packages = {
        name: export_resource_package(model_sources[name])
        for name in load_model_packages
    }


@pytest.fixture(params=[] if skip_torch else affinity_models)
def models(request):
    return pytest.model_packages[request.param]
