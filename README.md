# napari-affinities

[![License](https://img.shields.io/pypi/l/napari-affinities.svg?color=green)](https://github.com/pattonw/napari-affinities/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-affinities.svg?color=green)](https://pypi.org/project/napari-affinities)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-affinities.svg?color=green)](https://python.org)
[![tests](https://github.com/pattonw/napari-affinities/workflows/tests/badge.svg)](https://github.com/pattonw/napari-affinities/actions)
[![codecov](https://codecov.io/gh/pattonw/napari-affinities/branch/main/graph/badge.svg)](https://codecov.io/gh/pattonw/napari-affinities)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-affinities)](https://napari-hub.org/plugins/napari-affinities)

A plugin for creating, visualizing, and processing affinities

---

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

You can install `napari-affinities` via [pip]:

    pip install napari-affinities

To install latest development version :

    pip install git+https://github.com/pattonw/napari-affinities.git

Install torch according to your system [(follow the instructions here)](https://pytorch.org/get-started/locally/). For example with cuda 10.2 available, run:

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

Then install all requirements:

    conda install -c conda-forge affogato
    pip install -r requirements.txt

### Download example model:

#### 2D:

[epithelial example model](https://oc.embl.de/index.php/s/zfWMKu7HoQnSJLs)
Place the model zip file wherever you want. You can open it in the plugin with the "load from file" button.

#### 3D

[lightsheet example model](https://owncloud.gwdg.de/index.php/s/LsShICsOcilqPRs)
Unpack the tar file into test data (`lightsheet_nuclei_test_data` (an hdf5 file)) and model (`LightsheetNucleusSegmentation.zip` (a bioimageio model)).
Move the data into sample_data which will enable you to load the "Lightsheet Sample" data in napari.
Place the model zip file anywhere you want. You can open it in the plugin with the "load from file" button.

## Use

Requirements for the model:

1. Bioimageio packaged pytorch model
2. Outputs with names "affinities", "fgbg"(optional) or "lsds"(optional)
   - if these names are not used, it will be assumed that the outputs are affinities, fgbg, then lsds in that order

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-affinities" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[mit]: http://opensource.org/licenses/MIT
[bsd-3]: http://opensource.org/licenses/BSD-3-Clause
[gnu gpl v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[gnu lgpl v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[apache software license 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[mozilla public license 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/pattonw/napari-affinities/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[pypi]: https://pypi.org/
