#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

setup(install_requires=[
    "lsd @ git+https://github.com/pattonw/lsd.git@no-convenience-imports",
    "numpy",
    "zarr",
    "magicgui",
    "bioimageio.core",
    "gunpowder @ git+https://github.com/funkey/gunpowder.git@patch-1.2.3",
    "affogato",
    "matplotlib",
])
