# -*- coding: utf-8 -*-
from promebuilder import gen_metadata, setup
from setuptools import find_packages

METADATA = gen_metadata(
    name="promlib",
    description="PromLib package",
    email="alberto.cordioli@prometeia.com",
    keywords="lab",
    url="https://github.com/prometeia/promlib"
)

if __name__ == '__main__':
    setup(METADATA)
